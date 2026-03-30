#!/usr/bin/env python3
"""
Dequantize MoE Router Weights: INT4 → FP16
============================================

Surgically converts only the 48 MoE router (mlp.gate) weights from INT4 to FP16
in an existing OpenVINO IR model by editing the BIN/XML directly.

Strategy: Replace the binary data for each router's weight/scale/zero_point
so the existing dequant chain (Convert→Subtract→Multiply) becomes a no-op:
  - weight: dequantized FP16 values (was uint4)
  - zero_point: all zeros  (Subtract becomes no-op)
  - scale: all 1.0         (Multiply becomes no-op)

Size impact: ~18 MB increase (negligible vs 16.9 GB model)

Usage:
    python dequant_routers.py
    python dequant_routers.py --model-dir "C:\\working\\models\\...\\INT4"
    python dequant_routers.py --model-dir "..." --output-dir "..." --dry-run
"""

import argparse
import re
import shutil
import struct
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def unpack_uint4(packed: np.ndarray, shape: tuple) -> np.ndarray:
    """Unpack uint4 packed data (2 values per byte) to uint8 array."""
    total_elements = 1
    for d in shape:
        total_elements *= d

    result = np.empty(total_elements, dtype=np.uint8)
    result[0::2] = packed & 0x0F        # low nibble = first element
    result[1::2] = (packed >> 4) & 0x0F  # high nibble = second element
    return result.reshape(shape)


def parse_shape(shape_str: str) -> tuple:
    """Parse '128, 16, 128' → (128, 16, 128)."""
    return tuple(int(x.strip()) for x in shape_str.split(","))


def find_router_constants(xml_root):
    """Find all router-related constant layers in the XML.
    
    Returns dict: layer_idx → {weight: {id, offset, size, shape}, 
                                zero_point: {...}, scale: {...}}
    """
    routers = {}  # layer_idx → dict
    
    for layers_elem in xml_root.iter("layers"):
        for layer in layers_elem:
            name = layer.get("name", "")
            if "mlp.gate.weight" not in name:
                continue
            if layer.get("type") != "Const":
                continue
            
            # Parse layer index
            m = re.search(r"layers\.(\d+)\.mlp\.gate\.weight", name)
            if not m:
                continue
            layer_idx = int(m.group(1))
            
            data_elem = layer.find("data")
            if data_elem is None:
                continue
            
            info = {
                "layer_elem": layer,
                "data_elem": data_elem,
                "id": layer.get("id"),
                "name": name,
                "element_type": data_elem.get("element_type"),
                "shape": parse_shape(data_elem.get("shape")),
                "offset": int(data_elem.get("offset")),
                "size": int(data_elem.get("size")),
            }
            
            if layer_idx not in routers:
                routers[layer_idx] = {}
            
            if "/zero_point" in name:
                routers[layer_idx]["zero_point"] = info
            elif "/scale" in name:
                routers[layer_idx]["scale"] = info
            elif "/fq_weights" not in name and "/subtract" not in name:
                routers[layer_idx]["weight"] = info
    
    return routers


def dequantize_from_bin(bin_data: bytes, weight_info: dict, zp_info: dict, 
                         scale_info: dict) -> np.ndarray:
    """Read binary data and dequantize a router weight.
    
    Returns FP16 array of shape weight_info['shape'].
    """
    # Extract raw bytes
    w_bytes = bin_data[weight_info["offset"]:weight_info["offset"] + weight_info["size"]]
    zp_bytes = bin_data[zp_info["offset"]:zp_info["offset"] + zp_info["size"]]
    s_bytes = bin_data[scale_info["offset"]:scale_info["offset"] + scale_info["size"]]
    
    # Convert to numpy
    w_packed = np.frombuffer(w_bytes, dtype=np.uint8)
    zp_packed = np.frombuffer(zp_bytes, dtype=np.uint8)
    scale = np.frombuffer(s_bytes, dtype=np.float16).reshape(scale_info["shape"])
    
    # Unpack uint4
    weight_int = unpack_uint4(w_packed, weight_info["shape"]).astype(np.float32)
    zp_int = unpack_uint4(zp_packed, zp_info["shape"]).astype(np.float32)
    
    # Dequantize: fp = (uint4 - zero_point) * scale
    dequantized = (weight_int - zp_int) * scale.astype(np.float32)
    
    return dequantized.astype(np.float16)


def main():
    parser = argparse.ArgumentParser(
        description="Dequantize MoE router weights from INT4 to FP16",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-dir", type=str,
                        default=r"C:\working\models\Qwen3-Coder-30B-A3B-Instruct\INT4",
                        help="Input model directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: {model-dir}-RouterFP16)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only analyze, don't modify")
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    xml_path = model_dir / "openvino_model.xml"
    bin_path = model_dir / "openvino_model.bin"
    
    if not xml_path.exists():
        print(f"[ERROR] Model not found: {xml_path}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir) if args.output_dir else model_dir.parent / (model_dir.name + "-RouterFP16")
    
    print(f"\n{'='*80}")
    print(f"  MoE Router Dequantization: INT4 → FP16")
    print(f"{'='*80}")
    print(f"  Input:  {model_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Dry run: {args.dry_run}")
    print()
    
    # Parse XML
    print("[1/6] Parsing model XML...")
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    
    # Find router constants
    print("[2/6] Finding MoE router constants...")
    routers = find_router_constants(root)
    print(f"  Found {len(routers)} router layers")
    
    if not routers:
        print("[ERROR] No router constants found!")
        sys.exit(1)
    
    # Validate all routers have weight + zero_point + scale
    for idx in sorted(routers.keys()):
        r = routers[idx]
        if "weight" not in r or "zero_point" not in r or "scale" not in r:
            print(f"  [WARN] Layer {idx} missing components: {list(r.keys())}")
            del routers[idx]
    
    print(f"  Valid routers: {len(routers)}")
    r0 = routers[0]
    print(f"  Layer 0 weight: {r0['weight']['element_type']} {r0['weight']['shape']} "
          f"@ offset {r0['weight']['offset']}, size {r0['weight']['size']}")
    
    # Calculate size impact
    total_old = sum(r["weight"]["size"] + r["zero_point"]["size"] for r in routers.values())
    # New FP16 weight: same shape but 2 bytes/element instead of 0.5 bytes/element
    # New ZP and Scale remain the same size (ZP stays as u4 zeros, scale stays as f16)
    per_router_fp16 = 1
    for d in r0["weight"]["shape"]:
        per_router_fp16 *= d
    per_router_fp16 *= 2  # FP16 = 2 bytes per element
    total_new_weights = per_router_fp16 * len(routers)
    size_increase = total_new_weights - sum(r["weight"]["size"] for r in routers.values())
    
    print(f"\n  Current router weight data: {total_old / 1024:.0f} KB")
    print(f"  New FP16 weight data:       {total_new_weights / 1024:.0f} KB")
    print(f"  Size increase:              {size_increase / 1024 / 1024:.1f} MB")
    
    if args.dry_run:
        print("\n  [DRY RUN] No changes made.")
        return
    
    # Read entire BIN file
    print(f"\n[3/6] Reading binary data ({bin_path.stat().st_size / 1024**3:.2f} GB)...")
    t0 = time.perf_counter()
    with open(bin_path, "rb") as f:
        bin_data = f.read()
    print(f"  Read in {time.perf_counter() - t0:.1f}s")
    
    # Dequantize all routers and prepare new BIN data
    print("[4/6] Dequantizing router weights...")
    t0 = time.perf_counter()
    
    # We'll append new FP16 data at the end of the BIN
    new_data_offset = len(bin_data)
    new_bin_append = bytearray()
    
    for idx in sorted(routers.keys()):
        r = routers[idx]
        
        # Dequantize
        fp16_data = dequantize_from_bin(bin_data, r["weight"], r["zero_point"], r["scale"])
        fp16_bytes = fp16_data.tobytes()
        
        # Record where this router's new data will be
        router_new_offset = new_data_offset + len(new_bin_append)
        new_bin_append.extend(fp16_bytes)
        
        # Prepare scale = 1.0 (same shape as existing scale)
        scale_shape = r["scale"]["shape"]
        scale_ones = np.ones(scale_shape, dtype=np.float16)
        scale_bytes = scale_ones.tobytes()
        scale_new_offset = new_data_offset + len(new_bin_append)
        new_bin_append.extend(scale_bytes)
        
        # Store new offsets for XML update
        r["weight"]["new_offset"] = router_new_offset
        r["weight"]["new_size"] = len(fp16_bytes)
        r["weight"]["new_element_type"] = "f16"
        
        r["scale"]["new_offset"] = scale_new_offset
        r["scale"]["new_size"] = len(scale_bytes)
        
        # Zero point: set to zeros (keep same format u4, zeros in binary is all 0x00)
        zp_zeros = bytes(r["zero_point"]["size"])  # all zeros
        zp_new_offset = new_data_offset + len(new_bin_append)
        new_bin_append.extend(zp_zeros)
        
        r["zero_point"]["new_offset"] = zp_new_offset
        r["zero_point"]["new_size"] = len(zp_zeros)
        
        if (idx + 1) % 12 == 0 or idx == max(routers.keys()):
            print(f"  Processed {idx + 1}/{len(routers)} layers")
    
    print(f"  Dequantization completed in {time.perf_counter() - t0:.1f}s")
    print(f"  New data size: {len(new_bin_append) / 1024 / 1024:.1f} MB")
    
    # Update XML
    print("[5/6] Updating XML metadata...")
    for idx in sorted(routers.keys()):
        r = routers[idx]
        
        # Update weight constant
        w = r["weight"]
        w_data = w["data_elem"]
        w_data.set("element_type", w["new_element_type"])
        w_data.set("offset", str(w["new_offset"]))
        w_data.set("size", str(w["new_size"]))
        # Update shape string — keep same shape [128, 16, 128]
        
        # Update output port precision
        output_elem = w["layer_elem"].find("output")
        if output_elem is not None:
            port_elem = output_elem.find("port")
            if port_elem is not None:
                port_elem.set("precision", "FP16")
        
        # Update scale constant
        s = r["scale"]
        s_data = s["data_elem"]
        s_data.set("offset", str(s["new_offset"]))
        s_data.set("size", str(s["new_size"]))
        
        # Update zero_point constant
        zp = r["zero_point"]
        zp_data = zp["data_elem"]
        zp_data.set("offset", str(zp["new_offset"]))
        zp_data.set("size", str(zp["new_size"]))
    
    # Write output
    print(f"[6/6] Writing modified model to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy all files except XML and BIN
    for f in model_dir.iterdir():
        if f.name in ("openvino_model.xml", "openvino_model.bin"):
            continue
        dst = output_dir / f.name
        if f.is_file() and not dst.exists():
            shutil.copy2(f, dst)
        elif f.is_dir() and not dst.exists():
            shutil.copytree(f, dst)
    
    # Write new BIN (original + appended FP16 data)
    new_bin_path = output_dir / "openvino_model.bin"
    t0 = time.perf_counter()
    with open(new_bin_path, "wb") as f:
        f.write(bin_data)
        f.write(bytes(new_bin_append))
    print(f"  BIN written in {time.perf_counter() - t0:.1f}s")
    
    # Write new XML
    new_xml_path = output_dir / "openvino_model.xml"
    tree.write(str(new_xml_path), encoding="unicode", xml_declaration=True)
    print(f"  XML written")
    
    # Report
    old_size = bin_path.stat().st_size
    new_size = new_bin_path.stat().st_size
    print(f"\n{'='*80}")
    print(f"  DONE!")
    print(f"  Original model BIN: {old_size / 1024**3:.2f} GB")
    print(f"  Modified model BIN: {new_size / 1024**3:.2f} GB")
    print(f"  Size increase:      {(new_size - old_size) / 1024**2:.1f} MB")
    print(f"  Output: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
