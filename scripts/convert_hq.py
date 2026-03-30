#!/usr/bin/env python3
"""
High-Quality INT4 Conversion for Qwen3-Coder-30B-A3B-Instruct
==============================================================

Converts the model with MoE-optimized parameters to improve long-context
intelligence for agentic tool-calling workflows.

Key improvements over the default (data-free INT4_ASYM gs=128):
  - Protect MoE router layers (mlp.gate) from INT4 quantization → FP16/INT8
  - Smaller group_size (64) for finer quantization granularity
  - Data-aware AWQ + scale estimation for accuracy
  - Sensitivity-based mixed precision (most sensitive layers → INT8)

Tiers:
  tier3  — optimum-cli with AWQ + scale_estimation + gs=64 (fastest, no router protection)
  tier1  — NNCF Python API with router protection + AWQ + gs=64 (highest quality)

Usage:
    python convert_hq.py --tier tier3
    python convert_hq.py --tier tier1 --subset-size 64
    python convert_hq.py --inspect             # inspect existing model layers
    python convert_hq.py --inspect --model-dir "C:\\working\\models\\...\\INT4-HQ"
"""

import argparse
import os
import sys
import time
import shutil
import subprocess
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
BASE_MODEL_DIR = Path(r"C:\working\models\Qwen3-Coder-30B-A3B-Instruct")

OUTPUT_DIRS = {
    "tier3": BASE_MODEL_DIR / "INT4-HQ-T3",
    "tier1": BASE_MODEL_DIR / "INT4-HQ",
}

BASELINE_DIR = BASE_MODEL_DIR / "INT4"

# MoE Router layer pattern — these are the 48 routing layers (one per transformer layer)
# Node naming in OV IR: self.model.layers.{N}.mlp.gate.weight
# MatMul naming: __module.model.layers.{N}.mlp.gate/ov_ext::linear/MatMul
MOE_ROUTER_PATTERN = r".*mlp\.gate.*"


# ── Tier 3: optimum-cli (no router protection, but AWQ + gs=64) ───────────────

def run_tier3(args):
    """
    Tier 3: Use optimum-cli with advanced flags.
    Does NOT protect MoE router layers (CLI doesn't expose ignored_scope).
    But adds: AWQ, scale_estimation, group_size=64, ratio=0.9, data-aware.
    """
    output_dir = args.output_dir or OUTPUT_DIRS["tier3"]
    output_path = Path(output_dir)

    if output_path.exists() and any(output_path.glob("*.xml")):
        if not args.force:
            print(f"[INFO] Model already exists at {output_dir}. Use --force to overwrite.")
            return
        print(f"[WARN] Removing existing model at {output_dir}")
        shutil.rmtree(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    # Find optimum-cli
    venv_dir = os.environ.get("VIRTUAL_ENV", "")
    if venv_dir:
        cli_path = os.path.join(venv_dir, "Scripts", "optimum-cli")
    else:
        cli_path = str(Path(sys.executable).parent / "optimum-cli.exe")
    if not Path(cli_path).exists():
        cli_path = "optimum-cli"

    cmd = [
        cli_path, "export", "openvino",
        "--model", MODEL_ID,
        "--task", "text-generation-with-past",
        "--weight-format", "int4",
        "--group-size", str(args.group_size),
        "--ratio", str(args.ratio),
        "--awq",
        "--scale-estimation",
        "--dataset", args.dataset,
        "--num-samples", str(args.subset_size),
        "--sensitivity-metric", "max_activation_variance",
        "--all-layers",
        str(output_dir),
    ]

    if args.backup_precision:
        cmd.insert(-1, "--backup-precision")
        cmd.insert(-1, args.backup_precision)

    print(f"\n{'='*80}")
    print(f"  TIER 3: optimum-cli High-Quality Conversion")
    print(f"{'='*80}")
    print(f"  Model:              {MODEL_ID}")
    print(f"  Output:             {output_dir}")
    print(f"  Mode:               INT4_ASYM")
    print(f"  Group size:         {args.group_size}")
    print(f"  Ratio:              {args.ratio}")
    print(f"  AWQ:                True")
    print(f"  Scale estimation:   True")
    print(f"  Dataset:            {args.dataset}")
    print(f"  Num samples:        {args.subset_size}")
    print(f"  Sensitivity metric: max_activation_variance")
    print(f"  All layers:         True")
    print(f"  Router protection:  NO (CLI limitation)")
    print(f"{'='*80}")
    print(f"\n  Command: {' '.join(cmd)}\n")

    start = time.time()
    result = subprocess.run(cmd, env=os.environ.copy())
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n[ERROR] Conversion failed with return code {result.returncode}")
        sys.exit(1)

    total_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
    print(f"\n[INFO] Conversion completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"[INFO] Model size: {total_size / (1024**3):.2f} GB")


# ── Tier 1: NNCF Python API (with router protection) ──────────────────────────

def run_tier1(args):
    """
    Tier 1: Use NNCF compress_weights() directly.
    Protects MoE router layers via ignored_scope.
    Applies AWQ + scale_estimation + data-aware compression.
    """
    import openvino as ov
    import nncf
    from nncf import compress_weights, Dataset
    from nncf.parameters import CompressWeightsMode, SensitivityMetric, BackupMode
    from transformers import AutoTokenizer

    output_dir = args.output_dir or OUTPUT_DIRS["tier1"]
    output_path = Path(output_dir)

    if output_path.exists() and any(output_path.glob("*.xml")):
        if not args.force:
            print(f"[INFO] Model already exists at {output_dir}. Use --force to overwrite.")
            return
        print(f"[WARN] Removing existing model at {output_dir}")
        shutil.rmtree(output_dir)

    # ── Step 1: Ensure we have an FP16 / uncompressed OV IR ──
    # We need a non-quantized OV model to re-compress with NNCF.
    # Option A: Export fresh from HF (slow but guaranteed)
    # Option B: Use pre-converted model's graph but that's already quantized.
    # We'll use optimum-intel Python API to export FP16 first.

    fp16_dir = BASE_MODEL_DIR / "FP16-temp"

    if fp16_dir.exists() and any(fp16_dir.glob("*.xml")):
        print(f"[INFO] FP16 model found at {fp16_dir}, reusing.")
    else:
        print(f"\n{'='*80}")
        print(f"  Step 1: Exporting FP16 model from HuggingFace")
        print(f"  This downloads ~17 GB and converts to OV IR (FP16)")
        print(f"{'='*80}\n")

        fp16_dir.mkdir(parents=True, exist_ok=True)

        # Use optimum-cli for FP16 export (no quantization)
        venv_dir = os.environ.get("VIRTUAL_ENV", "")
        if venv_dir:
            cli_path = os.path.join(venv_dir, "Scripts", "optimum-cli")
        else:
            cli_path = str(Path(sys.executable).parent / "optimum-cli.exe")

        cmd = [
            cli_path, "export", "openvino",
            "--model", MODEL_ID,
            "--task", "text-generation-with-past",
            "--weight-format", "fp16",
            str(fp16_dir),
        ]

        print(f"  Command: {' '.join(cmd)}\n")
        start = time.time()
        result = subprocess.run(cmd, env=os.environ.copy())
        elapsed = time.time() - start

        if result.returncode != 0:
            print(f"\n[ERROR] FP16 export failed")
            sys.exit(1)

        total_size = sum(f.stat().st_size for f in fp16_dir.rglob("*") if f.is_file())
        print(f"\n[INFO] FP16 export completed in {elapsed:.0f}s, size: {total_size / (1024**3):.2f} GB")

    # ── Step 2: Load tokenizer and prepare calibration dataset ──
    print(f"\n{'='*80}")
    print(f"  Step 2: Preparing calibration dataset")
    print(f"{'='*80}\n")

    tokenizer = AutoTokenizer.from_pretrained(str(fp16_dir))

    # Prepare calibration data — use wikitext2 or custom tool-calling prompts
    calibration_data = _prepare_calibration_dataset(
        tokenizer, dataset_name=args.dataset, num_samples=args.subset_size
    )
    nncf_dataset = Dataset(calibration_data)

    # ── Step 3: Load FP16 OV IR and compress with NNCF ──
    print(f"\n{'='*80}")
    print(f"  Step 3: Compressing with NNCF (MoE-optimized)")
    print(f"{'='*80}")
    print(f"  Mode:               INT4_ASYM")
    print(f"  Group size:         {args.group_size}")
    print(f"  Ratio:              {args.ratio}")
    print(f"  AWQ:                True")
    print(f"  Scale estimation:   True")
    print(f"  Router protection:  YES (ignored_scope: {MOE_ROUTER_PATTERN})")
    print(f"  Sensitivity metric: MAX_ACTIVATION_VARIANCE")
    print(f"  Backup mode:        {args.backup_precision or 'INT8_ASYM'}")
    print(f"  Subset size:        {args.subset_size}")
    print(f"  All layers:         False (embeddings/lm_head → INT8)")
    print()

    core = ov.Core()
    ov_model = core.read_model(str(fp16_dir / "openvino_model.xml"))

    # Determine backup mode
    backup = BackupMode.INT8_ASYM
    if args.backup_precision == "int8_sym":
        backup = BackupMode.INT8_SYM
    elif args.backup_precision == "none":
        backup = BackupMode.NONE

    start = time.time()
    compressed_model = compress_weights(
        ov_model,
        mode=CompressWeightsMode.INT4_ASYM,
        group_size=args.group_size,
        ratio=args.ratio,
        ignored_scope=nncf.IgnoredScope(patterns=[MOE_ROUTER_PATTERN]),
        dataset=nncf_dataset,
        sensitivity_metric=SensitivityMetric.MAX_ACTIVATION_VARIANCE,
        subset_size=args.subset_size,
        awq=True,
        scale_estimation=True,
        gptq=False,
        lora_correction=False,
        backup_mode=backup,
        all_layers=False,  # Keep embeddings/lm_head at backup precision
    )
    compress_time = time.time() - start
    print(f"\n[INFO] NNCF compression completed in {compress_time:.0f}s ({compress_time/60:.1f} min)")

    # ── Step 4: Save compressed model ──
    print(f"\n{'='*80}")
    print(f"  Step 4: Saving compressed model to {output_dir}")
    print(f"{'='*80}\n")

    output_path.mkdir(parents=True, exist_ok=True)
    ov.save_model(compressed_model, str(output_path / "openvino_model.xml"))

    # Copy tokenizer and config files from the FP16 export
    for f in fp16_dir.iterdir():
        if f.name.startswith("openvino_model"):
            continue  # Skip model files (we just saved our own)
        dest = output_path / f.name
        if f.is_file() and not dest.exists():
            shutil.copy2(f, dest)
        elif f.is_dir() and not dest.exists():
            shutil.copytree(f, dest)

    total_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
    print(f"[INFO] Saved compressed model: {total_size / (1024**3):.2f} GB")
    print(f"[INFO] Output: {output_dir}")

    # ── Step 5: Verify router layers are not INT4 ──
    print(f"\n{'='*80}")
    print(f"  Step 5: Verification — checking router layer precision")
    print(f"{'='*80}\n")
    _inspect_model(str(output_path / "openvino_model.xml"), filter_pattern="mlp.gate")


def _prepare_calibration_dataset(tokenizer, dataset_name="wikitext2", num_samples=128):
    """Prepare calibration dataset for NNCF data-aware compression."""

    if dataset_name == "tool_calling":
        # Custom tool-calling calibration data
        return _prepare_tool_calling_dataset(tokenizer, num_samples)

    # Default: use wikitext2 via HuggingFace datasets
    from datasets import load_dataset

    print(f"  Loading dataset: {dataset_name}")
    if dataset_name == "wikitext2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text_column = "text"
    else:
        dataset = load_dataset(dataset_name, split="train")
        # Try common text column names
        text_column = None
        for col in ["text", "content", "sentence", "input"]:
            if col in dataset.column_names:
                text_column = col
                break
        if text_column is None:
            text_column = dataset.column_names[0]

    print(f"  Text column: {text_column}")
    print(f"  Preparing {num_samples} samples...")

    # Tokenize and create input dicts
    calibration_data = []
    for item in dataset:
        text = item[text_column]
        if not text or len(text.strip()) < 50:
            continue

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        calibration_data.append(dict(inputs))

        if len(calibration_data) >= num_samples:
            break

    print(f"  Collected {len(calibration_data)} calibration samples")
    return calibration_data


def _prepare_tool_calling_dataset(tokenizer, num_samples=64):
    """Create a calibration dataset focused on tool-calling patterns."""

    tool_definitions = '''You are an AI assistant with access to the following tools:

<tools>
[
  {"name": "search_web", "description": "Search the web for information", "parameters": {"query": {"type": "string"}}},
  {"name": "read_file", "description": "Read contents of a file", "parameters": {"path": {"type": "string"}}},
  {"name": "write_file", "description": "Write content to a file", "parameters": {"path": {"type": "string"}, "content": {"type": "string"}}},
  {"name": "run_command", "description": "Execute a shell command", "parameters": {"command": {"type": "string"}}},
  {"name": "list_directory", "description": "List contents of a directory", "parameters": {"path": {"type": "string"}}},
  {"name": "get_weather", "description": "Get weather for a location", "parameters": {"location": {"type": "string"}}},
  {"name": "calculate", "description": "Evaluate a mathematical expression", "parameters": {"expression": {"type": "string"}}},
  {"name": "create_task", "description": "Create a new task/issue", "parameters": {"title": {"type": "string"}, "description": {"type": "string"}, "priority": {"type": "string"}}}
]
</tools>

When you need to use a tool, respond with:
<tool_call>
{"name": "tool_name", "arguments": {"param1": "value1"}}
</tool_call>
'''

    conversations = [
        "Search for the latest OpenVINO release notes and summarize them.",
        "Read the file at /home/user/project/main.py and find any bugs.",
        "Create a shell script that backs up the database, then run it.",
        "List all Python files in the project directory, then read each one.",
        "What's the weather in Tokyo? Also calculate 2^32 - 1.",
        "Write a unit test file for the calculator module, save it, and run pytest.",
        "Search for Python best practices for error handling, then create a task to refactor our code.",
        "Read the config.json file, update the database URL, and write it back.",
    ]

    calibration_data = []
    for conv in conversations:
        for _ in range(max(1, num_samples // len(conversations))):
            # Simulate multi-turn with system prompt + user message
            messages = [
                {"role": "system", "content": tool_definitions},
                {"role": "user", "content": conv},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
            calibration_data.append(dict(inputs))

            if len(calibration_data) >= num_samples:
                break
        if len(calibration_data) >= num_samples:
            break

    print(f"  Collected {len(calibration_data)} tool-calling calibration samples")
    return calibration_data


# ── Model Inspection ───────────────────────────────────────────────────────────

def _inspect_model(model_path: str, filter_pattern: str = None):
    """Inspect quantization precision of model layers."""
    import openvino as ov

    core = ov.Core()
    model = core.read_model(model_path)

    print(f"  Model: {model_path}")
    print()

    # Count precision distribution
    precision_counts = {}
    filtered_nodes = []

    for op in model.get_ordered_ops():
        if op.get_type_name() != "Constant":
            continue

        name = op.get_friendly_name()

        # Skip scale/zero_point helper constants
        if "/scale" in name or "/zero_point" in name:
            continue

        et = str(op.get_output_element_type(0))
        pshape = str(op.get_output_partial_shape(0))

        # Track precision distribution for weight-like constants
        if "weight" in name:
            precision_counts[et] = precision_counts.get(et, 0) + 1

            if filter_pattern and filter_pattern in name:
                filtered_nodes.append((name, et, pshape))

    # Print overall distribution
    print("  Weight Precision Distribution:")
    for et, count in sorted(precision_counts.items()):
        print(f"    {et}: {count} tensors")

    # Print filtered nodes
    if filter_pattern and filtered_nodes:
        print(f"\n  Filtered nodes matching '{filter_pattern}':")
        for name, et, pshape in filtered_nodes:
            print(f"    {name}: {et} {pshape}")
    elif filter_pattern:
        print(f"\n  No weight nodes matching '{filter_pattern}'")


def inspect_model(args):
    """Inspect an existing model's quantization state."""
    model_dir = Path(args.model_dir)
    xml_path = model_dir / "openvino_model.xml"

    if not xml_path.exists():
        print(f"[ERROR] Model not found at {xml_path}")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"  MODEL INSPECTION")
    print(f"{'='*80}\n")

    _inspect_model(str(xml_path), filter_pattern=args.filter)

    # Also show key layer types
    import openvino as ov
    core = ov.Core()
    model = core.read_model(str(xml_path))

    categories = {
        "embedding": "embed_tokens",
        "lm_head": "lm_head",
        "attention": "self_attn",
        "moe_router": "mlp.gate",
        "shared_expert": "shared_expert",
        "experts": "mlp.experts",
    }

    print(f"\n  Key Layer Precision Summary:")
    for cat_name, pattern in categories.items():
        u4 = u8 = f16 = bf16 = other = 0
        for op in model.get_ordered_ops():
            if op.get_type_name() != "Constant":
                continue
            name = op.get_friendly_name()
            if pattern not in name or "/scale" in name or "/zero_point" in name:
                continue
            if "weight" not in name:
                continue
            et = str(op.get_output_element_type(0))
            if "uint4" in et or "int4" in et:
                u4 += 1
            elif "uint8" in et or "int8" in et:
                u8 += 1
            elif "float16" in et:
                f16 += 1
            elif "bfloat16" in et:
                bf16 += 1
            else:
                other += 1
        total = u4 + u8 + f16 + bf16 + other
        if total > 0:
            parts = []
            if u4: parts.append(f"INT4:{u4}")
            if u8: parts.append(f"INT8:{u8}")
            if f16: parts.append(f"FP16:{f16}")
            if bf16: parts.append(f"BF16:{bf16}")
            if other: parts.append(f"other:{other}")
            print(f"    {cat_name:20s}: {', '.join(parts)} ({total} total)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="High-Quality INT4 Conversion for Qwen3-Coder-30B-A3B-Instruct",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--tier", choices=["tier3", "tier1"], default=None,
                        help="Conversion tier (tier3=CLI, tier1=NNCF API with router protection)")
    parser.add_argument("--inspect", action="store_true",
                        help="Inspect an existing model's quantization state")
    parser.add_argument("--model-dir", type=str, default=str(BASELINE_DIR),
                        help=f"Model directory for inspection (default: {BASELINE_DIR})")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--group-size", type=int, default=64,
                        help="Quantization group size (default: 64)")
    parser.add_argument("--ratio", type=float, default=0.9,
                        help="INT4 vs INT8 ratio (default: 0.9)")
    parser.add_argument("--dataset", type=str, default="wikitext2",
                        choices=["wikitext2", "tool_calling"],
                        help="Calibration dataset (default: wikitext2)")
    parser.add_argument("--subset-size", type=int, default=128,
                        help="Number of calibration samples (default: 128)")
    parser.add_argument("--backup-precision", type=str, default=None,
                        choices=["int8_sym", "int8_asym", "none"],
                        help="Backup precision for non-INT4 layers")
    parser.add_argument("--filter", type=str, default=None,
                        help="Filter pattern for --inspect mode")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output directory")

    args = parser.parse_args()

    if args.inspect:
        inspect_model(args)
        return

    if not args.tier:
        parser.error("Please specify --tier (tier3 or tier1) or --inspect")

    if args.tier == "tier3":
        run_tier3(args)
    elif args.tier == "tier1":
        run_tier1(args)


if __name__ == "__main__":
    main()
