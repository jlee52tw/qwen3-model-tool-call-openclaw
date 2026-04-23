#!/usr/bin/env python3
"""
Batched Statistics Collection for NNCF Weight Compression
==========================================================

Solves the NNCF OOM problem for large MoE models by collecting weight
compression statistics in batches instead of all at once.

Problem:
  NNCF's ``compress_weights()`` with ``scale_estimation=True`` or ``awq=True``
  runs ``StatisticsAggregator.collect_statistics()`` which:
    1. Clones the OV model adding ~242 Result nodes (one per MatMul target)
    2. Runs forward passes with ALL outputs alive simultaneously
    3. OpenVINO cannot reuse activation buffers across these extra outputs
  For Qwen3-Coder-30B-A3B (48 layers, 128 experts, MoE), this requires ~79 GB
  at seq_len=1024. At seq_len>1400, it OOMs on 96 GB RAM.

Solution:
  Partition the ~242 statistic targets into N batches (default 3, ~80 each).
  For each batch:
    1. Create a fresh StatisticsAggregator with only that batch's targets
    2. ``collect_statistics()`` — only ~80 Result nodes → much less memory
    3. ``dump_statistics()`` to a temporary directory (NNCF safetensors format)
    4. Free memory (del aggregator, gc.collect)
  After all batches:
    5. Merge all batch directories into one statistics directory
    6. Pass directory as ``statistics_path`` to ``compress_weights()``

Memory impact:
  All 242 outputs → ~79 GB peak @ 1024 tokens
  88 outputs/batch → ~65 GB peak @ 1024 tokens → 31 GB free RAM

Usage:
  Called from convert_hq.py's run_tier1() when --layer-batches is specified.

    from batched_statistics import collect_statistics_batched

    stats_dir = collect_statistics_batched(
        ov_model=ov_model,
        nncf_dataset=nncf_dataset,
        subset_size=8,
        stats_dir="path/to/stats",
        num_batches=3,
    )

    # Then pass to compress_weights via advanced_parameters:
    from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
    compressed = compress_weights(
        ov_model, ...,
        advanced_parameters=AdvancedCompressionParameters(statistics_path=str(stats_dir)),
    )
"""

import gc
import json
import re
import shutil
import time
from pathlib import Path
from typing import Optional

import openvino as ov


# ── Public API ─────────────────────────────────────────────────────────────────

def collect_statistics_batched(
    ov_model: ov.Model,
    nncf_dataset,
    subset_size: int,
    stats_dir: str,
    num_batches: int = 3,
    nvme_offload_dir: Optional[str] = None,
) -> str:
    """
    Collect NNCF weight compression statistics in layer-batched passes.

    Drop-in replacement for NNCF's ``cache_weight_compression_statistics()``
    that limits simultaneous Result nodes per inference pass.

    :param ov_model: OpenVINO model (FP16, not yet quantized).
    :param nncf_dataset: NNCF Dataset wrapping calibration data.
    :param subset_size: Number of calibration samples per batch.
    :param stats_dir: Output directory for merged statistics (safetensors).
    :param num_batches: Number of batches to split targets into.
    :param nvme_offload_dir: Optional override for temporary batch storage.
    :return: Path to the merged statistics directory (= stats_dir).
    """
    from nncf.common.factory import NNCFGraphFactory, StatisticsAggregatorFactory
    from nncf.openvino.graph.model_utils import remove_friendly_name_duplicates
    from nncf.quantization.algorithms.weight_compression.algorithm import (
        WeightCompression,
        get_weight_compression_configuration,
    )
    from nncf.quantization.statistics_caching import register_all_statistics

    stats_path = Path(stats_dir)

    # If stats already exist, skip collection entirely
    if stats_path.exists() and (stats_path / "statistics_metadata.json").exists():
        print(f"  [BatchedStats] Statistics already exist at {stats_dir}, reusing.")
        return stats_dir

    print(f"\n  BATCHED STATISTICS COLLECTION")
    print(f"  {'─' * 50}")
    print(f"  Batches:     {num_batches}")
    print(f"  Subset size: {subset_size}")
    print(f"  Output:      {stats_dir}")
    print()

    # ── Step 1: Build NNCF graph and discover all stat points ──
    print("  [Step 1] Analyzing model graph and statistic points...")
    t0 = time.time()

    # NNCF requires deduplicated friendly names
    model_for_graph = remove_friendly_name_duplicates(ov_model)
    graph = NNCFGraphFactory.create(model_for_graph)

    # Create weight compression algo with superset config (same as
    # cache_weight_compression_statistics) to discover ALL possible stat points
    config = get_weight_compression_configuration(
        awq=True, scale_estimation=True, lora_correction=True
    )
    master_algo = WeightCompression(**config, subset_size=subset_size)
    master_algo.set_backend_entity(model_for_graph)

    # Register stat points on a throwaway aggregator to discover targets.
    # Disable mixed precision stats (enable_mixed_precision=False) because:
    #   1. MoE expert activations are 3D: [128, seq_len, hidden_dim]
    #   2. Mixed precision hessian stats do x*x on those → 1+ GB per tensor → numpy OOM
    #   3. With ratio=1.0 (all-INT4), mixed precision sensitivity isn't needed
    discovery_agg = StatisticsAggregatorFactory.create(model_for_graph, nncf_dataset)
    register_all_statistics(
        discovery_agg, model_for_graph, graph, subset_size, master_algo,
        enable_mixed_precision=False,
    )

    # Extract target node names (keys of the StatisticPointsContainer)
    all_target_keys = list(discovery_agg.statistic_points.keys())
    total_targets = len(all_target_keys)

    print(f"  Total target nodes: {total_targets}")
    print(f"  Analysis time:      {time.time() - t0:.1f}s")

    # Free discovery objects (we'll create fresh ones per batch)
    del discovery_agg, master_algo
    gc.collect()

    # ── Step 2: Partition target keys into batches by layer number ──
    print(f"\n  [Step 2] Partitioning {total_targets} targets into {num_batches} batches...")
    key_batches = _partition_keys_by_layer(all_target_keys, num_batches)

    for i, keys in enumerate(key_batches):
        layer_info = _get_layer_range(keys)
        print(f"    Batch {i + 1}: {len(keys)} targets ({layer_info})")

    # ── Step 3: Collect statistics for each batch ──
    batch_temp_base = Path(nvme_offload_dir) if nvme_offload_dir else stats_path / "_batches"
    batch_dirs = []
    total_start = time.time()

    for i, batch_keys in enumerate(key_batches):
        batch_dir = batch_temp_base / f"batch_{i}"
        batch_dirs.append(batch_dir)
        batch_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  [Step 3.{i + 1}] Collecting batch {i + 1}/{num_batches} "
              f"({len(batch_keys)} targets)...")

        t_batch = time.time()
        batch_key_set = set(batch_keys)

        # Create FRESH compression algo and aggregator for each batch
        # (fresh collectors = no accumulated data from previous batches)
        batch_algo = WeightCompression(**config, subset_size=subset_size)
        batch_algo.set_backend_entity(model_for_graph)

        batch_agg = StatisticsAggregatorFactory.create(model_for_graph, nncf_dataset)

        # Register stat points (no mixed precision — avoids numpy OOM on MoE 3D tensors)
        register_all_statistics(
            batch_agg, model_for_graph, graph, subset_size, batch_algo,
            enable_mixed_precision=False,
        )

        # ...then REMOVE targets not in this batch.
        # This ensures only this batch's Result nodes get inserted during inference,
        # dramatically reducing peak memory.
        keys_to_remove = [k for k in list(batch_agg.statistic_points.keys())
                          if k not in batch_key_set]
        for k in keys_to_remove:
            del batch_agg.statistic_points[k]

        remaining = len(batch_agg.statistic_points)
        print(f"           Registered {remaining} targets (removed {len(keys_to_remove)})")

        # Run forward passes — only this batch's Result nodes added to model
        batch_agg.collect_statistics(model_for_graph, graph)

        # Dump batch statistics to safetensors
        batch_agg.dump_statistics(batch_dir)

        elapsed = time.time() - t_batch
        print(f"           Completed in {elapsed:.0f}s ({elapsed / 60:.1f} min)")

        # Report RAM usage if psutil is available
        try:
            import psutil
            ram = psutil.virtual_memory()
            print(f"           RAM: {ram.used / (1024 ** 3):.1f} GB used, "
                  f"{ram.available / (1024 ** 3):.1f} GB free")
        except ImportError:
            pass

        # Aggressive memory cleanup between batches
        del batch_agg, batch_algo
        gc.collect()

    total_elapsed = time.time() - total_start
    print(f"\n  [Step 3] All batches collected in {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)")

    # ── Step 4: Merge batch directories into final stats directory ──
    print(f"\n  [Step 4] Merging {len(batch_dirs)} batch directories...")
    _merge_statistics_dirs(batch_dirs, stats_path)

    # Clean up batch temp directories
    batches_subdir = stats_path / "_batches"
    if batches_subdir.exists():
        shutil.rmtree(batches_subdir, ignore_errors=True)
    if nvme_offload_dir and Path(nvme_offload_dir).exists():
        shutil.rmtree(nvme_offload_dir, ignore_errors=True)

    print(f"  Merged statistics saved to: {stats_dir}")
    stat_files = list(stats_path.glob("*.safetensors"))
    print(f"  Total safetensors files: {len(stat_files)}")

    return stats_dir


# ── Internal helpers ───────────────────────────────────────────────────────────

def _partition_keys_by_layer(keys: list, num_batches: int) -> list:
    """
    Partition target node keys into N batches by layer number.

    Keys look like:
      ``__module.model.layers.5.self_attn.q_proj/ov_ext::linear/MatMul``
    We extract the layer number and group sequentially, then split.
    Non-layer keys (embeddings, lm_head) go in the first batch.
    """
    layer_pattern = re.compile(r"layers[._](\d+)")

    # Map each key to its layer number (-1 for non-layer nodes)
    key_layers = {}
    for key in keys:
        match = layer_pattern.search(key)
        key_layers[key] = int(match.group(1)) if match else -1

    # Sort by layer number for sequential batching
    sorted_keys = sorted(keys, key=lambda k: key_layers[k])

    # Split into approximately equal batches
    batch_size = max(1, len(sorted_keys) // num_batches)
    batches = []

    for i in range(num_batches):
        start = i * batch_size
        end = len(sorted_keys) if i == num_batches - 1 else start + batch_size
        batch = sorted_keys[start:end]
        if batch:
            batches.append(batch)

    return batches


def _get_layer_range(keys: list) -> str:
    """Get human-readable layer range string for a batch of keys."""
    layer_pattern = re.compile(r"layers[._](\d+)")
    layers = set()
    non_layer = 0

    for key in keys:
        match = layer_pattern.search(key)
        if match:
            layers.add(int(match.group(1)))
        else:
            non_layer += 1

    parts = []
    if layers:
        parts.append(f"layers {min(layers)}-{max(layers)}")
    if non_layer:
        parts.append(f"{non_layer} non-layer nodes")

    return ", ".join(parts) if parts else "empty"


def _merge_statistics_dirs(batch_dirs: list, output_dir: Path) -> None:
    """
    Merge multiple NNCF batch statistics directories into one.

    Each batch directory contains:
      - statistics_metadata.json with {"mapping": {sanitized_name: original_name}}
      - Individual .safetensors files

    Batches are disjoint by target node, so no filename conflicts expected.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_mapping = {}
    merged_metadata = {}

    for batch_dir in batch_dirs:
        metadata_path = batch_dir / "statistics_metadata.json"
        if not metadata_path.exists():
            print(f"  [WARN] No statistics_metadata.json in {batch_dir}, skipping")
            continue

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Copy non-mapping metadata from first batch
        if not merged_metadata:
            merged_metadata = {k: v for k, v in metadata.items() if k != "mapping"}

        # Copy safetensors files and accumulate mapping
        for saved_name, original_name in metadata.get("mapping", {}).items():
            # Find the source file (try common naming patterns)
            candidates = [
                batch_dir / f"{saved_name}.safetensors",
                batch_dir / saved_name,
            ]

            src = None
            for candidate in candidates:
                if candidate.exists():
                    src = candidate
                    break

            if src is None:
                matches = list(batch_dir.glob(f"{saved_name}*"))
                if matches:
                    src = matches[0]

            if src is None:
                print(f"  [WARN] Missing statistics file for {saved_name} in {batch_dir}")
                continue

            dst = output_dir / src.name
            if saved_name in merged_mapping:
                print(f"  [WARN] Duplicate stat key {saved_name}, overwriting")

            shutil.copy2(src, dst)
            merged_mapping[saved_name] = original_name

    # Write merged metadata
    merged_metadata["mapping"] = merged_mapping
    with open(output_dir / "statistics_metadata.json", "w") as f:
        json.dump(merged_metadata, f, indent=2)

    print(f"  Merged {len(merged_mapping)} statistics entries from {len(batch_dirs)} batches")


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("batched_statistics.py — no standalone tests (requires NNCF + OV model)")
    print("Use via convert_hq.py --tier tier1 --layer-batches N")
