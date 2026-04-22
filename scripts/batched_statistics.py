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
  Partition the ~242 statistic targets into N batches (default 4, ~60 each).
  For each batch:
    1. Create a fresh StatisticsAggregator with only that batch's targets
    2. ``collect_statistics()`` — only ~60 Result nodes → much less memory
    3. ``dump_statistics()`` to a temporary directory (NNCF safetensors format)
    4. Free memory (del aggregator, gc.collect)
  After all batches:
    5. Merge all batch directories into one statistics directory
    6. Pass directory as ``statistics_path`` to ``compress_weights()``

Memory impact:
  All 242 outputs → ~79 GB peak @ 1024 tokens
  60 outputs/batch → ~63 GB peak @ 1024 tokens → supports seq_len ≈ 4096

Usage:
  Called from convert_hq.py's run_tier1() when --layer-batches is specified.

    from batched_statistics import collect_statistics_batched

    stats_dir = collect_statistics_batched(
        ov_model=ov_model,
        nncf_dataset=nncf_dataset,
        subset_size=32,
        stats_dir="D:/nvme-temp/stats",
        num_batches=4,
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
    num_batches: int = 4,
    nvme_offload_dir: Optional[str] = None,
) -> str:
    """
    Collect NNCF weight compression statistics in layer-batched passes.

    This is a drop-in replacement for NNCF's ``cache_weight_compression_statistics()``
    that limits the number of simultaneous Result nodes in the transformed model,
    dramatically reducing peak memory.

    :param ov_model: OpenVINO model (FP16, not yet quantized).
    :param nncf_dataset: NNCF Dataset wrapping calibration data.
    :param subset_size: Number of calibration samples per batch.
    :param stats_dir: Output directory for merged statistics (safetensors).
    :param num_batches: Number of batches to split targets into (default: 4).
    :param nvme_offload_dir: Optional override for temporary batch storage.
                             If None, batch temps are stored under stats_dir.
    :return: Path to the merged statistics directory (= stats_dir).
    """
    from nncf.common.factory import StatisticsAggregatorFactory, build_graph
    from nncf.openvino.graph.model_utils import remove_friendly_name_duplicates
    from nncf.quantization.algorithms.weight_compression.algorithm import (
        WeightCompression,
        get_weight_compression_configuration,
    )
    from nncf.quantization.statistics_caching import (
        register_all_statistics,
    )
    from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer

    stats_path = Path(stats_dir)

    # If stats already exist, skip collection entirely
    if stats_path.exists() and (stats_path / "statistics_metadata.json").exists():
        print(f"  [BatchedStats] Statistics directory already exists at {stats_dir}, reusing.")
        return stats_dir

    print(f"\n{'='*80}")
    print(f"  BATCHED STATISTICS COLLECTION")
    print(f"{'='*80}")
    print(f"  Batches:     {num_batches}")
    print(f"  Subset size: {subset_size}")
    print(f"  Output:      {stats_dir}")
    print()

    # ── Step 1: Build NNCF graph and get all statistic points ──
    print("  [Step 1] Analyzing model graph and statistic points...")
    t0 = time.time()

    # NNCF requires deduplicated friendly names
    model_clean = remove_friendly_name_duplicates(ov_model)
    graph = build_graph(model_clean)

    # Create compression algorithm with superset config (same as
    # cache_weight_compression_statistics) to collect ALL possible stats
    config = get_weight_compression_configuration(
        awq=True, scale_estimation=True, lora_correction=True
    )
    compression_algo = WeightCompression(**config, subset_size=subset_size)
    compression_algo.set_backend_entity(model_clean)

    # Register all statistics (main + mixed precision) on a master aggregator
    master_aggregator = StatisticsAggregatorFactory.create(model_clean, nncf_dataset)
    register_all_statistics(
        master_aggregator, model_clean, graph, subset_size, compression_algo
    )

    all_points = master_aggregator.statistic_points
    total_targets = len(all_points)
    total_collectors = sum(
        1 for _ in all_points.get_tensor_collectors()
    )

    print(f"  Total target nodes:     {total_targets}")
    print(f"  Total tensor collectors: {total_collectors}")
    print(f"  Analysis time:          {time.time() - t0:.1f}s")

    # ── Step 2: Partition targets into batches by layer number ──
    print(f"\n  [Step 2] Partitioning {total_targets} targets into {num_batches} batches...")
    batches = _partition_statistic_points(all_points, num_batches)

    for i, batch in enumerate(batches):
        batch_count = sum(1 for _ in batch.get_tensor_collectors())
        layer_info = _get_batch_layer_range(batch)
        print(f"    Batch {i+1}: {len(batch)} target nodes, {batch_count} collectors ({layer_info})")

    # Free the master aggregator's heavyweight objects (keep stat points for reference)
    del master_aggregator
    gc.collect()

    # ── Step 3: Collect statistics for each batch ──
    batch_temp_base = Path(nvme_offload_dir) if nvme_offload_dir else stats_path / "_batches"
    batch_dirs = []
    total_start = time.time()

    for i, batch_points in enumerate(batches):
        batch_dir = batch_temp_base / f"batch_{i}"
        batch_dirs.append(batch_dir)

        print(f"\n  [Step 3.{i+1}] Collecting batch {i+1}/{num_batches} "
              f"({len(batch_points)} targets)...")

        t_batch = time.time()

        # Create fresh aggregator + algorithm for this batch
        # (fresh objects = fresh collectors with no accumulated data)
        batch_algo = WeightCompression(**config, subset_size=subset_size)
        batch_algo.set_backend_entity(model_clean)

        batch_aggregator = StatisticsAggregatorFactory.create(model_clean, nncf_dataset)

        # Register only this batch's statistic points
        # We must create FRESH stat points for each batch (not reuse from master)
        # because collectors accumulate data during collection
        _register_batch_statistics(
            batch_aggregator, model_clean, graph, subset_size,
            batch_algo, batch_points
        )

        # Run forward passes — only this batch's Result nodes are added to the model
        batch_aggregator.collect_statistics(model_clean, graph)

        # Dump to safetensors
        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_aggregator.dump_statistics(batch_dir)

        elapsed = time.time() - t_batch
        print(f"           Completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")

        # Aggressive memory cleanup between batches
        del batch_aggregator, batch_algo
        gc.collect()

    total_elapsed = time.time() - total_start
    print(f"\n  [Step 3] All batches collected in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

    # ── Step 4: Merge batch directories into final stats directory ──
    print(f"\n  [Step 4] Merging {num_batches} batch directories...")
    _merge_statistics_dirs(batch_dirs, stats_path)

    # Clean up batch temp directories
    if batch_temp_base != stats_path:
        shutil.rmtree(batch_temp_base, ignore_errors=True)
    else:
        # Remove _batches subdir from within stats_path
        batches_subdir = stats_path / "_batches"
        if batches_subdir.exists():
            shutil.rmtree(batches_subdir, ignore_errors=True)

    print(f"  Merged statistics saved to: {stats_dir}")
    print(f"  Total files: {sum(1 for _ in stats_path.iterdir())}")

    return stats_dir


# ── Internal helpers ───────────────────────────────────────────────────────────

def _partition_statistic_points(
    points,
    num_batches: int,
) -> list:
    """
    Partition a StatisticPointsContainer into N batches by layer number.

    The container keys are target node names like:
      ``__module.model.layers.5.self_attn.q_proj/ov_ext::linear/MatMul``
    We extract the layer number and group sequentially, then split into batches.

    Non-layer nodes (embeddings, lm_head) are included in the first batch.
    """
    from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer

    layer_pattern = re.compile(r"layers[._](\d+)")

    # Map each key to its layer number
    key_layers = {}
    for key in points.keys():
        match = layer_pattern.search(key)
        if match:
            key_layers[key] = int(match.group(1))
        else:
            key_layers[key] = -1  # Non-layer nodes (embed, lm_head)

    # Sort keys by layer number for sequential batching
    sorted_keys = sorted(points.keys(), key=lambda k: key_layers.get(k, -1))

    # Split into approximately equal batches
    batch_size = max(1, len(sorted_keys) // num_batches)
    batches = []

    for i in range(num_batches):
        start = i * batch_size
        if i == num_batches - 1:
            # Last batch gets remainder
            end = len(sorted_keys)
        else:
            end = start + batch_size

        batch = StatisticPointsContainer()
        for key in sorted_keys[start:end]:
            batch[key] = points[key]

        if batch:  # Don't add empty batches
            batches.append(batch)

    return batches


def _get_batch_layer_range(batch_points) -> str:
    """Get human-readable layer range string for a batch."""
    layer_pattern = re.compile(r"layers[._](\d+)")
    layers = set()
    non_layer = 0

    for key in batch_points.keys():
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


def _register_batch_statistics(
    aggregator,
    model,
    graph,
    subset_size: int,
    compression_algo,
    batch_points,
):
    """
    Register statistic points for a specific batch.

    Instead of registering ALL stat points (which would create too many Result
    nodes during collection), we only register the points whose target node
    names match this batch's keys.

    We replicate the full statistics registration (main + mixed precision)
    but filter to only keep targets in this batch.
    """
    from nncf.quantization.statistics_caching import register_all_statistics
    from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer

    # Register ALL stat points on a temporary aggregator
    temp_aggregator_points = StatisticPointsContainer()

    # Get the batch's target node names
    batch_keys = set(batch_points.keys())

    # Create a fresh superset of all stat points
    from nncf.common.factory import StatisticsAggregatorFactory
    temp_agg = StatisticsAggregatorFactory.create(model, aggregator.dataset)
    register_all_statistics(temp_agg, model, graph, subset_size, compression_algo)

    # Filter to only this batch's target nodes
    for key, stat_point_list in temp_agg.statistic_points.items():
        if key in batch_keys:
            temp_aggregator_points[key] = stat_point_list

    # Register the filtered points on the actual aggregator
    aggregator.register_statistic_points(temp_aggregator_points)

    # Clean up
    del temp_agg
    gc.collect()


def _merge_statistics_dirs(batch_dirs: list, output_dir: Path) -> None:
    """
    Merge multiple NNCF batch statistics directories into one.

    Each batch directory contains:
      - statistics_metadata.json with {"mapping": {sanitized: original}, ...}
      - Individual .safetensors files

    We combine all safetensors files and merge the metadata mappings.
    Since batches are disjoint by target node, there should be no filename
    conflicts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_mapping = {}
    merged_metadata = {}

    for batch_dir in batch_dirs:
        metadata_path = batch_dir / "statistics_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Missing statistics_metadata.json in batch directory: {batch_dir}"
            )

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Copy non-mapping metadata from first batch
        if not merged_metadata:
            merged_metadata = {k: v for k, v in metadata.items() if k != "mapping"}

        # Copy safetensors files and accumulate mapping
        for saved_name, original_name in metadata.get("mapping", {}).items():
            # safetensors files might be saved_name.safetensors or just saved_name
            src_with_ext = batch_dir / f"{saved_name}.safetensors"
            src_without_ext = batch_dir / saved_name

            if src_with_ext.exists():
                src = src_with_ext
                dst = output_dir / f"{saved_name}.safetensors"
            elif src_without_ext.exists():
                src = src_without_ext
                dst = output_dir / saved_name
            else:
                # Try glob matching
                matches = list(batch_dir.glob(f"{saved_name}*"))
                if matches:
                    src = matches[0]
                    dst = output_dir / src.name
                else:
                    print(f"  [WARN] Missing statistics file for {saved_name} in {batch_dir}")
                    continue

            # Check for conflicts (shouldn't happen with disjoint batches)
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
    print("Use via convert_hq.py --tier tier1 --layer-batches 4")
