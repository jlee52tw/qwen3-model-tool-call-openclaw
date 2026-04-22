# 2026-04-22 Project Status Summary & NNCF NVMe Offloading Plan

## Project Overview

**Goal:** Improve long-context tool-calling accuracy for Qwen3-Coder-30B-A3B-Instruct (INT4) on Intel iGPU by using NNCF data-aware quantization with agentic calibration data.

**Hardware:** Intel Core Ultra + Arc B390 iGPU (76 GB override), 96 GB RAM, 225 GB page file, Windows 11

## Completed Work (Tasks 1–7)

| # | Task | Status | Key Result |
|---|------|--------|------------|
| 1 | Baseline benchmark suite | ✅ Done | NIAH 100%@64K, Easy 5.0/5@64K, **Hard 4.0/5@24K+** |
| 2 | Extended stress test (32K–64K) | ✅ Done | `read_file` specifically fails: 1/5@32K → 0/5@64K |
| 3 | Router dequantization experiment | ✅ Done | No improvement — routers are NOT the bottleneck |
| 4 | GPU compatibility testing | ✅ Done | Only INT4_ASYM gs=128 ratio=1.0 works; gs=64, ratio<1.0, AWQ all crash |
| 5 | Memory profiling for NNCF | ✅ Done | FP16 model=57GB, peak 79GB@1024tok, OOM above ~1400tok |
| 6 | `convert_hq.py` implementation | ✅ Done | Tier1 (NNCF API) + Tier3 (optimum-cli), FP16 monkey-patch |
| 7 | Calibration dataset (`long_tool_calling`) | ✅ Done | OpenClaw-style prompts, compact system prompt (~389tok), prompt saving |

## Current Blocker: NNCF OOM at >1400 Tokens

### Root Cause (from NNCF source investigation)

NNCF `compress_weights()` with `scale_estimation=True` or `awq=True` requires forward-pass statistics collection. The pipeline:

1. **`StatisticsAggregator.collect_statistics()`** (`nncf/common/tensor_statistics/aggregator.py:67-96`)
   - Creates a SINGLE modified OV model with ALL extra Result nodes (~242 for MoE-128)
   - Runs `subset_size` forward passes through this model
   - All 242 outputs coexist in memory simultaneously

2. **`_insert_outputs()`** (`nncf/openvino/graph/model_transformer.py`)
   - Clones the OV model adding ~242 Result nodes for MatMul inputs/outputs
   - OpenVINO cannot reuse buffers across these extra outputs

3. **`NoopAggregator`** (`nncf/common/tensor_statistics/collectors.py`)
   - Uses `deque(maxlen=None)` — stores ALL samples unreduced
   - For `subset_size=32` × 242 targets = 7,744 tensors in memory

### Memory Budget

| Component | Size |
|---|---|
| FP16 model weights | 57 GB |
| NNCF overhead (242 outputs × seq_len=1024 × FP16) | ~22 GB |
| **Total peak** | **~79 GB** (observed, safe in 96 GB) |
| At seq_len=1536 | ~90 GB (estimated, tight) |
| At seq_len=2048 | OOM (observed, 768MB alloc failure) |

## Plan: Layer-Batched Statistics with NVMe Offloading

### Architecture

Split the 242 statistic collection points into **4 batches of ~60**, run separate forward passes per batch, and spill completed batch results to NVMe via Direct I/O.

```
Batch 1: layers 0-11 (~60 stat points)  → collect → save to NVMe → free RAM
Batch 2: layers 12-23 (~60 stat points) → collect → save to NVMe → free RAM
Batch 3: layers 24-35 (~60 stat points) → collect → save to NVMe → free RAM
Batch 4: layers 36-47 (~60 stat points) → collect → load all → feed to compress_weights
```

### Memory Impact

| Scenario | Peak RAM | Max seq_len |
|---|---|---|
| Current (all 242 outputs) | 79 GB @ 1024tok | ~1400 |
| Batched (60 outputs) | ~63 GB @ 1024tok | **~4096** |
| Batched + NVMe spill | ~63 GB @ 1024tok | **~4096+** |

### Implementation Plan

#### File 1: `scripts/nvme_direct_io.py`
- Windows Direct I/O using `CreateFileW` + `FILE_FLAG_NO_BUFFERING` + `FILE_FLAG_WRITE_THROUGH`
- `VirtualAlloc` for sector-aligned buffers (4096-byte alignment)
- `save_array(path, numpy_array)` / `load_array(path) → numpy_array`
- Bypasses OS page cache to avoid memory pressure from spilled data

#### File 2: `scripts/batched_statistics.py`
- `BatchedStatisticsCollector` class
- Partitions NNCF's statistic point list into N batches (default 4)
- For each batch: create modified OV model with only that batch's Result nodes → run forward passes → serialize results → free memory
- After all batches: reassemble into NNCF's expected `statistics_path` `.safetensors` format
- Uses NNCF's existing `statistics_path` mechanism to skip re-collection

#### File 3: Modify `scripts/convert_hq.py`
- Add CLI flags: `--nvme-offload <path>` and `--layer-batches <N>`
- When offloading enabled: call `BatchedStatisticsCollector` instead of `compress_weights()` directly
- Raise `MAX_SEQ_LEN` from 1536 to 4096 when offloading is active
- Pass collected stats via `statistics_path` to `compress_weights()`

### Key NNCF Integration Points

- **`statistics_path` parameter** in `compress_weights()`: NNCF loads pre-collected stats from `.safetensors` file, skipping the collection phase entirely. This is the clean integration point — no monkey-patching needed for the batching itself.
- **Statistic point names**: Extractable from `StatisticsAggregator._statistic_points` after `algorithm.get_statistic_points()`.
- **`InplaceInsertionFnType`**: NNCF's existing inplace reducers already reduce full activations to `[hidden_dim]` channel statistics. The batching reduces the NUMBER of simultaneous outputs, not the per-output size.

### Verification

1. Run with `--nvme-offload D:\nvme-temp --layer-batches 4 --subset-size 32 --dataset long_tool_calling`
2. Monitor peak RAM with Task Manager — should stay under 65 GB
3. Compare output model quality: run benchmark suite at 4K/16K/32K and compare Hard Tool-Call scores against baseline
4. Verify `statistics_path` .safetensors file is valid by loading with safetensors library

## Next Steps

1. Implement `nvme_direct_io.py` (Windows Direct I/O)
2. Implement `batched_statistics.py` (layer-batched collection)
3. Integrate into `convert_hq.py` (CLI flags + orchestration)
4. Test with small subset_size (4) at seq_len=2048 to validate memory reduction
5. Full calibration run: subset_size=32, seq_len=4096, 4 batches
6. Benchmark the resulting INT4-HQ model against baseline
