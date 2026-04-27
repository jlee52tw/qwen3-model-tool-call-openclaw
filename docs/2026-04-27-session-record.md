# Session Record: Qwen3-30B INT4 Quantization — April 23-27, 2026

## CRITICAL: Resume Actions Required

### 1. FREE DISK SPACE BEFORE RESUMING
The C:\ drive is **completely full (0.0 GB free)** on 1862 GB total.
Model directories alone consume **209.4 GB**:

| Directory | Size | Keep? |
|-----------|------|-------|
| FP16-temp | 57.0 GB | **YES** — needed for all quantization runs |
| HF | 56.9 GB | **YES** — local HuggingFace source weights |
| INT4 | 15.2 GB | Maybe — baseline model, can regenerate |
| INT4_GS128_R08_DATAFREE | 17.9 GB | **DELETE** — failed experiment (ratio 0.8 crashes GPU) |
| INT4_GS128_R09_DATAFREE | 16.6 GB | **DELETE** — failed experiment (ratio 0.9 crashes GPU) |
| INT4_GS128_R10_FRESH | 15.2 GB | Maybe — data-free baseline |
| INT4_GS128_TEST | 15.2 GB | **DELETE** — test only |
| INT4-RouterFP16 | 15.2 GB | Maybe — router dequant variant |
| INT4-HQ-1K | 0.0 GB | Empty (save failed) |

**Recommended cleanup** (recovers ~49.7 GB):
```powershell
Remove-Item -Recurse -Force "C:\working\models\Qwen3-Coder-30B-A3B-Instruct\INT4_GS128_R08_DATAFREE"
Remove-Item -Recurse -Force "C:\working\models\Qwen3-Coder-30B-A3B-Instruct\INT4_GS128_R09_DATAFREE"
Remove-Item -Recurse -Force "C:\working\models\Qwen3-Coder-30B-A3B-Instruct\INT4_GS128_TEST"
```

### 2. RE-RUN THE 1K QUANTIZATION
The full pipeline **completed successfully** but crashed saving due to full disk.
After freeing disk, re-run with:
```powershell
cd C:\working\qwen3-model-tool-call-enhancement\scripts
python .\convert_hq.py --tier tier1 --dataset tool_calling --subset-size 8 --max-seq-len 1024 --ratio 1.0 --force --output-dir "C:\working\models\Qwen3-Coder-30B-A3B-Instruct\INT4-HQ-1K"
```
**Expected time: ~66 minutes** (56 min stats + 7 min scale_estimation + 2 min compression + 1 min save)

---

## Milestone: First Successful Full Pipeline (April 23, 2026)

The complete NNCF quantization pipeline ran end-to-end for the first time:

| Phase | Time | Status |
|-------|------|--------|
| Statistics collection | 56:12 (8/8 samples) | ✅ Complete |
| Bitwidth distribution | — | ✅ 48 float (routers), 2 INT8 (embed/lm_head), 336 INT4 |
| Scale Estimation | 07:19 (338/338) | ✅ Complete (144 MoE experts skipped as expected) |
| Weight Compression | 01:52 | ✅ Complete |
| Total NNCF time | 65.6 min | ✅ |
| Save model | — | ❌ Disk full: `RuntimeError: ios_base::badbit set: iostream stream error` |

### What works:
- `tool_calling` dataset with 8 samples @ 1024 tokens
- Non-batched mode (all 242 stat targets at once) on 96 GB RAM
- FP16 inference patch (halves memory vs NNCF default FP32)
- Scale estimation patch (catches IndexError for 144 3D MoE expert weights)
- Router protection via `ignored_scope` (48 mlp.gate layers → float)
- Ratio=1.0 (100% INT4 for non-router weight layers)

---

## Technical Details

### Hardware
- 96 GB RAM, Intel Core Ultra + Arc B390 iGPU
- 1862 GB C:\ drive (currently FULL)
- 225 GB page file (system-managed)

### Software
- NNCF 2.19.0, OpenVINO 2026.2.0-21415, Python 3.12, torch 2.8.0+cpu
- Repository: `jlee52tw/qwen3-model-tool-call-openclaw`, branch `main`

### Model
- Qwen3-Coder-30B-A3B-Instruct (MoE: 48 layers, 128 experts, 8 active/token)
- FP16 OV IR: 57 GB at `C:\working\models\Qwen3-Coder-30B-A3B-Instruct\FP16-temp`
- 386 MatMul ops: 192 attention (2D), 144 MoE expert (3D: [128,768,2048]), 48 router, 2 embed/lm_head
- 241 NNCF stat target nodes

### GPU Constraints (Arc B390)
- **Must use**: INT4_ASYM, group_size=128, ratio=1.0
- **Crashes with**: gs=64, ratio<1.0, AWQ=True

### Key Code Patches in convert_hq.py

1. **FP16 inference** (line ~312): Monkey-patches `OVNativeEngine.__init__` to `use_fp32_precision=False`
2. **Scale estimation 3D skip** (line ~330-375): Catches `IndexError` on 3D MoE expert weights [128,768,2048] where activation stats reduce to 1D but NNCF expects 3D-matching dimensions
3. **Auto-batch threshold** (line ~261): Only enables batching for `max_seq_len > 1536` (not by default)

### Commits (all pushed to GitHub)
- `36d8f89` — max_seq_len fix
- `0c7dfe7` — scale_estimation patch (first version)
- `1db8381` — always-batch rewrite
- `0eb44e0` — mixed precision stats disabled
- `435bcba` — round-robin + tighter budget
- `f173c00` — fix: remove non-existent `matmul_has_transposed_activations` + revert auto-batch

---

## Bug History & Fixes

| # | Bug | Root Cause | Fix | Commit |
|---|-----|-----------|-----|--------|
| 1 | `IndexError: tuple index out of range` in scale_estimation | 3D MoE expert weights [128,768,2048] produce 1D stats, but `reshape_weight_for_grouped_quantization` expects matching dims | Monkey-patch `ScaleEstimation.apply` to catch IndexError per-layer | 0c7dfe7, f173c00 |
| 2 | `AttributeError: matmul_has_transposed_activations` | Our patch called an API that doesn't exist in NNCF 2.19.0 (from older version) | Removed the call, simplified to match actual NNCF source | f173c00 |
| 3 | `ArrayMemoryError: Unable to allocate 989 MiB` | Mixed precision hessian stats compute `x*x` on 3D MoE activation tensors → huge numpy arrays | `enable_mixed_precision=False` in batched stats (unnecessary with ratio=1.0) | 0eb44e0 |
| 4 | OV batch transition failure | OV CPU plugin doesn't release compiled model buffers between batches in same process | Only batch when required (seq_len > 1536), use non-batched for ≤1536 | f173c00 |
| 5 | `ArrayMemoryError: (1, 732, 151936) float32` | `long_tool_calling` samples are longer (avg 866 tokens) → larger LM head output tensor under memory pressure | Use `tool_calling` dataset (shorter tokens, same patterns) | Runtime choice |
| 6 | `ios_base::badbit` on save | C:\ drive completely full (0.0 GB free) | **FREE DISK SPACE** (~50 GB needed) | Pending |

---

## Batching Status (for 2K+ tokens)

Batched statistics collection is **implemented but blocked** by OV CPU plugin bug:
- Batch 1 always completes successfully
- Batch 2+ always fails with `RuntimeError: Failed to allocate N bytes` even with 58+ GB free RAM
- Root cause: OV internal memory pool management, not system memory

**For 2K+ tokens**, options:
1. **Subprocess per batch** — spawn new Python process for each batch (avoids OV memory leak)
2. **256 GB server** — brute force, no batching needed up to ~4K tokens
3. **PyTorch-based stats collection** — bypass OV entirely for statistics

**For 1K tokens**: Non-batched works fine (56 min, peaks ~80 GB with paging).

---

## Progressive Quantization Plan

Original goal: 1K → 2K → 4K → 8K → 32K token calibration runs.

| Seq Len | Batching Needed | Status |
|---------|----------------|--------|
| 1K (1024) | No | ✅ Pipeline complete, need disk space to save |
| 2K (2048) | Yes (OOM at ~1400 tokens without batching) | ⏳ Blocked by OV batch bug |
| 4K (4096) | Yes | ⏳ |
| 8K (8192) | Yes | ⏳ |
| 32K (32768) | Yes | ⏳ |

---

## File Inventory

### Scripts
- `scripts/convert_hq.py` (~1527 lines) — Main conversion script with all patches
- `scripts/batched_statistics.py` (~369 lines) — Batched NNCF stats (works for batch 1, blocked at batch 2+)
- `scripts/nvme_direct_io.py` (~290 lines) — Windows Direct I/O for NVMe offload

### Logs
- `C:\working\models\Qwen3-Coder-30B-A3B-Instruct\convert_1k_tc.log` — Successful run log (tool_calling, failed at save)
- `C:\working\models\Qwen3-Coder-30B-A3B-Instruct\convert_1k_long.log` — Failed run log (long_tool_calling, OOM at sample 5/8)
- `C:\working\models\Qwen3-Coder-30B-A3B-Instruct\convert_1k.log` — Earlier failed run (matmul_has_transposed_activations bug)

### Previous docs
- `docs/2026-03-30-plan.md` — Original optimization plan
