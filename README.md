# Qwen3-Coder Tool-Call Enhancement

Improving long-context tool-calling accuracy for [Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) running on Intel iGPU via [OpenVINO](https://github.com/openvinotoolkit/openvino).

## Problem

When using the INT4-quantized Qwen3-Coder-30B-A3B as an agentic coding assistant, tool-calling accuracy degrades at **24K+ token context lengths** in realistic agentic workflows. The model stops executing tool calls and instead narrates analysis of conversation history — a classic attention-distraction failure.

## Model

| Property | Value |
|---|---|
| Model | [Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) |
| Architecture | Mixture of Experts (MoE) — 30B total, 3.3B active per token |
| Layers | 48 transformer layers, 128 experts, 8 active per token |
| Baseline INT4 | [OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int4-ov](https://huggingface.co/OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int4-ov) |
| Quantization | INT4_ASYM, group_size=128, data-free (no AWQ, no calibration dataset) |
| Model Size | 15.19 GB |

**Hardware:** Intel Core Ultra (integrated GPU), Windows, 96 GB RAM, iGPU memory override 76 GB

## Benchmark Design

Three test suites measure different aspects of long-context intelligence:

| Test | Description | Scoring |
|---|---|---|
| **NIAH** (Needle in a Haystack) | Insert a 6-digit passkey at various depths in filler text, ask model to retrieve it | PASS/FAIL per (depth × context_length) |
| **Tool-Call (Easy)** | 5 test cases, 7 tool definitions, generic filler text padding | 0-5 per case (tags, JSON, tool_name, arg_key, arg_value) |
| **Tool-Call (Hard)** | 5 test cases, 12 tool definitions, realistic agentic conversation history with prior tool call/result turns | 0-5 per case |

Context lengths: 4K, 8K, 16K, 24K, 32K tokens (quick mode) or 4K–32K in 4K steps (full mode).
Extended lengths: 32K, 48K, 64K tokens for stress testing.

The **hard** test is the most representative — it simulates a real coding agent conversation with interleaved tool calls and results (12 tool definitions, ~190 prior tool call/result turns at 32K), then asks the model to use a specific tool with specific arguments at the very end of the prompt.

## Results

### Baseline INT4 (gs=128, ratio=1.0, data-free)

#### Quick Mode (4K–32K)

| Test | 4K | 8K | 16K | 24K | 32K |
|---|---|---|---|---|---|
| NIAH | 100% | 100% | 100% | 100% | 100% |
| Tool-Call (Easy) | 5.0/5 | 5.0/5 | 5.0/5 | 5.0/5 | 5.0/5 |
| **Tool-Call (Hard)** | **4.8/5** | **4.8/5** | **5.0/5** | **4.0/5** | **4.2/5** |

#### Extended Stress Test (32K–64K, 2026-04-02)

| Test | 32K | 48K | 64K |
|---|---|---|---|
| Tool-Call (Easy, 7 tools) | **5.0/5** | **5.0/5** | **5.0/5** |
| **Tool-Call (Hard, 12 tools)** | **4.0/5** | **4.2/5** | **4.0/5** |

Hard tool-call failure details at extended lengths:

| Context | `read_file` | `search_code` | `get_diagnostics` | `replace_in_file` | `run_command` |
|---|---|---|---|---|---|
| 32K | **FAIL (1/5)** — 70s | PERFECT | PERFECT | PARTIAL (4/5) | PERFECT |
| 48K | **FAIL (1/5)** — 155s | PERFECT | PERFECT | PERFECT | PERFECT |
| 64K | **FAIL (0/5)** — 218s | PERFECT | PERFECT | PERFECT | PERFECT |

**Failure mode:** The model refuses to use the `read_file` tool because it has seen ~95 prior `read_file` calls to the same file (`handlers.py`) in the agentic conversation history. Instead of executing the requested tool call, it generates a long narrative:
> *"I've reviewed the project structure and codebase multiple times, but I haven't found any actual bugs in the code..."*

The extremely long elapsed times (70–218s vs ~2-4s for passing tests) confirm the model is generating hundreds of tokens of explanation rather than a concise tool call.

Key observations:
- NIAH and easy tool-call are **perfect through 64K** — the model handles basic recall and simple tool use flawlessly even at very long contexts
- Hard tool-call **degrades at 24K+**: model narrates code analysis instead of executing tool calls
- The `read_file` failure is **consistent and worsening** — score drops from 1/5 at 32K to 0/5 at 64K
- `replace_in_file` test: model sometimes chooses `read_file` instead (read-before-modify strategy)
- Other tools (`search_code`, `get_diagnostics`, `run_command`) remain **perfect through 64K**

### Experiment 1: Router Dequantization (INT4-RouterFP16)

**Hypothesis:** INT4-quantized MoE router (`mlp.gate`) weights cause expert misrouting at long contexts.

**Method:** Surgically dequantized all 48 router layers from INT4 → FP16 via XML/BIN direct editing (`dequant_routers.py`). Model size increased by only 24 MB (15.19 → 15.22 GB).

| Test | INT4 Baseline | INT4-RouterFP16 |
|---|---|---|
| NIAH (4K-32K) | 100% | 100% |
| Easy tool-call (4K-32K) | 5.0/5 all | 5.0/5 all |
| Hard 16K | **5.0/5** | 4.8/5 |
| Hard 24K | 4.0/5 | 4.0/5 |
| Hard 32K | 4.2/5 | 4.2/5 |

**Conclusion:** Router quantization is **NOT** the bottleneck. The degradation comes from attention layer (q/k/v/o_proj) INT4 quantization affecting the model's ability to prioritize recent instructions over extensive conversation history.

### Experiment 2: GPU Compatibility Matrix (2026-04-01)

Systematically tested all feasible INT4 parameter combinations on the Intel Arc B390 iGPU:

| Configuration | GPU Status | Model Size | Notes |
|---|---|---|---|
| gs=128, ratio=1.0 (baseline) | **OK** | 15.22 GB | All weights INT4 (384 constants) |
| gs=128, ratio=1.0, RouterFP16 | **OK** | 15.25 GB | 48 routers FP16, rest INT4 |
| gs=128, ratio=0.9 | **CRASH** (segfault) | — | Mixed INT4+INT8 unsupported |
| gs=128, ratio=0.8 | **CRASH** (segfault) | 17.94 GB | 105 INT4, 137 INT8, 48 BF16 |
| gs=64, ratio=1.0 | **CRASH** (segfault) | — | GPU lacks gs=64 decompression kernels |
| gs=64, ratio=0.8 | **CRASH** (segfault) | — | Both gs=64 and mixed precision fail |

**Key constraint:** The Intel Arc B390 GPU plugin only supports **uniform INT4 gs=128 (ratio=1.0)**. Any ratio < 1.0 introduces INT8 weight constants that crash during `compile_model()`. All crashed configs work fine on CPU.

### Experiment 3: NNCF Calibration Memory Limits (2026-03-31)

Tested NNCF `compress_weights()` with tool-calling calibration data at various sequence lengths on 96 GB RAM:

| MAX_SEQ_LEN | Peak RAM | Result |
|---|---|---|
| 1024 tokens | ~79 GB | **OK** (but only truncated system prompts, no tool-call data) |
| 1280 tokens | ~85 GB est. | **OK** (compact prompt, 0/8 samples truncated) |
| 1536 tokens | OOM (1.43 GB alloc failure) | **CRASH** on sample 2 at 1469 tokens |
| 2048 tokens | OOM (768 MB alloc failure) | **CRASH** on first sample |

**Hard limit:** ~1300-1400 tokens per calibration sample with this model on 96 GB RAM.

### HuggingFace INT8 Reference Models

For comparison, the official OpenVINO Hub provides INT8 variants:

| Model | Precision | group_size | Source | Est. Size |
|---|---|---|---|---|
| [OpenVINO/Qwen3-30B-A3B-int8-ov](https://huggingface.co/OpenVINO/Qwen3-30B-A3B-int8-ov) | INT8_ASYM | -1 (per-channel) | Qwen3-30B-A3B (base) | ~30 GB |
| [OpenVINO/Qwen3-30B-A3B-Instruct-2507-int8-ov](https://huggingface.co/OpenVINO/Qwen3-30B-A3B-Instruct-2507-int8-ov) | INT8_ASYM | -1 (per-channel) | Qwen3-30B-A3B-Instruct-2507 | ~30 GB |

INT8 per-channel would give better quality (256 levels vs 16), but at ~30 GB model weight + KV cache for 32K+ context, these would push the 76 GB iGPU limit. Our INT4 at ~15 GB leaves ~60 GB for KV cache.

### Next Steps: NNCF Calibrated Conversion

**Hypothesis:** NNCF `scale_estimation` with tool-calling calibration data will optimize INT4 scales/zero-points for tool-call activation patterns, improving hard tool-call accuracy even at ratio=1.0.

**GPU constraints require:** ratio=1.0 only (no mixed INT4+INT8), gs=128 only (no gs=64).

**Remaining viable optimization levers:**
- `scale_estimation` — optimizes per-group scales using calibration data (the main quality driver)
- Tool-calling calibration dataset — compact 8-tool system prompt + multi-turn tool call/result exchanges
- `max_activation_variance` sensitivity metric — identifies most-sensitive layers

**Planned command:**
```bash
python scripts/convert_hq.py --tier tier1 --dataset long_tool_calling \
  --group-size 128 --ratio 1.0 --subset-size 32 \
  --sensitivity-metric max_activation_variance \
  --output-dir "INT4-TOOLCAL"
```

**Calibration data design:**
- MAX_SEQ_LEN = 1280 tokens (hard memory limit ~1400)
- Compact system prompt with 8 core tools (~389 tokens)
- 1-2 complete tool-call exchanges per sample
- Pre-truncation diagnostics confirm 0/8 samples truncated

**What was ruled out:**
- AWQ: crashes with MoE models (`ValueError: matmul shape mismatch` in NNCF 2.19.0)
- gs=64: GPU crashes on Intel Arc B390
- ratio < 1.0: GPU crashes (mixed INT4+INT8 unsupported)
- MAX_SEQ_LEN > 1400: OOM on 96 GB RAM

**Status:** `convert_hq.py` fully updated with compact prompt, AWQ fix, correct defaults. Ready to run.

## Project Structure

```
qwen3-model-tool-call-enhancement/
├── README.md
├── .gitignore
├── scripts/
│   ├── benchmark_context.py     # Long-context benchmark (NIAH + tool-calling)
│   ├── convert_hq.py            # High-quality INT4 conversion (MoE-optimized)
│   ├── dequant_routers.py       # Surgical MoE router dequantization (INT4→FP16)
│   ├── dump_prompts.py          # Dump benchmark prompt contents for inspection
│   └── llm_code_assistant.py    # Standalone CLI code assistant (baseline app)
├── benchmarks/
│   ├── benchmark_INT4_*.json         # Baseline INT4 results
│   ├── benchmark_INT4_*.csv
│   ├── benchmark_INT4-RouterFP16_*.json  # RouterFP16 results
│   └── benchmark_INT4-RouterFP16_*.csv
├── prompt_samples/                   # Saved benchmark prompts for inspection
└── docs/
    └── 2026-03-30-plan.md            # Work log and plan (Chinese)
```

## Setup

### Prerequisites

- Python 3.10+
- Intel GPU (or CPU)

### Install Dependencies

```bash
python -m venv venv
venv\Scripts\activate  # Windows

pip install openvino openvino-genai openvino-tokenizers nncf datasets
pip install "git+https://github.com/huggingface/optimum-intel.git"
pip install hf_xet  # optional, for faster downloads
```

## Usage

### Run Benchmark

```bash
# Quick benchmark (5 context lengths)
python scripts/benchmark_context.py --model-dir "C:\working\models\Qwen3-Coder-30B-A3B-Instruct\INT4" --quick

# Hard tool-call test only
python scripts/benchmark_context.py --model-dir "..." --tests tool_call_hard --quick

# Compare two models
python scripts/benchmark_context.py --compare "...\INT4" "...\INT4-HQ" --tests tool_call_hard --quick

# Full benchmark (all depths × 8 context lengths)
python scripts/benchmark_context.py --model-dir "..." --tests niah tool_call tool_call_hard
```

### Router Dequantization

```bash
# Dry run — analyze without modifying
python scripts/dequant_routers.py --dry-run

# Dequantize routers to FP16
python scripts/dequant_routers.py --model-dir "...\INT4" --output-dir "...\INT4-RouterFP16"
```

### High-Quality Conversion

```bash
# NNCF calibrated with tool-calling data (ratio=1.0 required for GPU)
python scripts/convert_hq.py --tier tier1 --dataset long_tool_calling \
  --group-size 128 --ratio 1.0 --subset-size 32 \
  --sensitivity-metric max_activation_variance \
  --output-dir "INT4-TOOLCAL"

# Tier 3: optimum-cli data-free (uses local HF weights if available)
python scripts/convert_hq.py --tier tier3 --group-size 128 --ratio 1.0

# Inspect model precision
python scripts/convert_hq.py --inspect --model-dir "...\INT4"
```

### Code Assistant CLI

```bash
# Interactive chat
python scripts/llm_code_assistant.py --task chat --device GPU --max-tokens 512 --no-think

# Code generation
python scripts/llm_code_assistant.py --task generate --device GPU --prompt "Write a binary search"

# Bug fix / Security audit
python scripts/llm_code_assistant.py --task fix --device GPU --no-think
python scripts/llm_code_assistant.py --task audit --device GPU --no-think
```

## Key Findings

1. **NIAH and simple tool-calling are unaffected by INT4 quantization** — the model handles these perfectly up to **64K tokens**
2. **Hard agentic tool-calling degrades at 24K+** — the model fails to prioritize the latest user instruction when surrounded by extensive conversation history
3. **The `read_file` tool is the specific failure** — because the agentic history contains ~95 prior `read_file` calls to the same file, the model thinks it already read it. Score: 1/5 at 32K → 0/5 at 64K
4. **MoE router quantization is NOT the cause** — surgically dequantizing all 48 routers to FP16 had zero measurable impact
5. **Attention layers are the likely bottleneck** — 192 INT4-quantized q/k/v/o_proj layers affect the model's attention patterns at long contexts
6. **The failure mode is "narration"** — instead of calling the requested tool, the model generates hundreds of tokens analyzing the codebase (70-218 seconds vs 2-4 seconds for passing tests)
7. **Intel Arc B390 GPU only supports uniform INT4 gs=128** — ratio < 1.0 (mixed INT4/INT8) crashes, gs=64 crashes. Only ratio=1.0 works.
8. **AWQ is broken for MoE models** in NNCF 2.19.0 — `matmul shape mismatch` in awq.py
9. **NNCF calibration OOMs above ~1400 tokens** on 96 GB RAM with this model
10. **LayerNorm weights are already FP32** — no need to protect them
11. **Embedding/lm_head are already INT8** — the default conversion already protects these
12. **Long-context calibration data is the key missing piece** — standard short-text calibration (wikitext2 @ 2K tokens) cannot teach scale_estimation about attention patterns at 24K+

## License

Based on [OpenVINO Notebooks](https://github.com/openvinotoolkit/openvino_notebooks), licensed under Apache 2.0.
