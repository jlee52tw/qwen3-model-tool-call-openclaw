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

**Hardware:** Intel Core Ultra (integrated GPU), Windows, 32 GB RAM, iGPU memory override 27.5 GB

## Benchmark Design

Three test suites measure different aspects of long-context intelligence:

| Test | Description | Scoring |
|---|---|---|
| **NIAH** (Needle in a Haystack) | Insert a 6-digit passkey at various depths in filler text, ask model to retrieve it | PASS/FAIL per (depth × context_length) |
| **Tool-Call (Easy)** | 5 test cases, 7 tool definitions, generic filler text padding | 0-5 per case (tags, JSON, tool_name, arg_key, arg_value) |
| **Tool-Call (Hard)** | 5 test cases, 12 tool definitions, realistic agentic conversation history with prior tool call/result turns | 0-5 per case |

Context lengths: 4K, 8K, 16K, 24K, 32K tokens (quick mode) or 4K–32K in 4K steps (full mode).

The **hard** test is the most representative — it simulates a real coding agent conversation with interleaved tool calls and results, then asks the model to use a specific tool with specific arguments.

## Results

### Baseline INT4

| Test | 4K | 8K | 16K | 24K | 32K |
|---|---|---|---|---|---|
| NIAH | 100% | 100% | 100% | 100% | 100% |
| Tool-Call (Easy) | 5.0/5 | 5.0/5 | 5.0/5 | 5.0/5 | 5.0/5 |
| **Tool-Call (Hard)** | **4.8/5** | **4.8/5** | **5.0/5** | **4.0/5** | **4.2/5** |

Key observations:
- NIAH and easy tool-call are **perfect** — the model handles basic recall and simple tool use flawlessly
- Hard tool-call **degrades at 24K+**: model narrates code analysis instead of executing tool calls
- `replace_in_file` test: model consistently chooses `read_file` instead (read-before-modify strategy, all contexts)

### Experiment 1: Router Dequantization (INT4-RouterFP16)

**Hypothesis:** INT4-quantized MoE router (`mlp.gate`) weights cause expert misrouting at long contexts.

**Method:** Surgically dequantized all 48 router layers from INT4 → FP16 via XML/BIN direct editing (`dequant_routers.py`). Model size increased by only 24 MB (15.19 → 15.22 GB).

| Test | INT4 Baseline | INT4-RouterFP16 |
|---|---|---|
| NIAH (4K-32K) | 100% | 100% |
| Easy tool-call | 5.0/5 all | 5.0/5 all |
| Hard 16K | **5.0/5** | 4.8/5 |
| Hard 24K | 4.0/5 | 4.0/5 |
| Hard 32K | 4.2/5 | 4.2/5 |

**Conclusion:** Router quantization is **NOT** the bottleneck. The degradation comes from attention layer (q/k/v/o_proj) INT4 quantization affecting the model's ability to prioritize recent instructions over extensive conversation history.

### Experiment 2: High-Quality Re-Conversion (Planned)

**Hypothesis:** Data-aware INT4 with AWQ + scale_estimation + smaller group_size will preserve attention precision better.

**Method:** Re-convert from HuggingFace weights using `convert_hq.py`:
- AWQ (Activation-aware Weight Quantization) — calibrates quantization ranges using real data
- Scale estimation — optimizes per-channel scales
- Group size 64 (vs baseline 128) — finer granularity
- Sensitivity metric — protects most-sensitive layers
- Calibration dataset — wikitext2, 128 samples

**Status:** Model downloading, conversion pending.

## Project Structure

```
qwen3-model-tool-call-enhancement/
├── README.md
├── .gitignore
├── scripts/
│   ├── benchmark_context.py     # Long-context benchmark (NIAH + tool-calling)
│   ├── convert_hq.py            # High-quality INT4 conversion (MoE-optimized)
│   ├── dequant_routers.py       # Surgical MoE router dequantization (INT4→FP16)
│   └── llm_code_assistant.py    # Standalone CLI code assistant (baseline app)
└── benchmarks/
    ├── benchmark_INT4_*.json         # Baseline INT4 results
    ├── benchmark_INT4_*.csv
    ├── benchmark_INT4-RouterFP16_*.json  # RouterFP16 results
    └── benchmark_INT4-RouterFP16_*.csv
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
# Tier 3: optimum-cli with AWQ + gs=64
python scripts/convert_hq.py --tier tier3

# Tier 1: NNCF API with router protection + AWQ
python scripts/convert_hq.py --tier tier1 --subset-size 64

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

1. **NIAH and simple tool-calling are unaffected by INT4 quantization** — the model handles these perfectly up to 32K tokens
2. **Hard agentic tool-calling degrades at 24K+** — the model fails to prioritize the latest user instruction when surrounded by extensive conversation history
3. **MoE router quantization is NOT the cause** — surgically dequantizing all 48 routers to FP16 had zero measurable impact
4. **Attention layers are the likely bottleneck** — 192 INT4-quantized q/k/v/o_proj layers affect the model's attention patterns at long contexts
5. **The failure mode is "narration"** — instead of calling the requested tool, the model analyzes and describes the code in the conversation history

## License

Based on [OpenVINO Notebooks](https://github.com/openvinotoolkit/openvino_notebooks), licensed under Apache 2.0.
