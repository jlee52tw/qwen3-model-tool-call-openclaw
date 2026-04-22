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
  - Long-context (20K+) agentic calibration data for attention layer protection

Tiers:
  tier3  — optimum-cli with AWQ + scale_estimation + gs=64 (fastest, no router protection)
  tier1  — NNCF Python API with router protection + AWQ + gs=64 (highest quality)

Usage:
    python convert_hq.py --tier tier3
    python convert_hq.py --tier tier3 --local-model "C:\\...\\HF"  # use local weights
    python convert_hq.py --tier tier1 --subset-size 64 --dataset long_tool_calling
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
LOCAL_HF_DIR = BASE_MODEL_DIR / "HF"

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

    # Determine model source: local HF dir or HuggingFace Hub
    model_source = args.local_model or str(LOCAL_HF_DIR) if LOCAL_HF_DIR.exists() else MODEL_ID
    if args.local_model:
        model_source = args.local_model
    elif LOCAL_HF_DIR.exists() and any(LOCAL_HF_DIR.glob("*.safetensors")):
        model_source = str(LOCAL_HF_DIR)
        print(f"[INFO] Using local HF weights: {model_source}")
    else:
        model_source = MODEL_ID

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
        "--model", model_source,
        "--task", "text-generation-with-past",
        "--weight-format", "int4",
        "--group-size", str(args.group_size),
        "--ratio", str(args.ratio),
        "--awq",
        "--scale-estimation",
        "--dataset", args.dataset if args.dataset != "long_tool_calling" else "wikitext2",
        "--num-samples", str(args.subset_size),
        "--sensitivity-metric", args.sensitivity_metric,
        "--all-layers",
        str(output_dir),
    ]

    if args.backup_precision:
        cmd.insert(-1, "--backup-precision")
        cmd.insert(-1, args.backup_precision)

    print(f"\n{'='*80}")
    print(f"  TIER 3: optimum-cli High-Quality Conversion")
    print(f"{'='*80}")
    print(f"  Model:              {model_source}")
    print(f"  Output:             {output_dir}")
    print(f"  Mode:               INT4_ASYM")
    print(f"  Group size:         {args.group_size}")
    print(f"  Ratio:              {args.ratio}")
    print(f"  AWQ:                True")
    print(f"  Scale estimation:   True")
    print(f"  Dataset:            {args.dataset}")
    print(f"  Num samples:        {args.subset_size}")
    print(f"  Sensitivity metric: {args.sensitivity_metric}")
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

    # Determine model source for FP16 export
    model_source = args.local_model or str(LOCAL_HF_DIR) if LOCAL_HF_DIR.exists() else MODEL_ID
    if args.local_model:
        model_source = args.local_model
    elif LOCAL_HF_DIR.exists() and any(LOCAL_HF_DIR.glob("*.safetensors")):
        model_source = str(LOCAL_HF_DIR)
        print(f"[INFO] Using local HF weights: {model_source}")
    else:
        model_source = MODEL_ID

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
            "--model", model_source,
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
    print(f"  AWQ:                {args.awq}")
    print(f"  Scale estimation:   True")
    print(f"  Router protection:  YES (ignored_scope: {MOE_ROUTER_PATTERN})")
    print(f"  Sensitivity metric: {args.sensitivity_metric}")
    print(f"  Backup mode:        {args.backup_precision or 'INT8_ASYM'}")
    print(f"  Subset size:        {args.subset_size}")
    print(f"  All layers:         False (embeddings/lm_head → INT8)")
    print()

    # ── Monkey-patch NNCF to use FP16 inference for statistics collection ──
    # By default NNCF uses use_fp32_precision=True which forces FP32 inference,
    # doubling all intermediate tensor sizes. For MoE-128 models, NNCF inserts
    # hundreds of extra Result/Output nodes that prevent OpenVINO from reusing
    # buffers. Combined with FP32 precision, even short sequences cause OOM.
    # FP16 inference halves memory and is sufficient for channel-magnitude stats.
    import nncf.openvino.engine as _nncf_engine
    _orig_engine_init = _nncf_engine.OVNativeEngine.__init__
    def _patched_engine_init(self, model, use_fp32_precision=False):
        _orig_engine_init(self, model, use_fp32_precision=False)
    _nncf_engine.OVNativeEngine.__init__ = _patched_engine_init
    print("  [PATCH] NNCF inference precision: FP16 (reduced from FP32 for MoE memory)")

    core = ov.Core()
    ov_model = core.read_model(str(fp16_dir / "openvino_model.xml"))

    # Determine backup mode
    backup = BackupMode.INT8_ASYM
    if args.backup_precision == "int8_sym":
        backup = BackupMode.INT8_SYM
    elif args.backup_precision == "none":
        backup = BackupMode.NONE

    # Parse sensitivity metric
    metric_map = {
        "max_activation_variance": SensitivityMetric.MAX_ACTIVATION_VARIANCE,
        "mean_activation_variance": SensitivityMetric.MEAN_ACTIVATION_VARIANCE,
        "mean_activation_magnitude": SensitivityMetric.MEAN_ACTIVATION_MAGNITUDE,
        "hessian_input_activation": SensitivityMetric.HESSIAN_INPUT_ACTIVATION,
        "weight_quantization_error": SensitivityMetric.WEIGHT_QUANTIZATION_ERROR,
    }
    sensitivity = metric_map.get(args.sensitivity_metric, SensitivityMetric.MAX_ACTIVATION_VARIANCE)

    start = time.time()
    compressed_model = compress_weights(
        ov_model,
        mode=CompressWeightsMode.INT4_ASYM,
        group_size=args.group_size,
        ratio=args.ratio,
        ignored_scope=nncf.IgnoredScope(patterns=[MOE_ROUTER_PATTERN]),
        dataset=nncf_dataset,
        sensitivity_metric=sensitivity,
        subset_size=args.subset_size,
        awq=args.awq,
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
        # Short tool-calling calibration data (~4K tokens per sample)
        return _prepare_tool_calling_dataset(tokenizer, num_samples)
    elif dataset_name == "long_tool_calling":
        # Long-context agentic calibration data (~20K+ tokens per sample)
        return _prepare_long_tool_calling_dataset(tokenizer, num_samples)

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
        token_len = inputs["input_ids"].shape[1]
        import torch
        inputs["position_ids"] = torch.arange(token_len).unsqueeze(0)
        inputs["beam_idx"] = torch.zeros(1, dtype=torch.int32)
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
            token_len = inputs["input_ids"].shape[1]
            import torch
            inputs["position_ids"] = torch.arange(token_len).unsqueeze(0)
            inputs["beam_idx"] = torch.zeros(1, dtype=torch.int32)
            calibration_data.append(dict(inputs))

            if len(calibration_data) >= num_samples:
                break
        if len(calibration_data) >= num_samples:
            break

    print(f"  Collected {len(calibration_data)} tool-calling calibration samples")
    return calibration_data


def _prepare_long_tool_calling_dataset(tokenizer, num_samples=32):
    """
    Create OpenClaw v2026 tool-calling calibration samples for scale_estimation.

    Each sample simulates a real OpenClaw agentic session with:
    - Compact system prompt with 8 core tool definitions (~500 tokens)
    - Multi-turn conversation history with tool calls, results, and code context
    - Final user instruction requiring precise tool use

    Hardware constraint (96 GB RAM + 57 GB FP16 MoE-128 model):
      NNCF inserts extra Result nodes for ~242 MatMul targets, preventing
      OpenVINO buffer reuse. Observed memory: 79 GB peak at 1024 tokens.
      Overhead scales ~linearly with seq_len:
        seq_len=1024 + FP16: ~79 GB peak (observed, safe)
        seq_len=1536 + FP16: ~90 GB peak (estimated, tight fit in 96 GB)
        seq_len=2048 + FP16: OOM (tried, 768MB alloc failure on MatMul)
      We use MAX_SEQ_LEN=1280 as the proven safe limit.
      We monkey-patch NNCF to use FP16 inference (default is FP32 = 2x worse).

    CRITICAL: The system prompt with 20 tools + full instructions is ~1,865 tokens.
      At MAX_SEQ_LEN=1024, truncation cuts mid-system-prompt — all tool-call
      exchanges are lost. We use a COMPACT system prompt (~389 tokens) so that
      ~891 tokens remain for actual tool-call exchanges (1-2 complete rounds).

    Why this works:
      scale_estimation identifies important weight channels by ACTIVATION MAGNITUDES.
      The importance pattern is determined by CONTENT TYPE (tool calls, JSON structure,
      code context in tool results), not primarily by sequence length.
      Having 3-5 complete tool-call exchanges (user→tool_call→result→analysis) is far
      more valuable than a truncated system prompt with no exchanges.
    """
    import json
    import random

    # ── OpenClaw v2026.3.x Tool Definitions ──────────────────────────────────
    # Matches the actual tool set from OpenClaw's Full Mode system prompt.
    # Tool names are case-sensitive and must be called exactly as listed.
    TOOL_DEFINITIONS = [
        # Core file operations
        {"name": "read", "description": "Read raw text from a local file",
         "parameters": {"type": "object", "properties": {
             "file_path": {"type": "string", "description": "Path to the file to read"}
         }, "required": ["file_path"]}},
        {"name": "write", "description": "Overwrite or create a new file",
         "parameters": {"type": "object", "properties": {
             "file_path": {"type": "string", "description": "Path to the file"},
             "content": {"type": "string", "description": "Content to write"}
         }, "required": ["file_path", "content"]}},
        {"name": "edit", "description": "Precise string-replacement within a file",
         "parameters": {"type": "object", "properties": {
             "file_path": {"type": "string", "description": "Path to the file"},
             "search_string": {"type": "string", "description": "Exact text to find"},
             "replace_string": {"type": "string", "description": "Replacement text"}
         }, "required": ["file_path", "search_string", "replace_string"]}},
        {"name": "apply_patch", "description": "Apply a unified git diff/patch to the codebase",
         "parameters": {"type": "object", "properties": {
             "patch_content": {"type": "string", "description": "Unified diff content"}
         }, "required": ["patch_content"]}},
        # Search & navigation
        {"name": "grep", "description": "Search file contents for specific patterns",
         "parameters": {"type": "object", "properties": {
             "pattern": {"type": "string", "description": "Regex or literal pattern"},
             "path": {"type": "string", "description": "Directory or file to search"},
             "include": {"type": "string", "description": "Glob pattern for files"}
         }, "required": ["pattern"]}},
        {"name": "find", "description": "Find files by glob pattern",
         "parameters": {"type": "object", "properties": {
             "pattern": {"type": "string", "description": "Glob pattern to match"},
             "path": {"type": "string", "description": "Starting directory"}
         }, "required": ["pattern"]}},
        {"name": "ls", "description": "List directory contents",
         "parameters": {"type": "object", "properties": {
             "path": {"type": "string", "description": "Directory path"}
         }, "required": ["path"]}},
        # Execution & process management
        {"name": "exec", "description": "Run a shell command (pty available for TTY-required CLIs)",
         "parameters": {"type": "object", "properties": {
             "command": {"type": "string", "description": "Shell command to execute"}
         }, "required": ["command"]}},
        {"name": "process", "description": "Manage long-running background terminal tasks",
         "parameters": {"type": "object", "properties": {
             "action": {"type": "string", "enum": ["list", "logs", "kill"],
                        "description": "Action to perform"},
             "pid": {"type": "integer", "description": "Process ID (for logs/kill)"}
         }, "required": ["action"]}},
        # Web & browser
        {"name": "web_search", "description": "Search the web via Brave/Google Search API",
         "parameters": {"type": "object", "properties": {
             "query": {"type": "string", "description": "Search query"}
         }, "required": ["query"]}},
        {"name": "web_fetch", "description": "Fetch and extract readable content from a URL",
         "parameters": {"type": "object", "properties": {
             "url": {"type": "string", "description": "URL to fetch"}
         }, "required": ["url"]}},
        {"name": "browser", "description": "Control a Chrome instance via CDP",
         "parameters": {"type": "object", "properties": {
             "action": {"type": "string", "enum": ["goto", "click", "type", "screenshot", "act"],
                        "description": "Browser action"},
             "target": {"type": "string", "description": "Element selector or URL"},
             "value": {"type": "string", "description": "Value to type or act prompt"}
         }, "required": ["action"]}},
        # Multi-agent / sessions
        {"name": "sessions_spawn", "description": "Spawn a specialized sub-agent for a background task",
         "parameters": {"type": "object", "properties": {
             "task": {"type": "string", "description": "Task description for the sub-agent"},
             "agentId": {"type": "string", "description": "Agent role from IDENTITY.md"},
             "model": {"type": "string", "description": "Model to use for sub-agent"}
         }, "required": ["task"]}},
        {"name": "sessions_send", "description": "Send a message to another active session",
         "parameters": {"type": "object", "properties": {
             "sessionKey": {"type": "string", "description": "Target session key"},
             "message": {"type": "string", "description": "Message content"}
         }, "required": ["sessionKey", "message"]}},
        {"name": "sessions_yield", "description": "Pause and wait for a sub-agent to return a result",
         "parameters": {"type": "object", "properties": {
             "sessionKey": {"type": "string", "description": "Session key to wait for"},
             "timeout": {"type": "integer", "description": "Timeout in seconds"}
         }, "required": ["sessionKey"]}},
        # Memory & context
        {"name": "memory_search", "description": "Vector search over .openclaw/memory/ and Memory.md",
         "parameters": {"type": "object", "properties": {
             "query": {"type": "string", "description": "Search query for memory"}
         }, "required": ["query"]}},
        {"name": "memory_get", "description": "Retrieve a specific key-value preference",
         "parameters": {"type": "object", "properties": {
             "key": {"type": "string", "description": "Key to retrieve (e.g. coding_style)"}
         }, "required": ["key"]}},
        # Vision & media
        {"name": "image", "description": "Multi-modal analysis of a local image or screenshot",
         "parameters": {"type": "object", "properties": {
             "image_path": {"type": "string", "description": "Path to the image file"},
             "prompt": {"type": "string", "description": "Analysis prompt"}
         }, "required": ["image_path"]}},
        {"name": "canvas", "description": "Visual diagramming tool",
         "parameters": {"type": "object", "properties": {
             "action": {"type": "string", "enum": ["generate", "list", "update"],
                        "description": "Canvas action"},
             "prompt": {"type": "string", "description": "Diagram description"}
         }, "required": ["action", "prompt"]}},
        # External messaging
        {"name": "message", "description": "Send a message to an external channel (Slack, Discord, etc.)",
         "parameters": {"type": "object", "properties": {
             "channel": {"type": "string", "description": "Channel identifier"},
             "content": {"type": "string", "description": "Message content"}
         }, "required": ["channel", "content"]}},
    ]

    # Code snippets for realistic tool call/result padding
    CODE_SNIPPETS = [
        '''def process_data(items: list[dict]) -> list[dict]:
    """Process and validate data items."""
    results = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if "id" not in item or "value" not in item:
            continue
        processed = {
            "id": item["id"],
            "value": float(item["value"]),
            "normalized": float(item["value"]) / 100.0,
            "status": "valid" if float(item["value"]) > 0 else "invalid",
        }
        results.append(processed)
    return results''',
        '''class DatabaseConnection:
    """Manages database connection pooling."""
    def __init__(self, host: str, port: int = 5432, max_connections: int = 10):
        self.host = host
        self.port = port
        self.max_connections = max_connections
        self._pool = []
        self._active = 0

    async def acquire(self):
        if self._pool:
            conn = self._pool.pop()
            self._active += 1
            return conn
        if self._active < self.max_connections:
            conn = await self._create_connection()
            self._active += 1
            return conn
        raise RuntimeError("Connection pool exhausted")

    async def release(self, conn):
        self._active -= 1
        self._pool.append(conn)''',
        '''import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def find_config(start_dir: str, filename: str = "config.yaml") -> Optional[Path]:
    """Walk up directory tree to find configuration file."""
    current = Path(start_dir).resolve()
    while current != current.parent:
        candidate = current / filename
        if candidate.exists():
            logger.info(f"Found config at {candidate}")
            return candidate
        current = current.parent
    logger.warning(f"Config {filename} not found starting from {start_dir}")
    return None''',
        '''async function fetchWithRetry(url, options = {}, maxRetries = 3) {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const response = await fetch(url, {
        ...options,
        signal: AbortSignal.timeout(10000),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      if (attempt === maxRetries - 1) throw error;
      await new Promise(r => setTimeout(r, 1000 * Math.pow(2, attempt)));
    }
  }
}''',
        '''#include <vector>
#include <algorithm>
#include <numeric>

template<typename T>
class SlidingWindow {
    std::vector<T> buffer_;
    size_t capacity_;
    size_t head_ = 0;
    size_t count_ = 0;
public:
    explicit SlidingWindow(size_t capacity) : buffer_(capacity), capacity_(capacity) {}

    void push(T value) {
        buffer_[head_] = std::move(value);
        head_ = (head_ + 1) % capacity_;
        if (count_ < capacity_) ++count_;
    }

    T average() const {
        if (count_ == 0) return T{};
        return std::accumulate(buffer_.begin(), buffer_.begin() + count_, T{}) / count_;
    }
};''',
    ]

    # Additional longer code snippets for 48K/64K samples
    LONG_CODE_SNIPPETS = [
        '''class EventBus:
    """Publish-subscribe event system with async support."""
    def __init__(self):
        self._handlers: dict[str, list[callable]] = {}
        self._middleware: list[callable] = []

    def on(self, event: str, handler: callable):
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)
        return self

    def use(self, middleware: callable):
        self._middleware.append(middleware)
        return self

    async def emit(self, event: str, data: dict = None):
        context = {"event": event, "data": data or {}, "cancelled": False}
        for mw in self._middleware:
            await mw(context)
            if context["cancelled"]:
                return
        for handler in self._handlers.get(event, []):
            await handler(context["data"])

    def off(self, event: str, handler: callable = None):
        if handler is None:
            self._handlers.pop(event, None)
        elif event in self._handlers:
            self._handlers[event] = [h for h in self._handlers[event] if h != handler]''',
        '''from dataclasses import dataclass, field
from typing import Generic, TypeVar
from collections import defaultdict
import heapq

T = TypeVar("T")

@dataclass(order=True)
class PriorityItem(Generic[T]):
    priority: int
    item: T = field(compare=False)

class TaskScheduler:
    """Priority-based task scheduler with dependency tracking."""
    def __init__(self):
        self._queue: list[PriorityItem] = []
        self._dependencies: dict[str, set[str]] = defaultdict(set)
        self._completed: set[str] = set()
        self._blocked: dict[str, set[str]] = defaultdict(set)

    def add_task(self, task_id: str, priority: int = 0, depends_on: list[str] = None):
        if depends_on:
            pending = {d for d in depends_on if d not in self._completed}
            if pending:
                self._blocked[task_id] = pending
                self._dependencies[task_id] = pending
                for dep in pending:
                    self._blocked.setdefault(dep, set())
                return
        heapq.heappush(self._queue, PriorityItem(priority, task_id))

    def complete(self, task_id: str):
        self._completed.add(task_id)
        for blocked_task, deps in list(self._blocked.items()):
            deps.discard(task_id)
            if not deps and blocked_task not in self._completed:
                orig_deps = self._dependencies.get(blocked_task, set())
                heapq.heappush(self._queue, PriorityItem(0, blocked_task))
                del self._blocked[blocked_task]

    def next(self) -> str | None:
        while self._queue:
            item = heapq.heappop(self._queue)
            if item.item not in self._completed:
                return item.item
        return None''',
        '''// React component with complex state management
import React, { useReducer, useCallback, useMemo, useEffect } from "react";

interface AppState {
  items: Array<{ id: string; name: string; status: "pending" | "active" | "done" }>;
  filter: "all" | "pending" | "active" | "done";
  searchQuery: string;
  selectedIds: Set<string>;
  isLoading: boolean;
  error: string | null;
}

type Action =
  | { type: "SET_ITEMS"; payload: AppState["items"] }
  | { type: "UPDATE_STATUS"; id: string; status: AppState["items"][0]["status"] }
  | { type: "SET_FILTER"; filter: AppState["filter"] }
  | { type: "TOGGLE_SELECT"; id: string }
  | { type: "SET_SEARCH"; query: string }
  | { type: "SET_LOADING"; loading: boolean }
  | { type: "SET_ERROR"; error: string | null }
  | { type: "BULK_UPDATE"; ids: string[]; status: AppState["items"][0]["status"] };

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case "SET_ITEMS":
      return { ...state, items: action.payload, isLoading: false };
    case "UPDATE_STATUS":
      return {
        ...state,
        items: state.items.map((item) =>
          item.id === action.id ? { ...item, status: action.status } : item
        ),
      };
    case "SET_FILTER":
      return { ...state, filter: action.filter };
    case "TOGGLE_SELECT": {
      const newSelected = new Set(state.selectedIds);
      newSelected.has(action.id) ? newSelected.delete(action.id) : newSelected.add(action.id);
      return { ...state, selectedIds: newSelected };
    }
    case "SET_SEARCH":
      return { ...state, searchQuery: action.query };
    case "BULK_UPDATE":
      return {
        ...state,
        items: state.items.map((item) =>
          action.ids.includes(item.id) ? { ...item, status: action.status } : item
        ),
        selectedIds: new Set(),
      };
    default:
      return state;
  }
}''',
        '''# Kubernetes deployment manifest for a microservice
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
  namespace: production
  labels:
    app: api-gateway
    version: v2.1.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: api-gateway
  template:
    metadata:
      labels:
        app: api-gateway
        version: v2.1.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
    spec:
      serviceAccountName: api-gateway-sa
      containers:
      - name: api-gateway
        image: registry.internal/api-gateway:v2.1.0
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9090
          name: metrics
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /healthz
            port: http
          initialDelaySeconds: 15
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /readyz
            port: http
          initialDelaySeconds: 5
          periodSeconds: 5
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: api-gateway-secrets
              key: database-url
        - name: REDIS_HOST
          value: "redis-cluster.production.svc.cluster.local"
        - name: LOG_LEVEL
          value: "info"''',
    ]

    # Tool call/result conversation fragments
    # ── OpenClaw-style tool call / result exchanges for history padding ──────
    TOOL_CALL_EXCHANGES = [
        (
            {"name": "read", "arguments": {"file_path": "src/main.py"}},
            "File contents:\n```python\nimport sys\nfrom pathlib import Path\n\ndef main():\n    parser = argparse.ArgumentParser()\n    parser.add_argument('--config', type=str, required=True)\n    args = parser.parse_args()\n    config = load_config(args.config)\n    app = Application(config)\n    app.run()\n\nif __name__ == '__main__':\n    main()\n```"
        ),
        (
            {"name": "grep", "arguments": {"pattern": "def\\s+\\w+\\(.*\\)\\s*->", "path": "src/", "include": "*.py"}},
            "Found 15 matches:\nsrc/main.py:4: def main() -> None:\nsrc/utils.py:12: def load_config(path: str) -> dict:\nsrc/utils.py:28: def validate_schema(data: dict) -> bool:\nsrc/db.py:8: def connect(url: str) -> Connection:\nsrc/db.py:45: def execute_query(conn: Connection, query: str) -> list:"
        ),
        (
            {"name": "ls", "arguments": {"path": "."}},
            "Contents:\n  src/\n  tests/\n  docs/\n  config.yaml\n  pyproject.toml\n  README.md\n  .gitignore\n  Makefile"
        ),
        (
            {"name": "exec", "arguments": {"command": "python -m pytest tests/ -v --tb=short"}},
            "===== test session starts =====\ncollected 24 items\ntests/test_main.py::test_init PASSED\ntests/test_main.py::test_config_loading PASSED\ntests/test_utils.py::test_validate_schema PASSED\ntests/test_utils.py::test_load_config FAILED\ntests/test_db.py::test_connection PASSED\n===== 1 failed, 23 passed in 2.34s ====="
        ),
        (
            {"name": "find", "arguments": {"pattern": "*.py", "path": "src/"}},
            "Found 12 files:\n  src/main.py\n  src/utils.py\n  src/db.py\n  src/api.py\n  src/config.py\n  src/auth.py\n  src/models.py\n  src/routes.py\n  src/middleware.py\n  src/cache.py\n  src/tasks.py\n  src/schema.py"
        ),
        (
            {"name": "edit", "arguments": {"file_path": "src/utils.py", "search_string": "import os\nimport sys", "replace_string": "import sys"}},
            "Successfully replaced text in src/utils.py (1 occurrence)"
        ),
        (
            {"name": "write", "arguments": {"file_path": "tests/test_new.py", "content": "import pytest\nfrom src.utils import validate_schema\n\ndef test_empty_schema():\n    assert validate_schema({}) is False\n"}},
            "File written successfully: tests/test_new.py (5 lines)"
        ),
        (
            {"name": "web_search", "arguments": {"query": "python asyncio best practices 2026"}},
            "Results:\n1. 'Modern Asyncio Patterns in Python 3.12+' - realpython.com\n2. 'Asyncio Task Groups and Exception Handling' - docs.python.org\n3. 'Performance Tips for Python Async IO' - stackoverflow.com"
        ),
        (
            {"name": "web_fetch", "arguments": {"url": "https://docs.python.org/3/library/asyncio-task.html"}},
            "Fetched content (4.2 KB):\n# Coroutines and Tasks\nThis section outlines high-level asyncio APIs to work with coroutines and Tasks.\n## Coroutines\nCoroutines declared with the async/await syntax..."
        ),
        (
            {"name": "process", "arguments": {"action": "list"}},
            "Active processes:\n  PID 12847: python -m http.server 8080 (running, 45m)\n  PID 13201: npm run dev (running, 12m)"
        ),
        (
            {"name": "memory_search", "arguments": {"query": "database migration strategy"}},
            "Found 3 relevant entries:\n  [2026-03-15] Decided to use Alembic for migrations\n  [2026-03-20] PostgreSQL 16 as primary DB\n  [2026-03-22] Schema versioning follows semver"
        ),
        (
            {"name": "sessions_spawn", "arguments": {"task": "Analyze the test coverage report and identify untested modules", "agentId": "analyzer"}},
            "Spawned sub-agent session: ses_abc123 (agent: analyzer)\nTask: Analyze the test coverage report and identify untested modules"
        ),
    ]

    # ── Final requests: the model MUST respond with the correct tool call ────
    FINAL_REQUESTS = [
        ("Read the file src/config.py so I can review the settings.",
         {"name": "read", "arguments": {"file_path": "src/config.py"}}),
        ("Search for all TODO comments in the source code.",
         {"name": "grep", "arguments": {"pattern": "TODO|FIXME|HACK", "path": "src/", "include": "*.py"}}),
        ("Replace the deprecated API call in src/api.py — change requests.get(url) to httpx.get(url).",
         {"name": "edit", "arguments": {"file_path": "src/api.py", "search_string": "requests.get(url)", "replace_string": "httpx.get(url)"}}),
        ("Run the linter on the entire project.",
         {"name": "exec", "arguments": {"command": "python -m ruff check src/ --fix"}}),
        ("Find all YAML config files in the project.",
         {"name": "find", "arguments": {"pattern": "*.yaml", "path": "."}}),
        ("Create a new test file for the config module.",
         {"name": "write", "arguments": {"file_path": "tests/test_config.py", "content": "import pytest\nfrom src.config import Config\n\ndef test_default_config():\n    cfg = Config()\n    assert cfg.debug is False\n"}}),
        ("List all files in the deployment directory.",
         {"name": "ls", "arguments": {"path": "deploy/"}}),
        ("Apply this patch to fix the import ordering issue.",
         {"name": "apply_patch", "arguments": {"patch_content": "--- a/src/main.py\n+++ b/src/main.py\n@@ -1,3 +1,3 @@\n-import sys\n-import os\n+import os\n+import sys\n"}}),
        ("Search the web for the latest OpenVINO GenAI release notes.",
         {"name": "web_search", "arguments": {"query": "OpenVINO GenAI 2026 release notes"}}),
        ("Fetch the Python asyncio documentation page.",
         {"name": "web_fetch", "arguments": {"url": "https://docs.python.org/3/library/asyncio.html"}}),
        ("Check on the background dev server processes.",
         {"name": "process", "arguments": {"action": "list"}}),
        ("Spawn a sub-agent to refactor the database module while I continue working.",
         {"name": "sessions_spawn", "arguments": {"task": "Refactor src/db.py to use async connection pooling with SQLAlchemy 2.0"}}),
    ]

    # ── Compact system prompt for calibration ───────────────────────────────
    # The full OpenClaw prompt with 20 tools + instructions is ~1,865 tokens.
    # At MAX_SEQ_LEN=2048, that would leave almost no room for tool-call exchanges.
    # Instead, we use 8 CORE tools in compact JSON (~500 tokens total system prompt)
    # so ~1,500 tokens remain for 3-5 complete tool-call exchanges.
    #
    # The 8 core tools cover the most common operations in agentic coding:
    COMPACT_TOOLS = [
        {"name": "read", "description": "Read a file", "parameters": {"type": "object",
         "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}},
        {"name": "write", "description": "Write/create a file", "parameters": {"type": "object",
         "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}, "required": ["file_path", "content"]}},
        {"name": "edit", "description": "String replacement in a file", "parameters": {"type": "object",
         "properties": {"file_path": {"type": "string"}, "search_string": {"type": "string"}, "replace_string": {"type": "string"}}, "required": ["file_path", "search_string", "replace_string"]}},
        {"name": "grep", "description": "Search file contents", "parameters": {"type": "object",
         "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}, "include": {"type": "string"}}, "required": ["pattern"]}},
        {"name": "ls", "description": "List directory", "parameters": {"type": "object",
         "properties": {"path": {"type": "string"}}, "required": ["path"]}},
        {"name": "exec", "description": "Run shell command", "parameters": {"type": "object",
         "properties": {"command": {"type": "string"}}, "required": ["command"]}},
        {"name": "find", "description": "Find files by glob", "parameters": {"type": "object",
         "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}},
        {"name": "web_search", "description": "Search the web", "parameters": {"type": "object",
         "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    ]

    # Compact JSON (no indent, minimal whitespace) → ~300 tokens for 8 tools
    compact_tools_json = json.dumps(COMPACT_TOOLS, separators=(',', ':'))

    system_prompt = f"""You are an AI assistant inside OpenClaw. You perform actions using tools.

* Working Directory: /workspace/project
* Tool Policy: auto-approve-reads

<tools>
{compact_tools_json}
</tools>

When you need to use a tool, respond with:
<tool_call>
{{"name": "tool_name", "arguments": {{"param1": "value1"}}}}
</tool_call>

Execute tool calls directly. Do not narrate routine operations."""

    # ── Memory budget for NNCF statistics collection ────────────────────────
    # Observed: 79 GB peak at seq_len=1024 (57 GB model + 22 GB NNCF overhead).
    # Overhead scales ~linearly. At 2048: OOM (tried and failed - 768MB alloc failure).
    # At 1536: est ~90 GB (1.5× overhead) — tight but within 96 GB RAM.
    # System: 96 GB RAM + 225 GB page file.
    # ────────────────────────────────────────────────────────────────────────
    MAX_SEQ_LEN = 1536

    # Measure system prompt token count for diagnostics
    sys_test = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}],
        tokenize=False, add_generation_prompt=False
    )
    sys_tokens = len(tokenizer(sys_test)["input_ids"])
    exchange_budget = MAX_SEQ_LEN - sys_tokens
    print(f"  System prompt: {sys_tokens} tokens → {exchange_budget} tokens available for exchanges")
    print(f"  Generating {num_samples} OpenClaw tool-call calibration samples (max {MAX_SEQ_LEN} tokens)...")
    calibration_data = []

    # Vary exchange counts (1-2 per sample to fit within 1280 token budget)
    # Compact system prompt ≈ 389 tokens → 891 tokens for exchanges
    # Each exchange ≈ 300-500 tokens (user + tool_call + result + code + analysis)
    # 1-2 exchanges → 300-1000 tokens → most fit within budget
    # Use only compact CODE_SNIPPETS (not LONG_CODE_SNIPPETS) to save space
    EXCHANGE_RANGE = (1, 2)

    # Use only compact snippets to keep exchanges short enough for 1280 budget
    all_snippets = CODE_SNIPPETS

    for i in range(num_samples):
        random.seed(42 + i)  # Reproducible but varied

        messages = [{"role": "system", "content": system_prompt}]

        # Build multi-turn history with tool calls and results
        num_exchanges = random.randint(*EXCHANGE_RANGE)
        used_exchanges = random.choices(TOOL_CALL_EXCHANGES, k=num_exchanges)

        for tc_args, result_text in used_exchanges:
            # User request that led to this tool call
            messages.append({"role": "user", "content": f"Please help me with the codebase. " +
                           random.choice([
                               "I need to understand the project structure.",
                               "Let's fix the failing tests.",
                               "I want to refactor the database module.",
                               "Can you help debug this issue?",
                               "Let's improve the error handling.",
                               "I need to add a new feature for data export.",
                               "Please check the code quality.",
                               "Let's optimize the API response time.",
                               "Can you review the security of this module?",
                               "I need to add proper logging throughout.",
                               "Let's set up the CI/CD pipeline configuration.",
                               "Can you help migrate this to async/await?",
                           ])})

            # Assistant tool call
            tc_json = json.dumps(tc_args)
            messages.append({"role": "assistant", "content": f"<tool_call>\n{tc_json}\n</tool_call>"})

            # Tool result (with code snippet padding for realism)
            code_pad = random.choice(all_snippets)
            full_result = result_text + "\n\nRelated code context:\n```\n" + code_pad + "\n```"
            messages.append({"role": "user", "content": f"[Tool Result]\n{full_result}"})

            # Assistant analysis
            messages.append({"role": "assistant", "content": random.choice([
                "I see. Let me continue examining the codebase to address your request.",
                "Based on this result, I'll proceed with the next step.",
                "Got it. Let me check another part of the code that might be related.",
                "This confirms my suspicion. Let me investigate further.",
                "The output shows some areas that need attention. Let me look deeper.",
                "Good. I found what I was looking for. Let me now check the related module.",
                "I notice a pattern here. Let me verify by checking a few more files.",
                "The test results suggest we need to update the implementation. Let me trace the issue.",
            ])})

        # Final user request — this is what the model must respond to correctly
        final_req, expected_tc = random.choice(FINAL_REQUESTS)
        messages.append({"role": "user", "content": final_req})

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Add the expected correct response for the calibration
        text += f"<tool_call>\n{json.dumps(expected_tc)}\n</tool_call>"

        # Measure before truncation for diagnostics
        pre_trunc_ids = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"]
        pre_trunc_len = pre_trunc_ids.shape[1]

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
        token_len = inputs["input_ids"].shape[1]
        was_truncated = pre_trunc_len > MAX_SEQ_LEN

        # Add position_ids and beam_idx required by text-generation-with-past OV model
        import torch
        inputs["position_ids"] = torch.arange(token_len).unsqueeze(0)
        inputs["beam_idx"] = torch.zeros(1, dtype=torch.int32)

        calibration_data.append(dict(inputs))

        if i < 8:  # Print first few samples
            trunc_flag = " [TRUNCATED]" if was_truncated else ""
            print(f"    Sample {i}: {token_len}/{pre_trunc_len} tokens, {num_exchanges} exchanges{trunc_flag}")

    avg_len = sum(d["input_ids"].shape[1] for d in calibration_data) / len(calibration_data)
    n_truncated = sum(1 for d in calibration_data if d["input_ids"].shape[1] == MAX_SEQ_LEN)
    print(f"  Collected {len(calibration_data)} samples (avg {avg_len:.0f} tokens, max {MAX_SEQ_LEN})")
    print(f"  Truncated: {n_truncated}/{len(calibration_data)} samples")

    # ── Save generated prompts to disk for inspection ─────────────────────────
    # Regenerate the text prompts (same seeds) and write to calibration_prompts/
    prompts_dir = BASE_MODEL_DIR / "calibration_prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Saving generated prompts to: {prompts_dir}")

    prompt_manifest = []
    for i in range(len(calibration_data)):
        random.seed(42 + i)  # Same seed as above → identical content

        messages = [{"role": "system", "content": system_prompt}]
        num_exchanges = random.randint(*EXCHANGE_RANGE)
        used_exchanges = random.choices(TOOL_CALL_EXCHANGES, k=num_exchanges)

        for tc_args, result_text in used_exchanges:
            messages.append({"role": "user", "content": f"Please help me with the codebase. " +
                           random.choice([
                               "I need to understand the project structure.",
                               "Let's fix the failing tests.",
                               "I want to refactor the database module.",
                               "Can you help debug this issue?",
                               "Let's improve the error handling.",
                               "I need to add a new feature for data export.",
                               "Please check the code quality.",
                               "Let's optimize the API response time.",
                               "Can you review the security of this module?",
                               "I need to add proper logging throughout.",
                               "Let's set up the CI/CD pipeline configuration.",
                               "Can you help migrate this to async/await?",
                           ])})
            tc_json = json.dumps(tc_args)
            messages.append({"role": "assistant", "content": f"<tool_call>\n{tc_json}\n</tool_call>"})
            code_pad = random.choice(all_snippets)
            full_result = result_text + "\n\nRelated code context:\n```\n" + code_pad + "\n```"
            messages.append({"role": "user", "content": f"[Tool Result]\n{full_result}"})
            messages.append({"role": "assistant", "content": random.choice([
                "I see. Let me continue examining the codebase to address your request.",
                "Based on this result, I'll proceed with the next step.",
                "Got it. Let me check another part of the code that might be related.",
                "This confirms my suspicion. Let me investigate further.",
                "The output shows some areas that need attention. Let me look deeper.",
                "Good. I found what I was looking for. Let me now check the related module.",
                "I notice a pattern here. Let me verify by checking a few more files.",
                "The test results suggest we need to update the implementation. Let me trace the issue.",
            ])})

        final_req, expected_tc = random.choice(FINAL_REQUESTS)
        messages.append({"role": "user", "content": final_req})

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text += f"<tool_call>\n{json.dumps(expected_tc)}\n</tool_call>"

        token_len = calibration_data[i]["input_ids"].shape[1]
        prompt_file = prompts_dir / f"sample_{i:03d}.txt"
        prompt_file.write_text(text, encoding="utf-8")

        prompt_manifest.append({
            "sample": i,
            "file": prompt_file.name,
            "token_len": token_len,
            "num_exchanges": num_exchanges,
            "final_request": final_req,
            "expected_tool_call": expected_tc,
            "char_len": len(text),
        })

    # Write manifest summary
    manifest_file = prompts_dir / "manifest.json"
    manifest_file.write_text(json.dumps(prompt_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved {len(prompt_manifest)} prompt files + manifest.json to {prompts_dir}")

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
    parser.add_argument("--group-size", type=int, default=128,
                        help="Quantization group size (default: 128, only gs=128 works on Arc B390 GPU)")
    parser.add_argument("--ratio", type=float, default=0.8,
                        help="INT4 vs INT8 ratio (default: 0.8, lower = more INT8 for sensitive layers)")
    parser.add_argument("--dataset", type=str, default="wikitext2",
                        choices=["wikitext2", "tool_calling", "long_tool_calling"],
                        help="Calibration dataset (default: wikitext2)")
    parser.add_argument("--subset-size", type=int, default=128,
                        help="Number of calibration samples (default: 128)")
    parser.add_argument("--sensitivity-metric", type=str, default="max_activation_variance",
                        choices=["max_activation_variance", "mean_activation_variance",
                                 "mean_activation_magnitude", "hessian_input_activation",
                                 "weight_quantization_error"],
                        help="Sensitivity metric for mixed-precision (default: max_activation_variance)")
    parser.add_argument("--local-model", type=str, default=None,
                        help="Path to local HuggingFace model weights (skip download)")
    parser.add_argument("--awq", action="store_true", default=False,
                        help="Enable AWQ (BROKEN for MoE models in NNCF 2.19.0, use with caution)")
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
