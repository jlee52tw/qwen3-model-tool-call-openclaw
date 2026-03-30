#!/usr/bin/env python3
"""
LLM Code Assistant with OpenVINO GenAI — Standalone Project

Uses openvino_genai.LLMPipeline (C++ Generate API) for fast inference.
Supports code generation, bug fixing, security auditing, and interactive chat.

Default model: Qwen/Qwen3-Coder-30B-A3B-Instruct (MoE: 30B total, 3.3B active per token)
"""

import argparse
import ast
import json
import re
import subprocess
import sys
import time
import os
import statistics
import urllib.error
import urllib.request
from pathlib import Path

# Pre-converted OpenVINO models on HuggingFace
PRECONVERTED_MODELS = {
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": "OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int4-ov",
}

DEFAULT_SYSTEM_MESSAGE = "You are an expert programming assistant. Write clean, efficient code."


# ─── Model Download / Conversion ────────────────────────────────────────────


def download_model(model_id: str, output_dir: str):
    """Download a pre-converted OpenVINO model from HuggingFace."""
    from huggingface_hub import snapshot_download

    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.glob("*.xml")):
        print(f"Model already exists at {output_dir}, skipping download.")
        return output_dir

    repo_id = PRECONVERTED_MODELS.get(model_id, model_id)
    print(f"Downloading pre-converted model: {repo_id}")
    print(f"  Output: {output_dir}")
    print()

    start = time.time()
    snapshot_download(repo_id, local_dir=str(output_dir))
    elapsed = time.time() - start

    print(f"\nDownload completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    total_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
    print(f"Model size: {total_size / (1024**3):.2f} GB")
    return output_dir


def convert_model(model_id: str, output_dir: str, weight_format: str = "int4",
                  trust_remote_code: bool = False):
    """Convert a HuggingFace model to OpenVINO IR using optimum-cli."""
    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.glob("*.xml")):
        print(f"Model already exists at {output_dir}, skipping conversion.")
        return output_dir

    output_path.mkdir(parents=True, exist_ok=True)

    venv_dir = os.environ.get("VIRTUAL_ENV", "")
    if venv_dir:
        cli_path = os.path.join(venv_dir, "Scripts", "optimum-cli")
    else:
        cli_path = "optimum-cli"

    cmd = [
        cli_path, "export", "openvino",
        "--model", model_id,
        "--task", "text-generation-with-past",
        "--weight-format", weight_format,
    ]
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    cmd.append(str(output_dir))

    print(f"Converting model: {model_id}")
    print(f"  Weight format: {weight_format}")
    print(f"  Output: {output_dir}")
    print(f"  Command: {' '.join(cmd)}")

    start = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\nERROR: Conversion failed with return code {result.returncode}")
        sys.exit(1)

    print(f"\nConversion completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    total_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
    print(f"Model size: {total_size / (1024**3):.2f} GB")
    return output_dir


# ─── Core Generation Helpers ────────────────────────────────────────────────


def load_pipeline(model_dir: str, device: str):
    """Load the OpenVINO GenAI pipeline."""
    import openvino_genai as ov_genai
    print(f"Loading model from {model_dir}")
    print(f"Device: {device}")
    pipe = ov_genai.LLMPipeline(str(model_dir), device)
    print("Model loaded successfully!")
    return pipe


def generate(pipe, user_text: str, max_tokens: int = 512,
             do_sample: bool = True, temperature: float = 0.2, top_p: float = 0.95):
    """Generate text using the pipeline. Returns (text, perf_info_dict)."""
    start = time.perf_counter()
    result = pipe.generate(
        user_text,
        max_new_tokens=max_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )
    elapsed = time.perf_counter() - start
    text = str(result)

    perf = {}
    if hasattr(result, "perf_metrics"):
        metrics = result.perf_metrics
        perf["tokens"] = metrics.get_num_generated_tokens()
        perf["throughput"] = metrics.get_throughput().mean
        perf["elapsed"] = elapsed
        print(f"  Generated {perf['tokens']} tokens in {elapsed:.1f}s "
              f"({perf['throughput']:.1f} tok/s)")
    else:
        approx_tokens = int(len(text.split()) / 0.75)
        tps = approx_tokens / elapsed if elapsed > 0 else 0
        perf["tokens"] = approx_tokens
        perf["throughput"] = tps
        perf["elapsed"] = elapsed
        print(f"  Generated ~{approx_tokens} tokens in {elapsed:.1f}s ({tps:.1f} tok/s)")

    return text, perf


def generate_code(pipe, prompt: str, max_tokens: int = 512,
                  system_message: str = DEFAULT_SYSTEM_MESSAGE):
    """Single-turn code generation with system message via chat API."""
    pipe.start_chat(system_message=system_message)
    try:
        return generate(pipe, prompt, max_tokens=max_tokens)
    finally:
        pipe.finish_chat()


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ─── Task: Generate ─────────────────────────────────────────────────────────


def task_generate(pipe, prompt: str, max_tokens: int, no_think: bool):
    """Single prompt code generation with streaming output."""
    if no_think:
        prompt = prompt + " /no_think"

    print(f"\nPrompt: {prompt}")
    print(f"Max tokens: {max_tokens}")
    print("=" * 60)

    text, perf = generate_code(pipe, prompt, max_tokens=max_tokens)
    text = strip_thinking(text)
    print(text)
    print("=" * 60)


# ─── Task: Fix Bugs ─────────────────────────────────────────────────────────


BROKEN_CODE_SAMPLE = '''
def merge_sorted_lists(list1, list2):
    """Merge two sorted lists into one sorted list."""
    result = []
    i = j = 0
    while i <= len(list1) and j <= len(list2):  # Bug 1: should be <, not <=
        if list1[i] < list2[j]:
            result.append(list1[i])
            i += 1
        else:
            result.append(list2[j])
            j += 1
    result += list1[i:]
    result += list2[j:]
    return result

def find_duplicates(lst):
    """Return a list of duplicate elements."""
    seen = set()
    duplicates = []
    for item in lst:
        if item in seen:
            duplicates.append(item)  # Bug 2: appends every time, not unique
        seen.add(item)
    return duplicates

def flatten(nested_list):
    """Flatten a nested list."""
    result = []
    for item in nested_list:
        if type(item) == list:  # Bug 3: doesn't handle tuples/other iterables
            result.extend(item)  # Bug 4: extend instead of recursive flatten
        else:
            result.append(item)
    return result
'''


def task_fix(pipe, code_input: str = None, max_tokens: int = 1024, no_think: bool = False):
    """Bug detection and code correction."""
    code = code_input or BROKEN_CODE_SAMPLE

    prompt = f"""Fix ALL bugs in the following Python code. For each bug:
1. State what the bug is
2. Show the fix

Then provide the complete corrected code in a ```python block.

```python
{code}
```"""

    if no_think:
        prompt += " /no_think"

    print("Analyzing code for bugs...\n")

    text, perf = generate_code(pipe, prompt, max_tokens=max_tokens)
    text = strip_thinking(text)
    print(text)

    # Extract and verify fixed code
    fixed_code = None
    if "```python" in text:
        fixed_code = text.split("```python")[1].split("```")[0].strip()
    elif "```" in text:
        fixed_code = text.split("```")[1].split("```")[0].strip()

    if fixed_code:
        try:
            ast.parse(fixed_code)
            print("\n  AST syntax check: PASSED")
        except SyntaxError as e:
            print(f"\n  AST syntax check: FAILED — {e}")
    else:
        print("\n  Could not extract code block from model response.")


# ─── Task: Security Audit ───────────────────────────────────────────────────


VULNERABLE_CODE_SAMPLE = '''
import os
import sqlite3

# Hardcoded credentials (CWE-798)
DB_PASSWORD = "admin123"
API_KEY = "sk-proj-abc123secret456"

def get_user(username):
    """Fetch user from database."""
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    # SQL Injection (CWE-89)
    query = f"SELECT * FROM users WHERE username = \'{username}\'"
    cursor.execute(query)
    return cursor.fetchone()

def read_file(filename):
    """Read a file from the reports directory."""
    # Path Traversal (CWE-22)
    filepath = f"/var/reports/{filename}"
    with open(filepath, "r") as f:
        return f.read()

def run_diagnostic(host):
    """Ping a host to check connectivity."""
    # Command Injection (CWE-78)
    result = os.system(f"ping -c 1 {host}")
    return result
'''


def task_audit(pipe, code_input: str = None, max_tokens: int = 1024,
               max_attempts: int = 3, no_think: bool = False):
    """Security audit agent with self-correction loop."""
    code = code_input or VULNERABLE_CODE_SAMPLE

    print("Security Audit Agent starting...\n")
    syntax_errors = []

    for attempt in range(1, max_attempts + 1):
        print(f"{'='*60}")
        print(f"Attempt {attempt}/{max_attempts}")
        print(f"{'='*60}")

        if not syntax_errors:
            user_msg = f"Analyze this code for security vulnerabilities and provide the fixed version:\n\n```python\n{code}\n```"
            system_msg = (
                "You are a security code auditor. Analyze the Python code for security vulnerabilities.\n\n"
                "Your response MUST have two parts:\n"
                "1. A vulnerability report listing each issue with: vulnerability name, CWE ID, severity.\n"
                "2. The complete fixed code in a ```python block with all vulnerabilities resolved.\n\n"
                "Use parameterized queries, environment variables, path validation, and input sanitization."
            )
        else:
            last_code, last_error = syntax_errors[-1]
            system_msg = "You are a security code auditor. Your previous fix had a syntax error. Fix it."
            user_msg = (
                f"Original vulnerable code:\n```python\n{code}\n```\n\n"
                f"Your previous fix (has syntax error):\n```python\n{last_code}\n```\n\n"
                f"Syntax error: {last_error}\n\nFix the syntax error while keeping all security improvements."
            )

        if no_think:
            user_msg += " /no_think"

        text, perf = generate_code(pipe, user_msg, max_tokens=max_tokens, system_message=system_msg)
        text = strip_thinking(text)

        # Extract fixed code
        fixed_code = None
        if "```python" in text:
            fixed_code = text.split("```python")[1].split("```")[0].strip()
        elif "```" in text:
            fixed_code = text.split("```")[1].split("```")[0].strip()

        if not fixed_code:
            print("  Model did not return a code block.")
            continue

        try:
            ast.parse(fixed_code)
            print("  Fixed code passed syntax verification (AST parse).")
            if attempt > 1:
                print(f"  Model self-corrected after {attempt - 1} syntax error(s).")
            print()

            # Print report
            report = text.split("```")[0].strip() if "```" in text else text
            print(report)
            print("\n  FIXED CODE:")
            print("  " + "-" * 58)
            print(fixed_code)
            print("  " + "-" * 58)
            return
        except SyntaxError as e:
            err_msg = f"SyntaxError at line {e.lineno}: {e.msg}"
            print(f"  Fix has syntax error: {err_msg}")
            syntax_errors.append((fixed_code, err_msg))
            if attempt < max_attempts:
                print("  Feeding error back to model...\n")

    print(f"\n  Agent could not produce valid fix after {max_attempts} attempts.")


# ─── Task: Chat ─────────────────────────────────────────────────────────────


def task_chat(pipe, max_tokens: int, system_prompt: str = None, no_think: bool = False):
    """Interactive multi-turn coding chat."""
    sys_msg = system_prompt or DEFAULT_SYSTEM_MESSAGE

    pipe.start_chat(system_message=sys_msg)

    print(f"\nCode Assistant Chat (max {max_tokens} tokens/response)")
    if no_think:
        print("Thinking mode: OFF (fast responses)")
    else:
        print("Thinking mode: ON (step-by-step reasoning)")
    print("Type 'quit' or 'exit' to end, 'reset' to clear history")
    print("=" * 60)

    turn = 0
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            pipe.finish_chat()
            pipe.start_chat(system_message=sys_msg)
            turn = 0
            print("Chat history cleared.")
            continue

        turn += 1
        message = user_input + " /no_think" if no_think else user_input

        print(f"\nAssistant:")
        text, perf = generate(pipe, message, max_tokens=max_tokens)
        text = strip_thinking(text)
        print(text)

    pipe.finish_chat()


# ─── Task: Benchmark ────────────────────────────────────────────────────────


def task_benchmark(pipe, device: str, max_tokens: int = 512, runs: int = 3,
                   no_think: bool = False):
    """Benchmark generation performance with coding prompts."""
    prompts = [
        "Write a Python function to calculate fibonacci numbers.",
        "Write a binary search function in Python with type hints.",
        "Explain the difference between threading and multiprocessing in Python.",
        "Write a C++ quicksort implementation with comments.",
        "Write a Python decorator that retries a function up to 3 times on exception.",
    ]

    print(f"Device: {device}")
    print(f"Max tokens: {max_tokens}")
    print(f"Runs per prompt: {runs}")
    print(f"Thinking mode: {'OFF' if no_think else 'ON'}")
    print("=" * 80)

    all_results = []

    for i, prompt in enumerate(prompts):
        input_text = prompt + " /no_think" if no_think else prompt

        prompt_results = []
        print(f"\n{'='*80}")
        print(f"Prompt {i+1}/{len(prompts)}: {prompt}")
        print(f"{'='*80}")

        for run in range(runs):
            text, perf = generate_code(pipe, input_text, max_tokens=max_tokens)

            run_data = {
                "prompt": prompt,
                "response": strip_thinking(text),
                "tokens": perf["tokens"],
                "throughput": perf["throughput"],
                "elapsed": perf["elapsed"],
            }
            prompt_results.append(run_data)

            print(f"  Run {run+1}/{runs}: {perf['tokens']} tokens, "
                  f"{perf['elapsed']:.3f}s, {perf['throughput']:.1f} tok/s")

        # Summary for this prompt
        tps_list = [r["throughput"] for r in prompt_results]
        totals = [r["elapsed"] for r in prompt_results]
        token_counts = [r["tokens"] for r in prompt_results]

        print(f"\n  Median: {statistics.median(tps_list):.1f} tok/s, "
              f"total={statistics.median(totals):.3f}s, "
              f"{statistics.median(token_counts):.0f} tokens")

        # Show sample output
        sample = prompt_results[0]["response"]
        if len(sample) > 500:
            sample = sample[:500] + "..."
        print(f"\n  Sample output (run 1):")
        print(f"  {'-'*70}")
        print(f"  {sample}")
        print(f"  {'-'*70}")

        all_results.extend(prompt_results)

    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"Model dir: {pipe.get_tokenizer() if hasattr(pipe, 'get_tokenizer') else 'N/A'}")
    print(f"Device: {device}")
    print(f"Weight format: INT4")
    print(f"Max tokens: {max_tokens}")
    print(f"Total runs: {len(all_results)}")

    all_tps = [r["throughput"] for r in all_results]
    all_total = [r["elapsed"] for r in all_results]
    all_tokens = [r["tokens"] for r in all_results]

    print(f"\nThroughput (tok/s):    median={statistics.median(all_tps):.1f}, "
          f"mean={statistics.mean(all_tps):.1f}, "
          f"min={min(all_tps):.1f}, max={max(all_tps):.1f}")
    print(f"Total generation time: median={statistics.median(all_total):.3f}s, "
          f"mean={statistics.mean(all_total):.3f}s")
    print(f"Tokens per response:   median={statistics.median(all_tokens):.0f}, "
          f"mean={statistics.mean(all_tokens):.0f}")
    print(f"{'='*80}")

    return all_results


# ─── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="LLM Code Assistant with OpenVINO GenAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download pre-converted INT4 model
  python llm_code_assistant.py --task download

  # Convert model to INT4 (if no pre-converted version)
  python llm_code_assistant.py --task convert

  # Generate code from a prompt
  python llm_code_assistant.py --task generate --prompt "Write a binary search in Python"

  # Fix bugs in sample code
  python llm_code_assistant.py --task fix

  # Security audit
  python llm_code_assistant.py --task audit

  # Interactive chat
  python llm_code_assistant.py --task chat

  # Benchmark on GPU
  python llm_code_assistant.py --task benchmark --device GPU --runs 3

  # Non-thinking mode (faster)
  python llm_code_assistant.py --task chat --no-think
        """,
    )

    parser.add_argument("--task",
                        choices=["download", "convert", "generate", "fix", "audit", "chat", "benchmark"],
                        required=True, help="Task to perform")
    parser.add_argument("--model-id", default="Qwen/Qwen3-Coder-30B-A3B-Instruct",
                        help="HuggingFace model ID (default: Qwen/Qwen3-Coder-30B-A3B-Instruct)")
    parser.add_argument("--model-dir",
                        default=r"C:\working\models\Qwen3-Coder-30B-A3B-Instruct\INT4",
                        help="Path to converted model directory")
    parser.add_argument("--device", default="GPU",
                        help="Inference device: GPU, CPU, AUTO (default: GPU)")
    parser.add_argument("--weight-format", default="int4",
                        choices=["fp16", "int8", "int4"],
                        help="Weight compression format (default: int4)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Maximum new tokens to generate (default: 512)")
    parser.add_argument("--prompt", type=str,
                        default="Write a Python function to calculate fibonacci numbers.",
                        help="Input prompt for generate task")
    parser.add_argument("--code-file", type=str, default=None,
                        help="Path to code file for fix/audit tasks (uses built-in sample if omitted)")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of benchmark runs per prompt (default: 3)")
    parser.add_argument("--no-think", action="store_true",
                        help="Disable thinking mode (Qwen3 hybrid thinking)")
    parser.add_argument("--system-prompt", type=str, default=None,
                        help="System prompt for chat mode")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Trust remote code during model conversion")

    args = parser.parse_args()

    if args.task == "download":
        download_model(args.model_id, args.model_dir)
        return

    if args.task == "convert":
        convert_model(args.model_id, args.model_dir, args.weight_format,
                      args.trust_remote_code)
        return

    # All other tasks require loading the model
    pipe = load_pipeline(args.model_dir, args.device)

    # Load code from file if provided
    code_input = None
    if args.code_file:
        with open(args.code_file, "r") as f:
            code_input = f.read()

    if args.task == "generate":
        task_generate(pipe, args.prompt, args.max_tokens, args.no_think)

    elif args.task == "fix":
        task_fix(pipe, code_input, args.max_tokens, args.no_think)

    elif args.task == "audit":
        task_audit(pipe, code_input, args.max_tokens, no_think=args.no_think)

    elif args.task == "chat":
        task_chat(pipe, args.max_tokens, args.system_prompt, args.no_think)

    elif args.task == "benchmark":
        task_benchmark(pipe, args.device, args.max_tokens, args.runs, args.no_think)


if __name__ == "__main__":
    main()
