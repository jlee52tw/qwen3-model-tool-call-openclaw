#!/usr/bin/env python3
"""
Long-Context Benchmark for Qwen3-Coder-30B-A3B-Instruct
========================================================

Tests model intelligence at increasing context lengths using:
  1. Needle-in-a-Haystack (NIAH): Passkey retrieval at various depths/lengths
  2. Tool-Calling Accuracy: Structured tool call output at increasing context
  3. Multi-Needle: Multiple facts scattered throughout context

Supports both:
  - OpenVINO GenAI LLMPipeline (local inference)
  - OVMS OpenAI-compatible API (/v3/chat/completions)

Usage:
    # Test baseline model with GenAI
    python benchmark_context.py --backend genai --model-dir "C:\\working\\models\\Qwen3-Coder-30B-A3B-Instruct\\INT4" --device GPU

    # Test improved model
    python benchmark_context.py --backend genai --model-dir "C:\\working\\models\\Qwen3-Coder-30B-A3B-Instruct\\INT4-HQ" --device GPU

    # Test via OVMS
    python benchmark_context.py --backend ovms --ovms-url http://localhost:8000

    # Compare two models
    python benchmark_context.py --compare "C:\\working\\models\\...\\INT4" "C:\\working\\models\\...\\INT4-HQ"

    # Specific tests only
    python benchmark_context.py --backend genai --model-dir ... --tests niah tool_call

    # Quick test (fewer context lengths)
    python benchmark_context.py --backend genai --model-dir ... --quick
"""

import argparse
import json
import os
import random
import re
import string
import sys
import time
import csv
from pathlib import Path
from datetime import datetime


# ── Constants ──────────────────────────────────────────────────────────────────

# Context lengths to test (in tokens, approximate)
CONTEXT_LENGTHS = [4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768]
QUICK_LENGTHS = [4096, 8192, 16384, 24576, 32768]

# Needle depths (position within context as fraction: 0.0=start, 1.0=end)
NEEDLE_DEPTHS = [0.1, 0.25, 0.5, 0.75, 0.9]
QUICK_DEPTHS = [0.25, 0.5, 0.75]

# Padding text — uses Paul Graham essays style filler
FILLER_PARAGRAPH = (
    "The art of software engineering involves understanding complex systems "
    "and breaking them down into manageable components. Each component must "
    "be designed with clear interfaces and well-defined responsibilities. "
    "Testing ensures that these components work correctly both in isolation "
    "and when integrated together. Documentation helps future developers "
    "understand the design decisions and trade-offs that were made. "
    "Performance optimization should only be done after profiling reveals "
    "actual bottlenecks, not based on assumptions about where time is spent. "
    "Code review is a collaborative process that improves code quality and "
    "spreads knowledge across the team. Version control enables teams to "
    "work on features independently and merge changes safely. Continuous "
    "integration catches errors early by running tests automatically. "
    "Deployment pipelines ensure that software reaches users reliably. "
)


# ── Backend Abstraction ────────────────────────────────────────────────────────

class GenAIBackend:
    """OpenVINO GenAI LLMPipeline backend."""

    def __init__(self, model_dir: str, device: str = "GPU"):
        import openvino_genai as ov_genai
        print(f"[GenAI] Loading model from {model_dir} on {device}...")
        self.pipe = ov_genai.LLMPipeline(str(model_dir), device)
        self.model_dir = model_dir
        print(f"[GenAI] Model loaded.")

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> dict:
        """Generate text. Returns dict with 'text', 'tokens', 'elapsed'."""
        start = time.perf_counter()
        result = self.pipe.generate(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=False,     # Greedy for deterministic evaluation
            temperature=temperature,
        )
        elapsed = time.perf_counter() - start
        text = str(result)

        tokens = 0
        if hasattr(result, "perf_metrics"):
            tokens = result.perf_metrics.get_num_generated_tokens()
        else:
            tokens = len(text.split())  # rough estimate

        return {"text": text, "tokens": tokens, "elapsed": elapsed}

    def chat(self, messages: list, max_tokens: int = 256, temperature: float = 0.0) -> dict:
        """Chat-style generation. messages = [{"role": "...", "content": "..."}]."""
        # Use the tokenizer's apply_chat_template for proper formatting
        tok = self.pipe.get_tokenizer()
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True)
        return self.generate(prompt, max_tokens=max_tokens, temperature=temperature)


class OVMSBackend:
    """OVMS OpenAI-compatible API backend."""

    def __init__(self, url: str = "http://localhost:8000", model_name: str = "Qwen3-Coder-30B-A3B-Instruct"):
        import urllib.request
        self.url = url.rstrip("/")
        self.model_name = model_name
        self.endpoint = f"{self.url}/v3/chat/completions"
        print(f"[OVMS] Endpoint: {self.endpoint}")
        print(f"[OVMS] Model: {self.model_name}")

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> dict:
        """Generate from a raw prompt via chat API."""
        return self.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def chat(self, messages: list, max_tokens: int = 256, temperature: float = 0.0) -> dict:
        """Chat-style generation via OVMS OpenAI API."""
        import urllib.request

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
        )

        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        elapsed = time.perf_counter() - start

        choice = result["choices"][0]
        text = choice["message"]["content"]
        tokens = result.get("usage", {}).get("completion_tokens", len(text.split()))

        return {"text": text, "tokens": tokens, "elapsed": elapsed}


# ── Filler Text Generation ─────────────────────────────────────────────────────

def generate_filler(target_tokens: int) -> str:
    """Generate filler text of approximately target_tokens length."""
    # Rough estimate: 1 token ≈ 4 characters ≈ 0.75 words
    target_chars = target_tokens * 4
    filler = ""
    while len(filler) < target_chars:
        filler += FILLER_PARAGRAPH + "\n\n"
    return filler[:target_chars]


def generate_random_passkey() -> str:
    """Generate a random 6-digit passkey."""
    return "".join(random.choices(string.digits, k=6))


# ── Test 1: Needle-in-a-Haystack ───────────────────────────────────────────────

def test_niah(backend, context_lengths: list, depths: list, max_output_tokens: int = 64) -> list:
    """
    Needle-in-a-Haystack test.
    Inserts a passkey at various positions within padding text.
    Measures retrieval accuracy at each depth × context length.
    """
    print(f"\n{'='*80}")
    print(f"  TEST: Needle-in-a-Haystack (NIAH)")
    print(f"  Context lengths: {context_lengths}")
    print(f"  Depths: {depths}")
    print(f"{'='*80}\n")

    results = []

    for ctx_len in context_lengths:
        for depth in depths:
            passkey = generate_random_passkey()
            needle = f"\n\n**IMPORTANT: The secret passkey is {passkey}. Remember this number.**\n\n"

            # Generate filler, insert needle at specified depth
            # Reserve tokens for: system prompt (~100) + needle (~30) + question (~50)
            filler_tokens = ctx_len - 200
            filler = generate_filler(filler_tokens)

            insert_pos = int(len(filler) * depth)
            # Find a paragraph break near the target position
            para_break = filler.rfind("\n\n", 0, insert_pos)
            if para_break == -1:
                para_break = insert_pos
            context = filler[:para_break] + needle + filler[para_break:]

            question = "What is the secret passkey mentioned in the text above? Reply with ONLY the 6-digit number, nothing else."

            messages = [
                {"role": "system", "content": "You are a precise assistant. Answer questions exactly as asked."},
                {"role": "user", "content": context + "\n\n" + question},
            ]

            print(f"  ctx={ctx_len:>6}, depth={depth:.2f}, passkey={passkey} ... ", end="", flush=True)

            try:
                result = backend.chat(messages, max_tokens=max_output_tokens, temperature=0.0)
                answer = result["text"].strip()

                # Check if passkey is in the answer
                # Clean thinking tags if present
                clean_answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

                # Extract digits from answer
                digits = re.findall(r"\d{6}", clean_answer)
                found_passkey = digits[0] if digits else clean_answer[:20]
                correct = passkey in clean_answer

                status = "PASS" if correct else "FAIL"
                print(f"{status} (answer: {found_passkey}, elapsed: {result['elapsed']:.1f}s)")

                results.append({
                    "test": "niah",
                    "context_length": ctx_len,
                    "depth": depth,
                    "passkey": passkey,
                    "answer": found_passkey,
                    "correct": correct,
                    "elapsed": round(result["elapsed"], 2),
                    "output_tokens": result["tokens"],
                })

            except Exception as e:
                print(f"ERROR: {str(e)[:100]}")
                results.append({
                    "test": "niah",
                    "context_length": ctx_len,
                    "depth": depth,
                    "passkey": passkey,
                    "answer": "ERROR",
                    "correct": False,
                    "elapsed": 0,
                    "output_tokens": 0,
                    "error": str(e)[:200],
                })

    return results


# ── Test 2: Tool-Calling Accuracy ──────────────────────────────────────────────

TOOL_DEFINITIONS = """You have access to the following tools:

<tools>
[
  {"name": "search_web", "description": "Search the web", "parameters": {"query": {"type": "string", "description": "Search query"}}},
  {"name": "read_file", "description": "Read a file", "parameters": {"path": {"type": "string", "description": "File path"}}},
  {"name": "write_file", "description": "Write to a file", "parameters": {"path": {"type": "string", "description": "File path"}, "content": {"type": "string", "description": "File content"}}},
  {"name": "run_command", "description": "Run a shell command", "parameters": {"command": {"type": "string", "description": "Command to execute"}}},
  {"name": "list_directory", "description": "List directory contents", "parameters": {"path": {"type": "string", "description": "Directory path"}}},
  {"name": "get_weather", "description": "Get weather information", "parameters": {"location": {"type": "string", "description": "Location name"}}},
  {"name": "calculate", "description": "Calculate math expression", "parameters": {"expression": {"type": "string", "description": "Math expression"}}}
]
</tools>

When you need to use a tool, you MUST respond with exactly this format:
<tool_call>
{"name": "tool_name", "arguments": {"param": "value"}}
</tool_call>

Always use a tool when the user asks you to perform an action. Do not explain — just output the tool call."""


TOOL_CALL_TEST_CASES = [
    {
        "instruction": "Search the web for 'OpenVINO 2026 release notes'.",
        "expected_tool": "search_web",
        "expected_arg_key": "query",
        "expected_arg_contains": "openvino",
    },
    {
        "instruction": "Read the file at /home/user/config.json.",
        "expected_tool": "read_file",
        "expected_arg_key": "path",
        "expected_arg_contains": "config.json",
    },
    {
        "instruction": "Run the command 'pip install numpy'.",
        "expected_tool": "run_command",
        "expected_arg_key": "command",
        "expected_arg_contains": "pip",
    },
    {
        "instruction": "List the contents of the /var/log directory.",
        "expected_tool": "list_directory",
        "expected_arg_key": "path",
        "expected_arg_contains": "/var/log",
    },
    {
        "instruction": "Calculate 2^32 - 1.",
        "expected_tool": "calculate",
        "expected_arg_key": "expression",
        "expected_arg_contains": "2",
    },
]


# ── Hard Tool-Call Test (more realistic agentic workflow) ──────────────────────

HARD_TOOL_DEFINITIONS = """You are an AI coding assistant with access to the following tools:

<tools>
[
  {"name": "search_web", "description": "Search the internet for information", "parameters": {"query": {"type": "string", "description": "The search query"}, "max_results": {"type": "integer", "description": "Maximum results to return", "default": 5}}},
  {"name": "read_file", "description": "Read the contents of a file from the filesystem", "parameters": {"path": {"type": "string", "description": "Absolute path to the file"}, "start_line": {"type": "integer", "description": "Starting line number (1-based)", "default": 1}, "end_line": {"type": "integer", "description": "Ending line number (inclusive)", "default": -1}}},
  {"name": "write_file", "description": "Write content to a file, creating it if necessary", "parameters": {"path": {"type": "string", "description": "Absolute path to the file"}, "content": {"type": "string", "description": "Content to write"}, "mode": {"type": "string", "description": "Write mode: 'overwrite' or 'append'", "default": "overwrite"}}},
  {"name": "run_command", "description": "Execute a shell command in the user's terminal", "parameters": {"command": {"type": "string", "description": "Shell command to execute"}, "cwd": {"type": "string", "description": "Working directory", "default": "."}, "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30}}},
  {"name": "list_directory", "description": "List files and directories in a path", "parameters": {"path": {"type": "string", "description": "Directory path to list"}, "recursive": {"type": "boolean", "description": "Whether to list recursively", "default": false}, "pattern": {"type": "string", "description": "Glob pattern to filter results", "default": "*"}}},
  {"name": "search_code", "description": "Search for a pattern in code files using regex", "parameters": {"pattern": {"type": "string", "description": "Regex pattern to search"}, "directory": {"type": "string", "description": "Directory to search in"}, "file_pattern": {"type": "string", "description": "File glob pattern", "default": "*.py"}}},
  {"name": "get_diagnostics", "description": "Get lint/compile errors for a file", "parameters": {"path": {"type": "string", "description": "File path to check"}, "severity": {"type": "string", "description": "Minimum severity: error, warning, info", "default": "warning"}}},
  {"name": "git_diff", "description": "Show git diff for staged or unstaged changes", "parameters": {"path": {"type": "string", "description": "File or directory path"}, "staged": {"type": "boolean", "description": "Show staged changes only", "default": false}}},
  {"name": "replace_in_file", "description": "Replace text in a file", "parameters": {"path": {"type": "string", "description": "File path"}, "old_text": {"type": "string", "description": "Text to find and replace"}, "new_text": {"type": "string", "description": "Replacement text"}}},
  {"name": "create_terminal", "description": "Create a new terminal session", "parameters": {"name": {"type": "string", "description": "Terminal session name"}, "cwd": {"type": "string", "description": "Working directory"}}},
  {"name": "get_symbol_info", "description": "Get type information and documentation for a code symbol", "parameters": {"symbol": {"type": "string", "description": "Symbol name to look up"}, "file": {"type": "string", "description": "File context for the lookup"}}},
  {"name": "apply_diff", "description": "Apply a unified diff patch to a file", "parameters": {"path": {"type": "string", "description": "File to patch"}, "diff": {"type": "string", "description": "Unified diff content"}}}
]
</tools>

When you need to use a tool, respond with exactly this format:
<tool_call>
{"name": "tool_name", "arguments": {"param": "value"}}
</tool_call>

Do not explain your reasoning. Just output the tool call."""


HARD_TOOL_CALL_CASES = [
    {
        "instruction": "I found a bug in /workspace/src/api/handlers.py. Read lines 45 through 80 of that file.",
        "expected_tool": "read_file",
        "expected_arg_key": "path",
        "expected_arg_contains": "handlers.py",
    },
    {
        "instruction": "Search the codebase in /workspace/src for any function that calls 'process_request'. Look in Python files only.",
        "expected_tool": "search_code",
        "expected_arg_key": "pattern",
        "expected_arg_contains": "process_request",
    },
    {
        "instruction": "Check if there are any lint errors in /workspace/src/models/user.py, show errors only.",
        "expected_tool": "get_diagnostics",
        "expected_arg_key": "path",
        "expected_arg_contains": "user.py",
    },
    {
        "instruction": "In the file /workspace/src/utils.py, replace the text 'import os' with 'import os\\nimport sys'.",
        "expected_tool": "replace_in_file",
        "expected_arg_key": "path",
        "expected_arg_contains": "utils.py",
    },
    {
        "instruction": "Run 'pytest tests/ -v --tb=short' in the /workspace directory with a 60 second timeout.",
        "expected_tool": "run_command",
        "expected_arg_key": "command",
        "expected_arg_contains": "pytest",
    },
]


# ── Generated conversation history for hard mode ───────────────────────────────

def _generate_agentic_history(target_tokens: int) -> list:
    """Generate realistic agentic conversation history (tool calls + results)."""
    history = []
    
    fake_interactions = [
        (
            "Let me check the project structure first.",
            '<tool_call>\n{"name": "list_directory", "arguments": {"path": "/workspace", "recursive": false}}\n</tool_call>',
            "tool_result: src/ tests/ README.md requirements.txt setup.py .gitignore Makefile docs/",
        ),
        (
            "I'll read the main configuration file to understand the project setup.",
            '<tool_call>\n{"name": "read_file", "arguments": {"path": "/workspace/setup.py"}}\n</tool_call>',
            'tool_result: from setuptools import setup, find_packages\n\nsetup(\n    name="myproject",\n    version="0.1.0",\n    packages=find_packages(),\n    install_requires=[\n        "flask>=2.0",\n        "sqlalchemy>=1.4",\n        "pydantic>=1.8",\n        "redis>=4.0",\n        "celery>=5.0",\n    ],\n)',
        ),
        (
            "Let me look at the source code structure.",
            '<tool_call>\n{"name": "list_directory", "arguments": {"path": "/workspace/src", "recursive": true}}\n</tool_call>',
            "tool_result: api/ api/__init__.py api/handlers.py api/middleware.py api/routes.py models/ models/__init__.py models/user.py models/product.py models/order.py services/ services/__init__.py services/auth.py services/payment.py utils.py config.py",
        ),
        (
            "Let me check the main API handler file.",
            '<tool_call>\n{"name": "read_file", "arguments": {"path": "/workspace/src/api/handlers.py"}}\n</tool_call>',
            'tool_result: from flask import jsonify, request\nfrom ..models.user import User\nfrom ..models.product import Product\nfrom ..services.auth import verify_token\n\ndef handle_login(request):\n    data = request.get_json()\n    user = User.authenticate(data["username"], data["password"])\n    if user:\n        token = generate_token(user.id)\n        return jsonify({"token": token})\n    return jsonify({"error": "Invalid credentials"}), 401\n\ndef handle_get_products(request):\n    page = request.args.get("page", 1, type=int)\n    products = Product.query.paginate(page=page, per_page=20)\n    return jsonify([p.to_dict() for p in products.items])\n\ndef process_request(request):\n    """Main request processing pipeline."""\n    if not verify_token(request.headers.get("Authorization")):\n        return jsonify({"error": "Unauthorized"}), 401\n    return handle_request_body(request)\n',
        ),
        (
            "Now let me check the user model.",
            '<tool_call>\n{"name": "read_file", "arguments": {"path": "/workspace/src/models/user.py"}}\n</tool_call>',
            'tool_result: from sqlalchemy import Column, Integer, String, DateTime\nfrom sqlalchemy.ext.declarative import declarative_base\nfrom werkzeug.security import generate_password_hash, check_password_hash\nimport datetime\n\nBase = declarative_base()\n\nclass User(Base):\n    __tablename__ = "users"\n    id = Column(Integer, primary_key=True)\n    username = Column(String(80), unique=True, nullable=False)\n    email = Column(String(120), unique=True, nullable=False)\n    password_hash = Column(String(128))\n    created_at = Column(DateTime, default=datetime.datetime.utcnow)\n\n    def set_password(self, password):\n        self.password_hash = generate_password_hash(password)\n\n    def check_password(self, password):\n        return check_password_hash(self.password_hash, password)\n\n    @classmethod\n    def authenticate(cls, username, password):\n        user = cls.query.filter_by(username=username).first()\n        if user and user.check_password(password):\n            return user\n        return None\n',
        ),
        (
            "Let me check the tests.",
            '<tool_call>\n{"name": "run_command", "arguments": {"command": "pytest tests/ -v --tb=short", "cwd": "/workspace"}}\n</tool_call>',
            'tool_result: ============================= test session starts =============================\ncollected 15 items\n\ntests/test_auth.py::test_login PASSED\ntests/test_auth.py::test_invalid_login PASSED\ntests/test_auth.py::test_token_verification PASSED\ntests/test_products.py::test_get_products PASSED\ntests/test_products.py::test_product_pagination PASSED\ntests/test_users.py::test_create_user PASSED\ntests/test_users.py::test_duplicate_username PASSED\ntests/test_users.py::test_password_hashing PASSED\n\n===================== 8 passed in 2.34s ======================',
        ),
    ]
    
    token_count = 0
    msg_idx = 0
    while token_count < target_tokens:
        interaction = fake_interactions[msg_idx % len(fake_interactions)]
        msg_idx += 1
        
        # User asked something → assistant made a tool call → user provided result
        history.append({"role": "user", "content": f"Step {msg_idx}: {interaction[0]}"})
        history.append({"role": "assistant", "content": interaction[1]})
        history.append({"role": "user", "content": interaction[2]})
        
        # Rough token estimate
        for msg in history[-3:]:
            token_count += len(msg["content"]) // 4
    
    return history


def _check_tool_call(text: str, expected_tool: str, expected_arg_key: str,
                     expected_arg_contains: str) -> dict:
    """Parse and validate a tool call from model output."""
    # Clean thinking tags
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    result = {
        "has_tool_call_tags": False,
        "valid_json": False,
        "correct_tool": False,
        "correct_arg_key": False,
        "correct_arg_value": False,
        "score": 0,
    }

    # Check for tool_call tags
    tc_match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", clean, re.DOTALL)
    if not tc_match:
        # Also try without closing tag
        tc_match = re.search(r"<tool_call>\s*(.*?)$", clean, re.DOTALL)
    if not tc_match:
        # GenAI pipeline may strip the opening <tool_call> tag — match JSON + </tool_call>
        tc_match = re.search(r"(\{.*?\})\s*</tool_call>", clean, re.DOTALL)
    if not tc_match:
        # Last resort: match any JSON object that looks like a tool call
        tc_match = re.search(r'(\{"name"\s*:.*?\})\s*$', clean, re.DOTALL)

    if tc_match:
        result["has_tool_call_tags"] = True
        result["score"] += 1

        try:
            payload = json.loads(tc_match.group(1).strip())
            result["valid_json"] = True
            result["score"] += 1

            tool_name = payload.get("name", "")
            if tool_name == expected_tool:
                result["correct_tool"] = True
                result["score"] += 1

            args = payload.get("arguments", {})
            if expected_arg_key in args:
                result["correct_arg_key"] = True
                result["score"] += 1

                if expected_arg_contains.lower() in str(args[expected_arg_key]).lower():
                    result["correct_arg_value"] = True
                    result["score"] += 1

        except (json.JSONDecodeError, AttributeError):
            pass

    return result


def test_tool_calling(backend, context_lengths: list, max_output_tokens: int = 256) -> list:
    """
    Test tool-calling accuracy at increasing context lengths.
    Adds padding conversation history to increase context size.
    """
    print(f"\n{'='*80}")
    print(f"  TEST: Tool-Calling Accuracy")
    print(f"  Context lengths: {context_lengths}")
    print(f"  Test cases: {len(TOOL_CALL_TEST_CASES)}")
    print(f"{'='*80}\n")

    results = []

    for ctx_len in context_lengths:
        for tc in TOOL_CALL_TEST_CASES:
            # Build messages with padding conversation history
            system_content = TOOL_DEFINITIONS

            # Calculate padding needed
            # system prompt ≈ 800 tokens, each user/assistant turn ≈ 100 tokens
            system_tokens = 800
            question_tokens = 50
            remaining = ctx_len - system_tokens - question_tokens

            # Generate filler as fake conversation history
            history_messages = []
            history_tokens = 0
            turn = 0
            while history_tokens < remaining:
                filler_q = f"Tell me about topic number {turn + 1} in software engineering. Explain the key concepts and best practices involved."
                filler_a = generate_filler(min(200, remaining - history_tokens))

                history_messages.append({"role": "user", "content": filler_q})
                history_messages.append({"role": "assistant", "content": filler_a})
                history_tokens += len(filler_q) // 4 + len(filler_a) // 4
                turn += 1

            messages = [
                {"role": "system", "content": system_content},
                *history_messages,
                {"role": "user", "content": tc["instruction"] + " /no_think"},
            ]

            print(f"  ctx={ctx_len:>6}, tool={tc['expected_tool']:<16} ... ", end="", flush=True)

            try:
                result = backend.chat(messages, max_tokens=max_output_tokens, temperature=0.0)
                answer = result["text"]

                check = _check_tool_call(
                    answer, tc["expected_tool"],
                    tc["expected_arg_key"], tc["expected_arg_contains"],
                )

                status = f"score={check['score']}/5"
                if check["score"] == 5:
                    status = "PERFECT"
                elif check["score"] >= 3:
                    status = f"PARTIAL ({check['score']}/5)"
                else:
                    status = f"FAIL ({check['score']}/5)"

                print(f"{status} (elapsed: {result['elapsed']:.1f}s)")

                results.append({
                    "test": "tool_call",
                    "context_length": ctx_len,
                    "expected_tool": tc["expected_tool"],
                    "instruction": tc["instruction"],
                    "has_tool_call_tags": check["has_tool_call_tags"],
                    "valid_json": check["valid_json"],
                    "correct_tool": check["correct_tool"],
                    "correct_arg_key": check["correct_arg_key"],
                    "correct_arg_value": check["correct_arg_value"],
                    "score": check["score"],
                    "answer_preview": answer[:200],
                    "elapsed": round(result["elapsed"], 2),
                    "output_tokens": result["tokens"],
                })

            except Exception as e:
                print(f"ERROR: {str(e)[:100]}")
                results.append({
                    "test": "tool_call",
                    "context_length": ctx_len,
                    "expected_tool": tc["expected_tool"],
                    "instruction": tc["instruction"],
                    "score": 0,
                    "error": str(e)[:200],
                    "elapsed": 0,
                })

    return results


def test_tool_calling_hard(backend, context_lengths: list, max_output_tokens: int = 256) -> list:
    """
    Hard tool-calling test with realistic agentic conversation history.
    Uses 12 tool definitions and multi-turn tool call/result history.
    """
    print(f"\n{'='*80}")
    print(f"  TEST: Hard Tool-Calling (Agentic Workflow)")
    print(f"  Context lengths: {context_lengths}")
    print(f"  Test cases: {len(HARD_TOOL_CALL_CASES)}")
    print(f"{'='*80}\n")

    results = []

    for ctx_len in context_lengths:
        for tc in HARD_TOOL_CALL_CASES:
            # Build messages with realistic agentic history
            system_content = HARD_TOOL_DEFINITIONS

            # System prompt ≈ 1500 tokens for hard mode, question ≈ 50
            system_tokens = 1500
            question_tokens = 50
            remaining = max(0, ctx_len - system_tokens - question_tokens)

            # Generate agentic conversation history
            history = _generate_agentic_history(remaining)

            messages = [
                {"role": "system", "content": system_content},
                *history,
                {"role": "user", "content": tc["instruction"] + " /no_think"},
            ]

            print(f"  ctx={ctx_len:>6}, tool={tc['expected_tool']:<16} ... ", end="", flush=True)

            try:
                result = backend.chat(messages, max_tokens=max_output_tokens, temperature=0.0)
                answer = result["text"]

                check = _check_tool_call(
                    answer, tc["expected_tool"],
                    tc["expected_arg_key"], tc["expected_arg_contains"],
                )

                status = f"score={check['score']}/5"
                if check["score"] == 5:
                    status = "PERFECT"
                elif check["score"] >= 3:
                    status = f"PARTIAL ({check['score']}/5)"
                else:
                    status = f"FAIL ({check['score']}/5)"

                print(f"{status} (elapsed: {result['elapsed']:.1f}s)")

                results.append({
                    "test": "tool_call_hard",
                    "context_length": ctx_len,
                    "expected_tool": tc["expected_tool"],
                    "instruction": tc["instruction"],
                    "has_tool_call_tags": check["has_tool_call_tags"],
                    "valid_json": check["valid_json"],
                    "correct_tool": check["correct_tool"],
                    "correct_arg_key": check["correct_arg_key"],
                    "correct_arg_value": check["correct_arg_value"],
                    "score": check["score"],
                    "answer_preview": answer[:200],
                    "elapsed": round(result["elapsed"], 2),
                    "output_tokens": result["tokens"],
                })

            except Exception as e:
                print(f"ERROR: {str(e)[:100]}")
                results.append({
                    "test": "tool_call_hard",
                    "context_length": ctx_len,
                    "expected_tool": tc["expected_tool"],
                    "instruction": tc["instruction"],
                    "score": 0,
                    "error": str(e)[:200],
                    "elapsed": 0,
                })

    return results


# ── Results Summary ────────────────────────────────────────────────────────────

def print_niah_summary(results: list):
    """Print NIAH results as a depth × context_length matrix."""
    if not results:
        return

    context_lengths = sorted(set(r["context_length"] for r in results))
    depths = sorted(set(r["depth"] for r in results))

    print(f"\n{'='*80}")
    print(f"  NIAH Results: Passkey Retrieval Accuracy")
    print(f"{'='*80}")

    # Header
    header = f"{'Depth':>8}"
    for cl in context_lengths:
        header += f"  {cl//1024:>4}K"
    print(header)
    print("-" * len(header))

    for depth in depths:
        row = f"{depth:>8.2f}"
        for cl in context_lengths:
            matching = [r for r in results if r["context_length"] == cl and r["depth"] == depth]
            if matching:
                r = matching[0]
                mark = " PASS" if r["correct"] else " FAIL"
                row += f"  {mark}"
            else:
                row += f"    --"
        print(row)

    # Overall accuracy per context length
    print()
    row = f"{'Acc%':>8}"
    for cl in context_lengths:
        matching = [r for r in results if r["context_length"] == cl]
        if matching:
            acc = sum(1 for r in matching if r["correct"]) / len(matching) * 100
            row += f"  {acc:>4.0f}%"
        else:
            row += f"    --"
    print(row)
    print()


def print_tool_call_summary(results: list):
    """Print tool-calling results summary."""
    if not results:
        return

    context_lengths = sorted(set(r["context_length"] for r in results))

    print(f"\n{'='*80}")
    print(f"  Tool-Calling Results: Accuracy by Context Length")
    print(f"{'='*80}")

    print(f"\n{'Context':>8}  {'Avg Score':>10}  {'Perfect':>8}  {'Has Tags':>9}  {'Valid JSON':>11}  {'Right Tool':>11}")
    print("-" * 75)

    for cl in context_lengths:
        matching = [r for r in results if r["context_length"] == cl and "score" in r]
        if not matching:
            continue
        n = len(matching)
        avg_score = sum(r["score"] for r in matching) / n
        perfect = sum(1 for r in matching if r["score"] == 5)
        has_tags = sum(1 for r in matching if r.get("has_tool_call_tags"))
        valid_json = sum(1 for r in matching if r.get("valid_json"))
        right_tool = sum(1 for r in matching if r.get("correct_tool"))

        print(f"{cl//1024:>6}K  {avg_score:>10.1f}/5  {perfect:>6}/{n}  {has_tags:>7}/{n}  {valid_json:>9}/{n}  {right_tool:>9}/{n}")

    print()


def save_results(results: list, output_path: str):
    """Save results to JSON and CSV."""
    # JSON
    json_path = output_path + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Results saved to {json_path}")

    # CSV
    csv_path = output_path + ".csv"
    if results:
        keys = list(results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"  Results saved to {csv_path}")


# ── Compare Mode ───────────────────────────────────────────────────────────────

def run_comparison(model_dirs: list, device: str, tests: list,
                   context_lengths: list, depths: list, max_tokens: int):
    """Compare multiple models on the same test suite."""
    all_model_results = {}

    for model_dir in model_dirs:
        model_name = Path(model_dir).name
        print(f"\n\n{'#'*80}")
        print(f"  MODEL: {model_name}")
        print(f"  Path:  {model_dir}")
        print(f"{'#'*80}")

        backend = GenAIBackend(model_dir, device)
        results = []

        if "niah" in tests:
            results.extend(test_niah(backend, context_lengths, depths, max_output_tokens=max_tokens))
        if "tool_call" in tests:
            results.extend(test_tool_calling(backend, context_lengths, max_output_tokens=max_tokens))
        if "tool_call_hard" in tests:
            results.extend(test_tool_calling_hard(backend, context_lengths, max_output_tokens=max_tokens))

        all_model_results[model_name] = results

        # Free model
        del backend
        import gc
        gc.collect()
        time.sleep(5)  # Let GPU memory settle

    # Print comparison
    print(f"\n\n{'='*80}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*80}")

    for model_name, results in all_model_results.items():
        print(f"\n  --- {model_name} ---")
        niah = [r for r in results if r["test"] == "niah"]
        tc = [r for r in results if r["test"] == "tool_call"]
        tc_hard = [r for r in results if r["test"] == "tool_call_hard"]

        if niah:
            print_niah_summary(niah)
        if tc:
            print_tool_call_summary(tc)
        if tc_hard:
            print(f"\n  [Hard Tool-Call]")
            print_tool_call_summary(tc_hard)

        # Save individual results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_results(results, f"benchmark_{model_name}_{timestamp}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Long-Context Benchmark for Qwen3 Coder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--backend", choices=["genai", "ovms"], default="genai",
                        help="Inference backend (default: genai)")
    parser.add_argument("--model-dir", type=str,
                        default=r"C:\working\models\Qwen3-Coder-30B-A3B-Instruct\INT4",
                        help="Model directory for GenAI backend")
    parser.add_argument("--device", type=str, default="GPU",
                        help="Device for GenAI backend (default: GPU)")
    parser.add_argument("--ovms-url", type=str, default="http://localhost:8000",
                        help="OVMS URL for OVMS backend")
    parser.add_argument("--ovms-model", type=str,
                        default="Qwen3-Coder-30B-A3B-Instruct",
                        help="Model name for OVMS backend")
    parser.add_argument("--tests", nargs="+", default=["niah", "tool_call"],
                        choices=["niah", "tool_call", "tool_call_hard"],
                        help="Tests to run (default: niah + tool_call)")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Max output tokens for test responses (default: 128)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer context lengths and depths")
    parser.add_argument("--compare", nargs="+", metavar="MODEL_DIR",
                        help="Compare multiple model directories")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file prefix (default: auto-generated)")

    args = parser.parse_args()

    context_lengths = QUICK_LENGTHS if args.quick else CONTEXT_LENGTHS
    depths = QUICK_DEPTHS if args.quick else NEEDLE_DEPTHS

    # ── Compare mode ──
    if args.compare:
        run_comparison(args.compare, args.device, args.tests,
                       context_lengths, depths, args.max_tokens)
        return

    # ── Single model mode ──
    if args.backend == "genai":
        backend = GenAIBackend(args.model_dir, args.device)
    else:
        backend = OVMSBackend(args.ovms_url, args.ovms_model)

    all_results = []

    if "niah" in args.tests:
        niah_results = test_niah(backend, context_lengths, depths,
                                 max_output_tokens=args.max_tokens)
        all_results.extend(niah_results)
        print_niah_summary(niah_results)

    if "tool_call" in args.tests:
        tc_results = test_tool_calling(backend, context_lengths,
                                       max_output_tokens=args.max_tokens)
        all_results.extend(tc_results)
        print_tool_call_summary(tc_results)

    if "tool_call_hard" in args.tests:
        tc_hard_results = test_tool_calling_hard(backend, context_lengths,
                                                  max_output_tokens=args.max_tokens)
        all_results.extend(tc_hard_results)
        print_tool_call_summary(tc_hard_results)

    # Save results
    if all_results:
        model_name = Path(args.model_dir).name if args.backend == "genai" else "ovms"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_prefix = args.output or f"benchmark_{model_name}_{timestamp}"
        save_results(all_results, output_prefix)


if __name__ == "__main__":
    main()
