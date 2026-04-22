#!/usr/bin/env python3
"""Dump the actual prompts sent to the model during benchmark tests."""
import sys, json, os
sys.path.insert(0, os.path.dirname(__file__))

from benchmark_context import (
    HARD_TOOL_DEFINITIONS, HARD_TOOL_CALL_CASES,
    _generate_agentic_history, generate_filler,
    TOOL_DEFINITIONS, TOOL_CALL_TEST_CASES
)

out_dir = os.path.join(os.path.dirname(__file__), '..', 'prompt_samples')
os.makedirs(out_dir, exist_ok=True)

# ── Hard tool-call prompts (the ones that fail) ──
print("=" * 80)
print("  HARD TOOL-CALL PROMPTS (12 tools, agentic history)")
print("=" * 80)

for tc_idx, tc in enumerate(HARD_TOOL_CALL_CASES):
    tool_name = tc['expected_tool']
    instruction = tc['instruction']
    print(f"\nTest case {tc_idx}: {tool_name}")
    print(f"  Instruction: {instruction}")

    for ctx_len in [32768, 49152, 65536]:
        system_tokens = 1500
        question_tokens = 50
        remaining = max(0, ctx_len - system_tokens - question_tokens)

        history = _generate_agentic_history(remaining)

        messages = [
            {"role": "system", "content": HARD_TOOL_DEFINITIONS},
            *history,
            {"role": "user", "content": instruction + " /no_think"},
        ]

        n_msgs = len(messages)
        total_chars = sum(len(m["content"]) for m in messages)
        est_tokens = total_chars // 4

        # Save full prompt
        fname = os.path.join(out_dir, f"hard_{tool_name}_{ctx_len // 1024}K.txt")
        with open(fname, "w", encoding="utf-8") as f:
            for i, m in enumerate(messages):
                f.write(f"=== [{m['role']}] (msg {i+1}/{n_msgs}) ===\n")
                f.write(m["content"])
                f.write("\n\n")

        fsize = os.path.getsize(fname)
        print(f"  {ctx_len // 1024}K: {n_msgs} msgs, ~{est_tokens} est tokens, {fsize // 1024}KB")

# ── Standard tool-call prompts (the ones that pass) ──
print("\n" + "=" * 80)
print("  STANDARD TOOL-CALL PROMPTS (7 tools, filler padding)")
print("=" * 80)

tc = TOOL_CALL_TEST_CASES[0]  # search_web
tool_name = tc['expected_tool']
instruction = tc['instruction']
print(f"\nTest case: {tool_name}")
print(f"  Instruction: {instruction}")

for ctx_len in [32768, 49152, 65536]:
    system_content = TOOL_DEFINITIONS
    system_tokens = 800
    question_tokens = 50
    remaining = ctx_len - system_tokens - question_tokens

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
        {"role": "user", "content": instruction + " /no_think"},
    ]

    n_msgs = len(messages)
    total_chars = sum(len(m["content"]) for m in messages)
    est_tokens = total_chars // 4

    fname = os.path.join(out_dir, f"standard_{tool_name}_{ctx_len // 1024}K.txt")
    with open(fname, "w", encoding="utf-8") as f:
        for i, m in enumerate(messages):
            f.write(f"=== [{m['role']}] (msg {i+1}/{n_msgs}) ===\n")
            f.write(m["content"])
            f.write("\n\n")

    fsize = os.path.getsize(fname)
    print(f"  {ctx_len // 1024}K: {n_msgs} msgs, ~{est_tokens} est tokens, {fsize // 1024}KB")

# ── Show structure of 32K hard read_file (the main failure) ──
print("\n" + "=" * 80)
print("  STRUCTURE: hard_read_file_32K prompt")
print("=" * 80)

remaining = max(0, 32768 - 1500 - 50)
history = _generate_agentic_history(remaining)
messages = [
    {"role": "system", "content": HARD_TOOL_DEFINITIONS},
    *history,
    {"role": "user", "content": HARD_TOOL_CALL_CASES[0]["instruction"] + " /no_think"},
]

print(f"\nTotal messages: {len(messages)}")
print(f"\nmsg[0] SYSTEM ({len(messages[0]['content'])} chars):")
print(messages[0]["content"][:500] + "\n  ... [truncated] ...")
print(f"\nmsg[1] ({messages[1]['role']}, {len(messages[1]['content'])} chars):")
print(messages[1]["content"][:200])
print(f"\nmsg[2] ({messages[2]['role']}, {len(messages[2]['content'])} chars):")
print(messages[2]["content"][:200])
print(f"\nmsg[3] ({messages[3]['role']}, {len(messages[3]['content'])} chars):")
print(messages[3]["content"][:200])

print(f"\n  ... {len(messages) - 6} more messages (repeating agentic history) ...")

print(f"\nmsg[{len(messages)-3}] ({messages[-3]['role']}, {len(messages[-3]['content'])} chars):")
print(messages[-3]["content"][:200])
print(f"\nmsg[{len(messages)-2}] ({messages[-2]['role']}, {len(messages[-2]['content'])} chars):")
print(messages[-2]["content"][:200])
print(f"\nmsg[{len(messages)-1}] ({messages[-1]['role']}, {len(messages[-1]['content'])} chars):")
print(messages[-1]["content"])

print("\nDone! All prompts saved to prompt_samples/")
