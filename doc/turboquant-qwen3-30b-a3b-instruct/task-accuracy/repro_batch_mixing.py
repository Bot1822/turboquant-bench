#!/usr/bin/env python3
"""Minimal reproduction for batch mixing in tq_no_alloc mode."""

from __future__ import annotations

import json
import os
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from turboquant.vllm_attn_backend import enable_no_alloc


MODEL = os.environ.get("MODEL", "/share/models/official/Qwen3-30B-A3B-Instruct-2507")
TP = int(os.environ.get("TP", "2"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))
GPU_MEM = float(os.environ.get("GPU_MEM", "0.85"))
MODE = os.environ.get("MODE", "baseline")
TARGET_TOKENS = int(os.environ.get("TARGET_TOKENS", "2048"))


def build_prompt(tokenizer, answer: str) -> tuple[str, int]:
    filler_text = Path("/ceph/User/E01442/turboquant/vllm/benchmarks/sonnet.txt").read_text()
    filler_tokens = tokenizer.encode(filler_text, add_special_tokens=False)
    prefix = (
        "Read the long document and answer with the exact retrieval key only.\n"
    )
    anchor = f"\n\nImportant retrieval record: the exact retrieval key is {answer}.\n\n"
    prefix_tokens = tokenizer.encode(prefix + anchor, add_special_tokens=False)
    filler_budget = max(TARGET_TOKENS - len(prefix_tokens), 256)
    repeated = []
    while len(repeated) < filler_budget:
        repeated.extend(filler_tokens)
    filler = tokenizer.decode(repeated[:filler_budget])
    prompt = prefix + filler + anchor + "What is the exact retrieval key?"
    return prompt, len(tokenizer.encode(prompt, add_special_tokens=False))


def run(prompts):
    if MODE == "tq_no_alloc":
        enable_no_alloc()

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        tensor_parallel_size=TP,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM,
        max_num_seqs=max(len(prompts), 1),
        trust_remote_code=True,
    )
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0, max_tokens=16),
    )
    return [out.outputs[0].text.strip().splitlines()[0].strip() for out in outputs]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompt_a, toks_a = build_prompt(tokenizer, "alpha-key-777")
    prompt_b, toks_b = build_prompt(tokenizer, "beta-key-999")

    single_a = run([prompt_a])[0]
    single_b = run([prompt_b])[0]
    batch = run([prompt_a, prompt_b])

    print(json.dumps({
        "mode": MODE,
        "prompt_tokens": [toks_a, toks_b],
        "single_a": single_a,
        "single_b": single_b,
        "batch_a": batch[0],
        "batch_b": batch[1],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
