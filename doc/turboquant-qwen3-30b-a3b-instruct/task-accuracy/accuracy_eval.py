#!/usr/bin/env python3
"""Deterministic long-context retrieval accuracy benchmark."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from turboquant.accuracy_utils import exact_match
from turboquant.vllm_attn_backend import enable_no_alloc


MODEL = os.environ.get("MODEL", "/share/models/official/Qwen3-30B-A3B-Instruct-2507")
TP = int(os.environ.get("TP", "2"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))
GPU_MEM = float(os.environ.get("GPU_MEM", "0.85"))
MODE = os.environ.get("MODE", "baseline")
NUM_EXAMPLES = int(os.environ.get("NUM_EXAMPLES", "8"))
TARGET_TOKENS = int(os.environ.get("TARGET_TOKENS", "2048"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "16"))


def build_examples(tokenizer):
    filler_text = Path("/ceph/User/E01442/turboquant/vllm/benchmarks/sonnet.txt").read_text()
    filler_tokens = tokenizer.encode(filler_text, add_special_tokens=False)
    examples = []

    for idx in range(NUM_EXAMPLES):
        answer = f"key-{idx:02d}-alpha"
        question = (
            "Read the long document and answer with the exact retrieval key only. "
            "Do not explain.\n"
        )
        anchor = (
            f"\n\nImportant retrieval record {idx}: "
            f"the exact retrieval key is {answer}.\n\n"
        )
        question_tokens = tokenizer.encode(question, add_special_tokens=False)
        anchor_tokens = tokenizer.encode(anchor, add_special_tokens=False)
        filler_budget = max(TARGET_TOKENS - len(question_tokens) - len(anchor_tokens), 128)
        repeated = []
        while len(repeated) < filler_budget:
            repeated.extend(filler_tokens)
        filler = tokenizer.decode(repeated[:filler_budget])
        prompt = question + filler + anchor + "What is the exact retrieval key?"
        examples.append({
            "id": idx,
            "prompt": prompt,
            "answer": answer,
            "prompt_tokens": len(tokenizer.encode(prompt, add_special_tokens=False)),
        })
    return examples


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    examples = build_examples(tokenizer)

    if MODE == "tq_no_alloc":
        enable_no_alloc()

    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        tensor_parallel_size=TP,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM,
        max_num_seqs=1,
        trust_remote_code=True,
    )
    load_s = time.perf_counter() - t0

    preds = []
    for example in examples:
        out = llm.generate(
            [example["prompt"]],
            SamplingParams(temperature=0, max_tokens=MAX_TOKENS),
        )
        pred = out[0].outputs[0].text.strip().splitlines()[0].strip()
        preds.append({
            "id": example["id"],
            "prompt_tokens": example["prompt_tokens"],
            "answer": example["answer"],
            "prediction": pred,
            "correct": exact_match(pred, example["answer"]),
        })

    correct = sum(item["correct"] for item in preds)
    print(json.dumps({
        "mode": MODE,
        "model": MODEL,
        "tp": TP,
        "target_tokens": TARGET_TOKENS,
        "num_examples": NUM_EXAMPLES,
        "max_tokens": MAX_TOKENS,
        "load_s": round(load_s, 3),
        "accuracy": round(correct / max(NUM_EXAMPLES, 1), 4),
        "correct": correct,
        "results": preds,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
