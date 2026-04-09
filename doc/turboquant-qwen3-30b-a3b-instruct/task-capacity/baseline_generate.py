#!/usr/bin/env python3
"""Minimal baseline generation driver for Qwen3-30B-A3B-Instruct."""

from __future__ import annotations

import json
import os
import subprocess
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from turboquant.dialog_workload import build_batched_dialog_prompts


MODEL = os.environ.get("MODEL", "/share/models/official/Qwen3-30B-A3B-Instruct-2507")
TP = int(os.environ.get("TP", "2"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))
GPU_MEM = float(os.environ.get("GPU_MEM", "0.85"))
INPUT_TOKENS = int(os.environ.get("INPUT_TOKENS", "2048"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "1"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "64"))
ENFORCE_EAGER = os.environ.get("ENFORCE_EAGER", "0") == "1"


def gpu_mem():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [
        int(line.split(",")[1].strip())
        for line in result.stdout.strip().splitlines()
        if line.strip()
    ]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompts = build_batched_dialog_prompts(
        tokenizer=tokenizer,
        num_prompts=CONCURRENCY,
        target_tokens=INPUT_TOKENS,
    )
    prompt_texts = [item["prompt"] for item in prompts]
    prompt_token_counts = [item["token_count"] for item in prompts]

    mem_before = gpu_mem()
    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        tensor_parallel_size=TP,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM,
        max_num_seqs=max(CONCURRENCY, 1),
        enforce_eager=ENFORCE_EAGER,
        trust_remote_code=True,
    )
    load_s = time.perf_counter() - t0
    blocks = llm.llm_engine.vllm_config.cache_config.num_gpu_blocks
    block_size = llm.llm_engine.vllm_config.cache_config.block_size
    mem_loaded = gpu_mem()

    sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
    t1 = time.perf_counter()
    outputs = llm.generate(prompt_texts, sp)
    gen_s = time.perf_counter() - t1
    mem_after = gpu_mem()
    total_out_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)

    print(json.dumps({
        "model": MODEL,
        "tp": TP,
        "max_model_len": MAX_MODEL_LEN,
        "gpu_mem_utilization": GPU_MEM,
        "input_tokens_target": INPUT_TOKENS,
        "actual_prompt_token_counts": prompt_token_counts,
        "concurrency": CONCURRENCY,
        "enforce_eager": ENFORCE_EAGER,
        "load_s": round(load_s, 3),
        "gen_s": round(gen_s, 3),
        "total_out_tokens": total_out_tokens,
        "tok_per_s": round(total_out_tokens / gen_s, 3) if gen_s else None,
        "blocks": blocks,
        "block_size": block_size,
        "mem_before": mem_before,
        "mem_loaded": mem_loaded,
        "mem_after": mem_after,
        "sample_text": outputs[0].outputs[0].text[:300] if outputs else "",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
