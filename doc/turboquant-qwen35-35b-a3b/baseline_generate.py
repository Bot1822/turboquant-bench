#!/usr/bin/env python3
"""Minimal baseline vLLM generation script for Qwen3.5-35B-A3B."""

from __future__ import annotations

import json
import os
import subprocess
import time

from vllm import LLM, SamplingParams


MODEL = os.environ.get("MODEL", "/share/models/official/Qwen3.5-35B-A3B")
TP = int(os.environ.get("TP", "2"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))
GPU_MEM = float(os.environ.get("GPU_MEM", "0.85"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "64"))
ENFORCE_EAGER = os.environ.get("ENFORCE_EAGER", "0") == "1"
PROMPT = os.environ.get(
    "PROMPT",
    "Briefly explain what KV cache compression is and why it matters for "
    "long-context inference.",
)


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
    mem_before = gpu_mem()

    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        tensor_parallel_size=TP,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM,
        max_num_seqs=1,
        enforce_eager=ENFORCE_EAGER,
        trust_remote_code=True,
    )
    load_s = time.perf_counter() - t0
    blocks = llm.llm_engine.vllm_config.cache_config.num_gpu_blocks
    mem_loaded = gpu_mem()

    sampling_params = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
    t1 = time.perf_counter()
    output = llm.generate([PROMPT], sampling_params)
    gen_s = time.perf_counter() - t1
    mem_after = gpu_mem()

    text = output[0].outputs[0].text
    token_ids = output[0].outputs[0].token_ids
    result = {
        "model": MODEL,
        "tp": TP,
        "max_model_len": MAX_MODEL_LEN,
        "gpu_mem_utilization": GPU_MEM,
        "enforce_eager": ENFORCE_EAGER,
        "load_s": round(load_s, 3),
        "gen_s": round(gen_s, 3),
        "num_tokens": len(token_ids),
        "tok_per_s": round(len(token_ids) / gen_s, 3) if gen_s else None,
        "blocks": blocks,
        "mem_before": mem_before,
        "mem_loaded": mem_loaded,
        "mem_after": mem_after,
        "text": text[:400],
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
