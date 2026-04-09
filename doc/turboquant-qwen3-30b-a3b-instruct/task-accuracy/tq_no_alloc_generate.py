#!/usr/bin/env python3
"""Smoke test for compressed-decode generation path."""

from __future__ import annotations

import json
import os
import subprocess
import time

from vllm import LLM, SamplingParams

from turboquant.vllm_attn_backend import enable_no_alloc


MODEL = os.environ.get("MODEL", "/share/models/official/Qwen3-30B-A3B-Instruct-2507")
TP = int(os.environ.get("TP", "2"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))
GPU_MEM = float(os.environ.get("GPU_MEM", "0.85"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "64"))
PROMPT = os.environ.get(
    "PROMPT",
    "Explain KV cache compression in one concise paragraph.",
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

    t1 = time.perf_counter()
    out = llm.generate(
        [PROMPT],
        SamplingParams(temperature=0, max_tokens=MAX_TOKENS),
    )
    gen_s = time.perf_counter() - t1

    print(json.dumps({
        "load_s": round(load_s, 3),
        "gen_s": round(gen_s, 3),
        "mem_after": gpu_mem(),
        "text": out[0].outputs[0].text[:400],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
