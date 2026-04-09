#!/usr/bin/env python3
"""Load an engine, run one wave, optionally free KV cache, then hold process."""

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
INPUT_TOKENS = int(os.environ.get("INPUT_TOKENS", "4096"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "1"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "64"))
MODE = os.environ.get("MODE", "baseline")
SLEEP_SECS = int(os.environ.get("SLEEP_SECS", "120"))


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

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        tensor_parallel_size=TP,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM,
        max_num_seqs=max(CONCURRENCY, 1),
        trust_remote_code=True,
    )

    engine = llm.llm_engine
    core = getattr(engine, "engine_core", engine)
    inner = getattr(core, "engine_core", core)
    executor = inner.model_executor

    hook_count = 0
    if MODE == "tq":
        def _install(worker):
            from turboquant.vllm_attn_backend import install_turboquant_hooks, MODE_ACTIVE
            return len(install_turboquant_hooks(
                worker.model_runner,
                key_bits=3,
                value_bits=2,
                buffer_size=128,
                mode=MODE_ACTIVE,
            ))
        hook_count = executor.collective_rpc(_install)[0]

    outputs = llm.generate(
        prompt_texts,
        SamplingParams(temperature=0, max_tokens=MAX_TOKENS),
    )

    freed = None
    if MODE == "tq":
        def _free(worker):
            from turboquant.vllm_attn_backend import free_kv_cache
            return free_kv_cache(worker.model_runner)
        freed = executor.collective_rpc(_free)

    print(json.dumps({
        "ready": True,
        "mode": MODE,
        "hook_count": hook_count,
        "freed": freed,
        "mem_after_wave1": gpu_mem(),
        "sample_text": outputs[0].outputs[0].text[:200] if outputs else "",
    }, ensure_ascii=False), flush=True)

    time.sleep(SLEEP_SECS)


if __name__ == "__main__":
    main()
