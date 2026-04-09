#!/usr/bin/env python3
"""Structural KV occupancy inspection for TurboQuant on Qwen3-30B-A3B."""

from __future__ import annotations

import json
import os

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from turboquant.dialog_workload import build_batched_dialog_prompts
from turboquant.experiment_utils import (
    estimate_active_kv_bytes,
    summarize_structural_compression,
    unique_storage_bytes,
)


MODEL = os.environ.get("MODEL", "/share/models/official/Qwen3-30B-A3B-Instruct-2507")
TP = int(os.environ.get("TP", "2"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "32768"))
GPU_MEM = float(os.environ.get("GPU_MEM", "0.9"))
INPUT_TOKENS = int(os.environ.get("INPUT_TOKENS", "8192"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "1"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "32"))
MODE = os.environ.get("MODE", "accumulate")
ENFORCE_EAGER = os.environ.get("ENFORCE_EAGER", "0") == "1"


def inspect_worker(worker):
    static_ctx = worker.model_runner.compilation_config.static_forward_context
    tq_states = getattr(worker.model_runner, "_tq_states", {})

    layer_entries = []
    tq_cache_entries = []

    for layer_name, state in tq_states.items():
        attn_module = static_ctx.get(layer_name)
        kv_list = getattr(attn_module, "kv_cache", None)
        if kv_list and len(kv_list) > 0:
            tensor = kv_list[0]
            storage = tensor.untyped_storage()
            layer_entries.append({
                "layer_name": layer_name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "storage_ptr": int(storage.data_ptr()),
                "storage_nbytes": int(storage.nbytes()),
                "numel_bytes": int(tensor.nelement() * tensor.element_size()),
            })

        for seq_id, cache in state.seq_caches.items():
            memory = cache.memory_bytes()
            tq_cache_entries.append({
                "layer_name": layer_name,
                "seq_id": seq_id,
                "seq_len": int(cache.seq_len),
                "num_kv_heads": int(state.num_kv_heads),
                "head_dim": int(state.head_dim),
                **memory,
            })

    allocated = int(torch.cuda.memory_allocated())
    reserved = int(torch.cuda.memory_reserved())
    active_baseline_bytes = 0
    if tq_cache_entries:
        sample = tq_cache_entries[0]
        active_baseline_bytes = estimate_active_kv_bytes(
            seq_len=sample["seq_len"],
            layer_count=len(tq_cache_entries),
            num_kv_heads_per_rank=sample["num_kv_heads"],
            head_dim=sample["head_dim"],
        )

    return {
        "allocated": allocated,
        "reserved": reserved,
        "hook_count": len(tq_states),
        "layer_entries": layer_entries,
        "tq_cache_entries": tq_cache_entries,
        "hooked_kv_storage_bytes": unique_storage_bytes(layer_entries),
        "hooked_kv_numel_bytes": sum(entry["numel_bytes"] for entry in layer_entries),
        "active_baseline_kv_bytes": active_baseline_bytes,
        "tq_quantized_keys_bytes": sum(entry["quantized_keys"] for entry in tq_cache_entries),
        "tq_quantized_values_bytes": sum(entry["quantized_values"] for entry in tq_cache_entries),
        "tq_buffer_bytes": sum(entry["buffer"] for entry in tq_cache_entries),
        "tq_total_bytes": sum(entry["total"] for entry in tq_cache_entries),
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompts = build_batched_dialog_prompts(
        tokenizer=tokenizer,
        num_prompts=CONCURRENCY,
        target_tokens=INPUT_TOKENS,
    )
    prompt_texts = [item["prompt"] for item in prompts]
    prompt_token_counts = [item["token_count"] for item in prompts]

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
    engine = llm.llm_engine
    core = getattr(engine, "engine_core", engine)
    inner = getattr(core, "engine_core", core)
    executor = inner.model_executor

    def _install(worker):
        from turboquant.vllm_attn_backend import (
            MODE_ACCUMULATE,
            MODE_ACTIVE,
            MODE_SHADOW,
            install_turboquant_hooks,
        )

        mode_map = {
            "accumulate": MODE_ACCUMULATE,
            "active": MODE_ACTIVE,
            "shadow": MODE_SHADOW,
        }
        return len(
            install_turboquant_hooks(
                worker.model_runner,
                key_bits=3,
                value_bits=2,
                buffer_size=128,
                mode=mode_map[MODE],
            )
        )

    hooks = executor.collective_rpc(_install)

    llm.generate(
        prompt_texts,
        SamplingParams(temperature=0, max_tokens=MAX_TOKENS),
    )

    before_free = executor.collective_rpc(inspect_worker)

    def _free(worker):
        from turboquant.vllm_attn_backend import free_kv_cache

        return free_kv_cache(worker.model_runner)

    freed = executor.collective_rpc(_free)
    after_free = executor.collective_rpc(inspect_worker)

    structural = []
    for before, after in zip(before_free, after_free):
        structural.append(
            {
                "active": summarize_structural_compression(
                    baseline_bytes=before["active_baseline_kv_bytes"],
                    compressed_bytes=before["tq_total_bytes"],
                    placeholder_bytes=after["hooked_kv_storage_bytes"],
                ),
                "allocated_backing": summarize_structural_compression(
                    baseline_bytes=before["hooked_kv_storage_bytes"],
                    compressed_bytes=before["tq_total_bytes"],
                    placeholder_bytes=after["hooked_kv_storage_bytes"],
                ),
            }
        )

    result = {
        "model": MODEL,
        "tp": TP,
        "gpu_mem_utilization": GPU_MEM,
        "max_model_len": MAX_MODEL_LEN,
        "input_tokens_target": INPUT_TOKENS,
        "actual_prompt_token_counts": prompt_token_counts,
        "concurrency": CONCURRENCY,
        "max_tokens": MAX_TOKENS,
        "mode": MODE,
        "enforce_eager": ENFORCE_EAGER,
        "hooks_per_rank": hooks,
        "before_free": before_free,
        "freed_bytes": freed,
        "after_free": after_free,
        "structural_summary": structural,
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
