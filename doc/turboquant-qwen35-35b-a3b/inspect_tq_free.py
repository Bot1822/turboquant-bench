#!/usr/bin/env python3
"""Inspect TurboQuant KV-cache free behavior on the requested model."""

from __future__ import annotations

import json
import os
import gc

import torch
from vllm import LLM, SamplingParams


MODEL = os.environ.get("MODEL", "/share/models/official/Qwen3.5-35B-A3B")
TP = int(os.environ.get("TP", "2"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))
GPU_MEM = float(os.environ.get("GPU_MEM", "0.85"))
ENFORCE_EAGER = os.environ.get("ENFORCE_EAGER", "0") == "1"
MODE = os.environ.get("MODE", "active")
PROMPT = os.environ.get(
    "PROMPT",
    "Briefly explain what KV cache compression is and why it matters for long-context inference.",
)


def inspect_worker(worker, tracked_storage_ptrs=None):
    static_ctx = worker.model_runner.compilation_config.static_forward_context
    tq_states = getattr(worker.model_runner, "_tq_states", {})

    layer_entries = []
    for layer_name in tq_states:
        attn_module = static_ctx.get(layer_name)
        kv_list = getattr(attn_module, "kv_cache", None)
        if not kv_list:
            continue
        tensor = kv_list[0]
        storage = tensor.untyped_storage()
        layer_entries.append({
            "layer_name": layer_name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "data_ptr": int(tensor.data_ptr()),
            "storage_ptr": int(storage.data_ptr()),
            "storage_nbytes": int(storage.nbytes()),
            "numel_bytes": int(tensor.nelement() * tensor.element_size()),
        })

    runner_entries = []
    for entry in worker.model_runner.kv_caches:
        if isinstance(entry, list):
            tensor = entry[0]
        else:
            tensor = entry
        if not hasattr(tensor, "untyped_storage"):
            continue
        storage = tensor.untyped_storage()
        runner_entries.append({
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "data_ptr": int(tensor.data_ptr()),
            "storage_ptr": int(storage.data_ptr()),
            "storage_nbytes": int(storage.nbytes()),
        })

    unique_layer_storage = {
        (entry["storage_ptr"], entry["storage_nbytes"]) for entry in layer_entries
    }

    leaked_tensors = []
    if tracked_storage_ptrs:
        tracked_storage_ptrs = set(tracked_storage_ptrs)
        for obj in gc.get_objects():
            if not isinstance(obj, torch.Tensor):
                continue
            if not obj.is_cuda:
                continue
            try:
                storage = obj.untyped_storage()
                storage_ptr = int(storage.data_ptr())
            except Exception:
                continue
            if storage_ptr in tracked_storage_ptrs:
                leaked_tensors.append({
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "storage_ptr": storage_ptr,
                    "storage_nbytes": int(storage.nbytes()),
                })

    return {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
        "layer_tensor_ptrs": len({entry["data_ptr"] for entry in layer_entries}),
        "layer_storage_ptrs": len({entry["storage_ptr"] for entry in layer_entries}),
        "runner_tensor_ptrs": len({entry["data_ptr"] for entry in runner_entries}),
        "runner_storage_ptrs": len({entry["storage_ptr"] for entry in runner_entries}),
        "sum_layer_numel_bytes": sum(entry["numel_bytes"] for entry in layer_entries),
        "sum_unique_layer_storage_bytes": sum(size for _, size in unique_layer_storage),
        "sample_layers": layer_entries[:3],
        "sample_runner_entries": runner_entries[:3],
        "tracked_storage_ptrs": [entry["storage_ptr"] for entry in layer_entries],
        "leaked_tensor_count": len(leaked_tensors),
        "sample_leaked_tensors": leaked_tensors[:10],
    }


def main():
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
        tq_states = install_turboquant_hooks(
            worker.model_runner,
            key_bits=3,
            value_bits=2,
            buffer_size=128,
            mode=mode_map[MODE],
        )
        return len(tq_states)

    hooks = executor.collective_rpc(_install)

    before = executor.collective_rpc(inspect_worker)
    llm.generate([PROMPT], SamplingParams(temperature=0, max_tokens=64))
    after_gen = executor.collective_rpc(inspect_worker)
    tracked_storage_ptrs = []
    for worker_info in after_gen:
        tracked_storage_ptrs.extend(worker_info["tracked_storage_ptrs"])

    def _free(worker):
        from turboquant.vllm_attn_backend import free_kv_cache
        return free_kv_cache(worker.model_runner)

    freed = executor.collective_rpc(_free)
    after_free = executor.collective_rpc(
        lambda worker, _ptrs=tracked_storage_ptrs: inspect_worker(worker, _ptrs)
    )

    print(json.dumps({
        "hooks": hooks,
        "enforce_eager": ENFORCE_EAGER,
        "mode": MODE,
        "before": before,
        "after_gen": after_gen,
        "freed": freed,
        "after_free": after_free,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
