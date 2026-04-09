#!/usr/bin/env python3
"""Run lm_eval on baseline or TurboQuant-backed vLLM."""

from __future__ import annotations

import json
import os
import time

from lm_eval import simple_evaluate
from lm_eval.models.vllm_causallms import VLLM
from lm_eval.tasks import TaskManager

from turboquant.vllm_attn_backend import enable_no_alloc


MODEL = os.environ.get("MODEL", "/share/models/official/Qwen3-30B-A3B-Instruct-2507")
TASKS = [task for task in os.environ.get("TASKS", "mmlu_pro").split(",") if task]
MODE = os.environ.get("MODE", "baseline")
TP = int(os.environ.get("TP", "2"))
GPU_MEM = float(os.environ.get("GPU_MEM", "0.85"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "16384"))
BATCH_SIZE = os.environ.get("LM_EVAL_BATCH_SIZE", "auto")
LIMIT = os.environ.get("LIMIT")
NUM_FEWSHOT = os.environ.get("NUM_FEWSHOT")
GEN_KWARGS = os.environ.get("GEN_KWARGS", "temperature=0,top_p=1.0")


def pick_metric(task_result: dict) -> tuple[str, float]:
    for key in (
        "acc_norm,none",
        "acc,none",
        "exact_match,none",
        "exact_match,custom-extract",
    ):
        if key in task_result:
            return key, float(task_result[key])
    raise KeyError(f"No supported accuracy metric found in result keys: {list(task_result)}")


def main():
    if MODE == "tq_no_alloc":
        enable_no_alloc()

    task_manager = TaskManager()
    model = VLLM(
        pretrained=MODEL,
        tensor_parallel_size=TP,
        gpu_memory_utilization=GPU_MEM,
        max_model_len=MAX_MODEL_LEN,
        batch_size=BATCH_SIZE,
        trust_remote_code=True,
    )

    limit = None if LIMIT in (None, "", "none", "None") else float(LIMIT)
    num_fewshot = None if NUM_FEWSHOT in (None, "", "none", "None") else int(NUM_FEWSHOT)

    t0 = time.perf_counter()
    results = simple_evaluate(
        model=model,
        tasks=TASKS,
        batch_size=BATCH_SIZE,
        task_manager=task_manager,
        limit=limit,
        num_fewshot=num_fewshot,
        gen_kwargs=GEN_KWARGS,
        log_samples=False,
        verbosity="INFO",
    )["results"]
    elapsed = time.perf_counter() - t0

    summary = {}
    for task_name in TASKS:
        metric_name, metric_val = pick_metric(results[task_name])
        summary[task_name] = {
            "metric": metric_name,
            "score": round(metric_val * 100, 2),
        }

    avg = round(sum(item["score"] for item in summary.values()) / max(len(summary), 1), 2)
    print(json.dumps({
        "mode": MODE,
        "model": MODEL,
        "tasks": TASKS,
        "tp": TP,
        "gpu_mem_utilization": GPU_MEM,
        "max_model_len": MAX_MODEL_LEN,
        "batch_size": BATCH_SIZE,
        "limit": limit,
        "num_fewshot": num_fewshot,
        "elapsed_s": round(elapsed, 3),
        "summary": summary,
        "acc_avg": avg,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
