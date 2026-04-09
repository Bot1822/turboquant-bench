from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from benchmark_logic import extract_generation_metrics, summarize_trials


def query_nvidia_smi() -> list[dict[str, int]]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    rows: list[dict[str, int]] = []
    for line in proc.stdout.strip().splitlines():
        if not line.strip():
            continue
        idx, mem, util = [x.strip() for x in line.split(",")]
        rows.append(
            {"index": int(idx), "memory_used_mb": int(mem), "utilization_gpu": int(util)}
        )
    return rows


def query_torch_cuda_memory() -> dict[str, float] | None:
    if not torch.cuda.is_available():
        return None
    device = torch.cuda.current_device()
    return {
        "device": float(device),
        "allocated_mb": torch.cuda.memory_allocated(device) / (1024 * 1024),
        "reserved_mb": torch.cuda.memory_reserved(device) / (1024 * 1024),
        "max_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
        "max_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024 * 1024),
    }


def build_llm(mode: str, model: str, max_model_len: int, gpu_mem: float) -> LLM:
    kv_cache_dtype = "auto" if mode == "baseline" else mode
    return LLM(
        model=model,
        dtype="bfloat16",
        kv_cache_dtype=kv_cache_dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_num_seqs=1,
        disable_log_stats=False,
        trust_remote_code=True,
    )


def run_generation_trial(llm: LLM, prompt: str, max_tokens: int, trial_index: int) -> dict[str, object]:
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    started_at = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params)
    finished_at = time.perf_counter()

    request_output = outputs[0]
    completion = request_output.outputs[0]
    metrics = extract_generation_metrics(
        request_metrics=request_output.metrics,
        generated_tokens=len(completion.token_ids),
        wall_time_seconds=finished_at - started_at,
    )
    return {
        "trial_index": trial_index,
        "status": "success",
        "generated_text": completion.text,
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["baseline", "tq3", "tq4"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--gpu-mem", type=float, required=True)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--warmup-max-tokens", type=int, default=0)
    parser.add_argument("--measure-trials", type=int, default=1)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    result: dict[str, object] = {
        "mode": args.mode,
        "model": args.model,
        "prompt": args.prompt,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_mem,
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        },
        "memory_before": query_nvidia_smi(),
        "torch_memory_before": query_torch_cuda_memory(),
    }

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        prompt_tokens = len(tokenizer(args.prompt).input_ids)
        result["prompt_tokens"] = prompt_tokens

        t0 = time.perf_counter()
        llm = build_llm(args.mode, args.model, args.max_model_len, args.gpu_mem)
        t1 = time.perf_counter()
        result["load_seconds"] = t1 - t0
        result["memory_after_load"] = query_nvidia_smi()
        result["torch_memory_after_load"] = query_torch_cuda_memory()
        result["warmup_runs"] = args.warmup_runs
        warmup_max_tokens = args.warmup_max_tokens or args.max_tokens
        result["warmup_max_tokens"] = warmup_max_tokens
        result["measure_trials"] = args.measure_trials
        result["cache_dtype"] = llm.llm_engine.vllm_config.cache_config.cache_dtype
        result["block_size"] = llm.llm_engine.vllm_config.cache_config.block_size
        result["num_gpu_blocks"] = llm.llm_engine.vllm_config.cache_config.num_gpu_blocks

        for _ in range(args.warmup_runs):
            llm.generate(
                [args.prompt], SamplingParams(temperature=0.0, max_tokens=warmup_max_tokens)
            )

        trials: list[dict[str, object]] = []
        for trial_index in range(args.measure_trials):
            trials.append(run_generation_trial(llm, args.prompt, args.max_tokens, trial_index))

        result["trials"] = trials
        result["trial_summary"] = summarize_trials(trials)
        final_trial = trials[-1]
        result["generation_seconds"] = final_trial.get("wall_time_seconds")
        result["generated_text"] = final_trial.get("generated_text")
        result["generated_tokens"] = final_trial.get("generated_tokens")
        result["output_tokens_per_s"] = final_trial.get("output_tokens_per_s")
        result["ttft_seconds"] = final_trial.get("ttft_seconds")
        result["prefill_seconds"] = final_trial.get("prefill_seconds")
        result["decode_seconds"] = final_trial.get("decode_seconds")
        result["inference_seconds"] = final_trial.get("inference_seconds")
        result["decode_tokens_per_s"] = final_trial.get("decode_tokens_per_s")
        result["mean_inter_token_latency_seconds"] = final_trial.get(
            "mean_inter_token_latency_seconds"
        )
        result["mean_time_per_output_token_seconds"] = final_trial.get(
            "mean_time_per_output_token_seconds"
        )
        result["memory_after_generate"] = query_nvidia_smi()
        result["torch_memory_after_generate"] = query_torch_cuda_memory()
        result["status"] = "success"
    except Exception as exc:  # pragma: no cover - runtime harness path
        result["status"] = "error"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        result["memory_on_error"] = query_nvidia_smi()
        result["torch_memory_on_error"] = query_torch_cuda_memory()

    Path(args.output_json).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
