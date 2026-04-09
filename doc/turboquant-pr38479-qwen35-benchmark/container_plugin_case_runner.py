from __future__ import annotations

import argparse
import importlib.metadata
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
    # Do not touch torch.cuda in the parent process for this runner.
    # vLLM 0.19 uses multiprocessing for the engine core, and probing CUDA
    # here can poison later worker initialization via fork.
    return None


def build_llm(
    mode: str,
    model: str,
    max_model_len: int,
    gpu_mem: float,
    dtype: str,
    attention_backend: str | None,
    kv_cache_dtype: str | None,
) -> LLM:
    kwargs = dict(
        model=model,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_num_seqs=1,
        disable_log_stats=False,
        trust_remote_code=True,
    )
    if kv_cache_dtype:
        kwargs["kv_cache_dtype"] = kv_cache_dtype
    if mode == "plugin_tq4":
        from turboquant_vllm.vllm import register_tq4_backend

        register_tq4_backend()
        kwargs["attention_backend"] = "CUSTOM"
    elif attention_backend:
        kwargs["attention_backend"] = attention_backend
    return LLM(**kwargs)


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


def safe_version(pkg: str) -> str | None:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["baseline", "plugin_tq4"])
    parser.add_argument("--model", required=True)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--prompt-file")
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--gpu-mem", type=float, required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-backend")
    parser.add_argument("--kv-cache-dtype")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--warmup-max-tokens", type=int, default=0)
    parser.add_argument("--measure-trials", type=int, default=1)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")

    result: dict[str, object] = {
        "mode": args.mode,
        "model": args.model,
        "prompt": prompt,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_mem,
        "dtype": args.dtype,
        "requested_attention_backend": args.attention_backend,
        "requested_kv_cache_dtype": args.kv_cache_dtype,
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
            "TQ4_K_BITS": os.environ.get("TQ4_K_BITS"),
            "TQ4_V_BITS": os.environ.get("TQ4_V_BITS"),
            "TQ4_USE_FUSED_PAGED": os.environ.get("TQ4_USE_FUSED_PAGED"),
            "TQ4_USE_INT8_PREFILL": os.environ.get("TQ4_USE_INT8_PREFILL"),
        },
        "versions": {
            "python": subprocess.run(
                ["python3", "--version"], capture_output=True, text=True, check=False
            ).stdout.strip(),
            "vllm": safe_version("vllm"),
            "turboquant-vllm": safe_version("turboquant-vllm"),
            "transformers": safe_version("transformers"),
            "torch": safe_version("torch"),
        },
        "memory_before": query_nvidia_smi(),
        "torch_memory_before": query_torch_cuda_memory(),
    }

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        prompt_tokens = len(tokenizer(prompt).input_ids)
        result["prompt_tokens"] = prompt_tokens

        t0 = time.perf_counter()
        llm = build_llm(
            args.mode,
            args.model,
            args.max_model_len,
            args.gpu_mem,
            args.dtype,
            args.attention_backend,
            args.kv_cache_dtype,
        )
        t1 = time.perf_counter()
        result["load_seconds"] = t1 - t0
        result["memory_after_load"] = query_nvidia_smi()
        result["torch_memory_after_load"] = query_torch_cuda_memory()
        result["warmup_runs"] = args.warmup_runs
        warmup_max_tokens = args.warmup_max_tokens or args.max_tokens
        result["warmup_max_tokens"] = warmup_max_tokens
        result["measure_trials"] = args.measure_trials

        engine_config = llm.llm_engine.vllm_config
        result["cache_dtype"] = engine_config.cache_config.cache_dtype
        result["attention_backend"] = str(engine_config.attention_config.backend)
        result["block_size"] = engine_config.cache_config.block_size
        result["num_gpu_blocks"] = engine_config.cache_config.num_gpu_blocks

        for _ in range(args.warmup_runs):
            llm.generate(
                [prompt], SamplingParams(temperature=0.0, max_tokens=warmup_max_tokens)
            )

        trials: list[dict[str, object]] = []
        for trial_index in range(args.measure_trials):
            trials.append(run_generation_trial(llm, prompt, args.max_tokens, trial_index))

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
    except Exception as exc:  # pragma: no cover
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
