from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from benchmark_logic import extract_generation_metrics, summarize_batch_generation
from decode_benchmark import _build_decode_prompt


def build_llm(mode: str, model: str, max_model_len: int, gpu_mem: float, batch_size: int) -> LLM:
    kv_cache_dtype = "auto" if mode == "baseline" else mode
    return LLM(
        model=model,
        dtype="bfloat16",
        kv_cache_dtype=kv_cache_dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_num_seqs=batch_size,
        disable_log_stats=False,
        trust_remote_code=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", required=True, choices=["baseline", "tq3", "tq4"])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--prompt-target", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gpu-mem", type=float, default=0.94)
    args = parser.parse_args()

    prompt = _build_decode_prompt(
        "Write one concise sentence about TurboQuant.",
        args.prompt_target,
        max_tokens=args.max_tokens,
    )
    prompts = [prompt for _ in range(args.batch_size)]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_tokens = len(tokenizer(prompt).input_ids)

    t0 = time.perf_counter()
    llm = build_llm(args.mode, args.model, args.max_model_len, args.gpu_mem, args.batch_size)
    load_seconds = time.perf_counter() - t0

    llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=min(64, args.max_tokens)))

    t1 = time.perf_counter()
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))
    wall_time_seconds = time.perf_counter() - t1

    per_request: list[dict[str, object]] = []
    for request_index, output in enumerate(outputs):
        completion = output.outputs[0]
        metrics = extract_generation_metrics(
            request_metrics=output.metrics,
            generated_tokens=len(completion.token_ids),
            wall_time_seconds=wall_time_seconds,
        )
        per_request.append(
            {
                "request_index": request_index,
                "generated_text": completion.text,
                **metrics,
            }
        )

    row = {
        "mode": args.mode,
        "batch_size": args.batch_size,
        "prompt_tokens": prompt_tokens,
        "load_seconds": load_seconds,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "block_size": llm.llm_engine.vllm_config.cache_config.block_size,
        "num_gpu_blocks": llm.llm_engine.vllm_config.cache_config.num_gpu_blocks,
        "summary": summarize_batch_generation(
            per_request,
            wall_time_seconds=wall_time_seconds,
        ),
        "per_request": per_request,
    }

    Path(args.output_json).write_text(
        json.dumps(row, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
