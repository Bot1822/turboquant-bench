from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from transformers import AutoTokenizer

from benchmark_logic import (
    choose_effective_max_model_len,
    summarize_retrieval_scores,
    summarize_trials,
)
from container_case_runner import (
    build_llm,
    query_nvidia_smi,
    query_torch_cuda_memory,
    run_generation_trial,
)
from scoring import score_prediction


def build_retrieval_prompt(example: dict[str, object]) -> str:
    return (
        "You are given a document.\n"
        "Read it carefully and answer the final question exactly.\n\n"
        f"{example['document']}\n\nQuestion: {example['question']}"
    )


def load_examples(dataset_path: str | Path, target_tokens: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in Path(dataset_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row["target_tokens"]) == target_tokens:
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["baseline", "tq3", "tq4"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--gpu-mem", type=float, required=True)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--warmup-max-tokens", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    result: dict[str, object] = {
        "mode": args.mode,
        "model": args.model,
        "dataset_path": args.dataset_path,
        "target_tokens": args.target_tokens,
        "gpu_memory_utilization": args.gpu_mem,
        "max_tokens": args.max_tokens,
        "memory_before": query_nvidia_smi(),
        "torch_memory_before": query_torch_cuda_memory(),
    }

    try:
        examples = load_examples(args.dataset_path, args.target_tokens)
        if not examples:
            raise ValueError(f"No retrieval examples found for target_tokens={args.target_tokens}")

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        prompt_rows: list[tuple[dict[str, object], str, int]] = []
        for example in examples:
            prompt = build_retrieval_prompt(example)
            prompt_tokens = len(tokenizer(prompt).input_ids)
            prompt_rows.append((example, prompt, prompt_tokens))

        longest_prompt = max(prompt_rows, key=lambda row: row[2])[1]
        prompt_token_counts = [row[2] for row in prompt_rows]
        effective_max_model_len = choose_effective_max_model_len(
            prompt_token_counts=prompt_token_counts,
            requested_target_tokens=args.target_tokens,
            output_tokens=args.max_tokens,
            safety_margin=64,
        )
        result["num_examples"] = len(prompt_rows)
        result["example_ids"] = [str(row[0]["example_id"]) for row in prompt_rows]
        result["max_prompt_tokens"] = max(prompt_token_counts)
        result["mean_prompt_tokens"] = sum(prompt_token_counts) / len(prompt_token_counts)
        result["effective_max_model_len"] = effective_max_model_len

        llm = build_llm(args.mode, args.model, effective_max_model_len, args.gpu_mem)
        result["memory_after_load"] = query_nvidia_smi()
        result["torch_memory_after_load"] = query_torch_cuda_memory()
        result["cache_dtype"] = llm.llm_engine.vllm_config.cache_config.cache_dtype
        result["block_size"] = llm.llm_engine.vllm_config.cache_config.block_size
        result["num_gpu_blocks"] = llm.llm_engine.vllm_config.cache_config.num_gpu_blocks

        warmup_max_tokens = args.warmup_max_tokens or args.max_tokens
        for warmup_index in range(args.warmup_runs):
            run_generation_trial(llm, longest_prompt, warmup_max_tokens, -1 - warmup_index)

        rows: list[dict[str, object]] = []
        for row_index, (example, prompt, prompt_tokens) in enumerate(prompt_rows):
            trial = run_generation_trial(llm, prompt, args.max_tokens, row_index)
            generated_text = str(trial.get("generated_text", ""))
            rows.append(
                {
                    "mode": args.mode,
                    "example_id": example["example_id"],
                    "tier": example["tier"],
                    "seed": example["seed"],
                    "target_tokens": example["target_tokens"],
                    "question": example["question"],
                    "answer": example["answer"],
                    "prompt_tokens": prompt_tokens,
                    **trial,
                    "score": score_prediction(generated_text, str(example["answer"])),
                }
            )

        result["rows"] = rows
        result["summary"] = summarize_retrieval_scores(rows)
        result["generation_summary"] = summarize_trials(rows)
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
