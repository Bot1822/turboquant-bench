from __future__ import annotations

from typing import Any

from pr38479_runner import run_case


def _build_decode_prompt(prompt_prefix: str, target_len: int, max_tokens: int = 32) -> str:
    filler_sentence = " Detail retrieval benchmark filler sentence."
    # Leave headroom for tokenizer variance, prompt prefix, and output tokens.
    approx_budget = max(32, target_len - max_tokens - 96)
    # Roughly 6 tokens per filler sentence on this prompt family.
    repeats = max(1, approx_budget // 6)
    return prompt_prefix + "\n" + (filler_sentence * repeats).strip()


def run_decode_sweep(
    *,
    prompt_prefix: str,
    modes: list[str],
    context_lengths: list[int],
    output_dir: str,
    gpu_id: int = 1,
    max_tokens: int = 32,
    warmup_runs: int = 1,
    warmup_max_tokens: int = 0,
    measure_trials: int = 1,
    result_suffix: str | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for target_len in context_lengths:
        prompt = _build_decode_prompt(prompt_prefix, target_len, max_tokens=max_tokens)
        for mode in modes:
            results.append(
                run_case(
                    mode=mode,
                    prompt=prompt,
                    output_dir=output_dir,
                    gpu_id=gpu_id,
                    max_model_len=target_len,
                    max_tokens=max_tokens,
                    warmup_runs=warmup_runs,
                    warmup_max_tokens=warmup_max_tokens,
                    measure_trials=measure_trials,
                    result_suffix=result_suffix,
                )
            )
    return results
