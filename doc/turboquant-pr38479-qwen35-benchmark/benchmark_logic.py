from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.mean(values))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _non_negative_diff(later: float | None, earlier: float | None) -> float | None:
    if later is None or earlier is None:
        return None
    return max(float(later) - float(earlier), 0.0)


def _safe_ratio(numerator: float, denominator: float | None) -> float | None:
    if denominator is None or denominator <= 0:
        return None
    return float(numerator / denominator)


def extract_generation_metrics(
    *,
    request_metrics: Any | None,
    generated_tokens: int,
    wall_time_seconds: float,
) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {
        "generated_tokens": generated_tokens,
        "wall_time_seconds": wall_time_seconds,
        "output_tokens_per_s": _safe_ratio(generated_tokens, wall_time_seconds),
        "ttft_seconds": None,
        "prefill_seconds": None,
        "decode_seconds": None,
        "inference_seconds": None,
        "decode_tokens_per_s": None,
        "mean_inter_token_latency_seconds": None,
        "mean_time_per_output_token_seconds": None,
    }
    if request_metrics is None:
        return result

    scheduled_ts = getattr(request_metrics, "scheduled_ts", None)
    first_token_ts = getattr(request_metrics, "first_token_ts", None)
    last_token_ts = getattr(request_metrics, "last_token_ts", None)
    ttft_seconds = getattr(request_metrics, "first_token_latency", None)

    prefill_seconds = _non_negative_diff(first_token_ts, scheduled_ts)
    decode_seconds = _non_negative_diff(last_token_ts, first_token_ts)
    inference_seconds = _non_negative_diff(last_token_ts, scheduled_ts)

    result.update(
        {
            "ttft_seconds": float(ttft_seconds) if ttft_seconds is not None else None,
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "inference_seconds": inference_seconds,
            "decode_tokens_per_s": _safe_ratio(max(generated_tokens - 1, 0), decode_seconds),
            "mean_inter_token_latency_seconds": _safe_ratio(
                decode_seconds or 0.0,
                max(generated_tokens - 1, 0),
            ),
            "mean_time_per_output_token_seconds": _safe_ratio(
                inference_seconds or 0.0,
                generated_tokens,
            ),
        }
    )
    return result


def summarize_trials(trials: list[dict[str, Any]]) -> dict[str, float | int | None]:
    success_rows = [row for row in trials if row.get("status") == "success"]

    def values(key: str) -> list[float]:
        out: list[float] = []
        for row in success_rows:
            value = row.get(key)
            if isinstance(value, (int, float)):
                out.append(float(value))
        return out

    return {
        "num_trials": len(trials),
        "successful_trials": len(success_rows),
        "mean_wall_time_seconds": _mean(values("wall_time_seconds")),
        "median_wall_time_seconds": _median(values("wall_time_seconds")),
        "mean_output_tokens_per_s": _mean(values("output_tokens_per_s")),
        "mean_ttft_seconds": _mean(values("ttft_seconds")),
        "mean_decode_tokens_per_s": _mean(values("decode_tokens_per_s")),
        "mean_inter_token_latency_seconds": _mean(
            values("mean_inter_token_latency_seconds")
        ),
    }


def summarize_batch_generation(
    rows: list[dict[str, Any]],
    *,
    wall_time_seconds: float,
) -> dict[str, float | int | None]:
    generated_tokens = [
        int(row["generated_tokens"])
        for row in rows
        if isinstance(row.get("generated_tokens"), (int, float))
    ]
    decode_tok_s = [
        float(row["decode_tokens_per_s"])
        for row in rows
        if isinstance(row.get("decode_tokens_per_s"), (int, float))
    ]
    ttft_s = [
        float(row["ttft_seconds"])
        for row in rows
        if isinstance(row.get("ttft_seconds"), (int, float))
    ]
    total_generated_tokens = sum(generated_tokens)
    return {
        "batch_size": len(rows),
        "total_generated_tokens": total_generated_tokens,
        "wall_time_seconds": wall_time_seconds,
        "aggregate_output_tokens_per_s": _safe_ratio(
            total_generated_tokens,
            wall_time_seconds,
        ),
        "mean_request_decode_tokens_per_s": _mean(decode_tok_s),
        "mean_request_ttft_seconds": _mean(ttft_s),
    }


def group_examples_by_target_tokens(
    examples: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in examples:
        grouped[int(row["target_tokens"])].append(row)
    return dict(grouped)


def choose_effective_max_model_len(
    *,
    prompt_token_counts: list[int],
    requested_target_tokens: int,
    output_tokens: int,
    safety_margin: int = 64,
) -> int:
    prompt_budget = max(prompt_token_counts) if prompt_token_counts else 0
    required_budget = prompt_budget + max(output_tokens, 0) + max(safety_margin, 0)
    return max(int(requested_target_tokens), int(required_budget))


def _build_accuracy_row() -> dict[str, float | int]:
    return {"total_examples": 0, "exact_matches": 0, "accuracy": 0.0}


def _finalize_accuracy_rows(rows: dict[str, Any]) -> dict[str, Any]:
    finalized: dict[str, Any] = {}
    for key, value in rows.items():
        if isinstance(value, dict) and "total_examples" in value:
            total = int(value["total_examples"])
            exact = int(value["exact_matches"])
            finalized[key] = {
                "total_examples": total,
                "exact_matches": exact,
                "accuracy": float(exact / total) if total else 0.0,
            }
        else:
            finalized[key] = _finalize_accuracy_rows(value)
    return finalized


def summarize_retrieval_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    overall: dict[str, Any] = defaultdict(lambda: defaultdict(_build_accuracy_row))
    by_tier: dict[str, Any] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(_build_accuracy_row))
    )

    for row in rows:
        mode = str(row["mode"])
        context = str(row["target_tokens"])
        tier = str(row["tier"])
        exact_match = bool(row.get("score", {}).get("exact_match", False))

        overall_row = overall[mode][context]
        overall_row["total_examples"] += 1
        overall_row["exact_matches"] += int(exact_match)

        tier_row = by_tier[mode][context][tier]
        tier_row["total_examples"] += 1
        tier_row["exact_matches"] += int(exact_match)

    return {
        "overall": _finalize_accuracy_rows(dict(overall)),
        "by_tier": _finalize_accuracy_rows(dict(by_tier)),
    }
