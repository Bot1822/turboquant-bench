from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_helpers import append_markdown_row, ensure_dir, utc_timestamp, write_json
from decode_benchmark import run_decode_sweep
from pr38479_runner import query_host_gpu_memory
from retrieval_benchmark import run_retrieval_sweep


EXPERIMENT_LOG = Path(
    "/ceph/User/E01442/turboquant/doc/"
    "turboquant-pr38479-qwen35-benchmark/04-experiments.md"
)
OUTPUT_ROOT = Path(
    "/ceph/User/E01442/turboquant/doc/"
    "turboquant-pr38479-qwen35-benchmark/results"
)


def _fmt_float(value: object, digits: int = 2) -> str:
    if not isinstance(value, (int, float)):
        return "na"
    return f"{float(value):.{digits}f}"

def choose_gpu(requested_gpu: int) -> int:
    if requested_gpu >= 0:
        return requested_gpu
    usage = query_host_gpu_memory()
    candidates = sorted((mem, idx) for idx, mem in usage.items() if mem <= 1024)
    if not candidates:
        raise RuntimeError(f"No clean GPU found: {usage}")
    return candidates[0][1]



def log_result(area: str, mode: str, context: str, gpu: int, command: str, result: str, metrics: str, notes: str) -> None:
    append_markdown_row(
        EXPERIMENT_LOG,
        [utc_timestamp(), area, mode, context, str(gpu), command, result, metrics, notes],
)


def format_decode_metrics(result: dict[str, object]) -> str:
    trial_summary = result.get("trial_summary", {})
    if not isinstance(trial_summary, dict):
        trial_summary = {}
    return ", ".join(
        [
            f"load_s={_fmt_float(result.get('load_seconds'))}",
            f"wall_s={_fmt_float(trial_summary.get('mean_wall_time_seconds', result.get('generation_seconds')))}",
            f"wall_tok_s={_fmt_float(trial_summary.get('mean_output_tokens_per_s', result.get('output_tokens_per_s')))}",
            f"ttft_s={_fmt_float(trial_summary.get('mean_ttft_seconds', result.get('ttft_seconds')))}",
            f"decode_tok_s={_fmt_float(trial_summary.get('mean_decode_tokens_per_s', result.get('decode_tokens_per_s')))}",
            f"prompt_tokens={result.get('prompt_tokens', 'na')}",
            f"block={result.get('block_size', 'na')}",
            f"gpu_blocks={result.get('num_gpu_blocks', 'na')}",
        ]
    )


def format_retrieval_metrics(result: dict[str, object], mode: str, context: str) -> str:
    summary = result.get("summary", {})
    generation_summary = result.get("generation_summary", {})
    if not isinstance(summary, dict):
        summary = {}
    if not isinstance(generation_summary, dict):
        generation_summary = {}
    overall = (
        summary.get("overall", {})
        .get(mode, {})
        .get(context, {})
    )
    if not isinstance(overall, dict):
        overall = {}
    exact_matches = overall.get("exact_matches", "na")
    total_examples = overall.get("total_examples", "na")
    return ", ".join(
        [
            f"acc={_fmt_float(overall.get('accuracy'))}",
            f"exact={exact_matches}/{total_examples}",
            f"wall_s={_fmt_float(generation_summary.get('mean_wall_time_seconds'))}",
            f"ttft_s={_fmt_float(generation_summary.get('mean_ttft_seconds'))}",
            f"decode_tok_s={_fmt_float(generation_summary.get('mean_decode_tokens_per_s'))}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--area",
        choices=["decode", "decode_steady", "retrieval", "all"],
        default="all",
    )
    parser.add_argument("--gpu", type=int, default=-1)
    args = parser.parse_args()

    ensure_dir(OUTPUT_ROOT)

    gpu = choose_gpu(args.gpu)

    if args.area in ("decode", "decode_steady", "all"):
        decode_results = run_decode_sweep(
            prompt_prefix="Write one short sentence about TurboQuant.",
            modes=["baseline", "tq3", "tq4"],
            context_lengths=[1024, 4096, 8192],
            output_dir=str(OUTPUT_ROOT / "decode_steady"),
            gpu_id=gpu,
            max_tokens=32,
            warmup_runs=1,
            warmup_max_tokens=32,
            measure_trials=3,
            result_suffix="steady",
        )
        write_json(OUTPUT_ROOT / "decode_steady_results.json", decode_results)
        for row in decode_results:
            proc = row["process"]
            metrics = ""
            mode = row["mode"]
            context = str(row["max_model_len"])
            if "result" in row:
                result = row["result"]
                if result.get("status") == "success":
                    metrics = format_decode_metrics(result)
                    status = "success"
                    notes = (
                        f"steady decode; warmup_runs={result.get('warmup_runs')}, "
                        f"measure_trials={result.get('measure_trials')}"
                    )
                else:
                    status = "failure"
                    notes = str(result.get("error", ""))[:200]
            else:
                notes = (proc.get("stderr") or proc.get("stdout") or "")[:200]
                status = "failure"
            log_result(
                "decode_steady",
                mode,
                context,
                gpu,
                "docker run ... container_case_runner.py",
                status,
                metrics,
                notes,
            )

    if args.area in ("retrieval", "all"):
        retrieval_payload = run_retrieval_sweep(
            output_dir=OUTPUT_ROOT / "retrieval",
            context_lengths=[8192, 32768],
            seeds_per_tier=1,
            modes=["baseline", "tq3", "tq4"],
            gpu_id=gpu,
        )
        write_json(OUTPUT_ROOT / "retrieval_results.json", retrieval_payload)
        for batch in retrieval_payload["batches"]:
            mode = batch["mode"]
            context = str(batch["target_tokens"])
            result = batch.get("result", {})
            if isinstance(result, dict) and result.get("status") == "success":
                status = "success"
                metrics = format_retrieval_metrics(result, mode, context)
                notes = (
                    f"examples={result.get('num_examples')}, "
                    f"max_prompt_tokens={result.get('max_prompt_tokens')}, "
                    f"effective_max_model_len={result.get('effective_max_model_len')}, "
                    f"dataset={Path(str(result.get('dataset_path', ''))).name}"
                )
            else:
                status = "failure"
                metrics = ""
                notes = str(result.get("error", ""))[:200] if isinstance(result, dict) else ""
            log_result(
                "retrieval",
                mode,
                context,
                gpu,
                "docker run ... container_retrieval_runner.py",
                status,
                metrics,
                notes,
            )


if __name__ == "__main__":
    main()
