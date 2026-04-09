from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark_logic import group_examples_by_target_tokens, summarize_retrieval_scores
from pr38479_runner import run_retrieval_batch_case
from retrieval_dataset import write_examples_jsonl


def run_retrieval_sweep(
    *,
    output_dir: str | Path,
    context_lengths: list[int],
    seeds_per_tier: int,
    modes: list[str],
    gpu_id: int = 1,
) -> list[dict[str, Any]]:
    out_dir = Path(output_dir)
    dataset_path = write_examples_jsonl(
        output_dir=out_dir / "generated",
        context_lengths=context_lengths,
        seeds_per_tier=seeds_per_tier,
    )

    examples = [json.loads(line) for line in dataset_path.read_text().splitlines() if line]
    grouped_examples = group_examples_by_target_tokens(examples)

    rows: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    for target_tokens, _grouped_rows in grouped_examples.items():
        for mode in modes:
            payload = run_retrieval_batch_case(
                mode=mode,
                dataset_path=dataset_path,
                output_dir=out_dir / "runs" / f"ctx{target_tokens}" / mode,
                gpu_id=gpu_id,
                target_tokens=max(target_tokens, 128),
                max_tokens=24,
            )
            batches.append(payload)
            batch_result = payload.get("result", {})
            batch_rows = batch_result.get("rows", [])
            if isinstance(batch_rows, list):
                for row in batch_rows:
                    merged_row = dict(row)
                    merged_row["mode"] = mode
                    merged_row["batch_output_json"] = payload["output_json"]
                    rows.append(merged_row)

    return {
        "dataset_path": str(dataset_path),
        "example_counts": {str(key): len(value) for key, value in grouped_examples.items()},
        "batches": batches,
        "rows": rows,
        "summary": summarize_retrieval_scores(rows),
        "contexts": [int(key) for key in grouped_examples.keys()],
    }
