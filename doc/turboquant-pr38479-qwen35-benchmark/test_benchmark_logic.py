from __future__ import annotations

import unittest
from types import SimpleNamespace

from benchmark_logic import (
    choose_effective_max_model_len,
    extract_generation_metrics,
    group_examples_by_target_tokens,
    summarize_batch_generation,
    summarize_retrieval_scores,
    summarize_trials,
)
from scoring import exact_match


class BenchmarkLogicTest(unittest.TestCase):
    def test_extract_generation_metrics_uses_request_stats(self) -> None:
        metrics = SimpleNamespace(
            scheduled_ts=10.0,
            first_token_ts=13.0,
            last_token_ts=21.0,
            first_token_latency=12.5,
        )

        result = extract_generation_metrics(
            request_metrics=metrics,
            generated_tokens=5,
            wall_time_seconds=25.0,
        )

        self.assertEqual(result["generated_tokens"], 5)
        self.assertEqual(result["wall_time_seconds"], 25.0)
        self.assertEqual(result["ttft_seconds"], 12.5)
        self.assertEqual(result["prefill_seconds"], 3.0)
        self.assertEqual(result["decode_seconds"], 8.0)
        self.assertEqual(result["inference_seconds"], 11.0)
        self.assertAlmostEqual(result["decode_tokens_per_s"], 0.5)
        self.assertAlmostEqual(result["mean_inter_token_latency_seconds"], 2.0)
        self.assertAlmostEqual(result["output_tokens_per_s"], 0.2)

    def test_summarize_trials_aggregates_successful_trials(self) -> None:
        trials = [
            {
                "status": "success",
                "wall_time_seconds": 10.0,
                "output_tokens_per_s": 3.2,
                "ttft_seconds": 1.5,
                "decode_tokens_per_s": 4.0,
            },
            {
                "status": "success",
                "wall_time_seconds": 14.0,
                "output_tokens_per_s": 2.8,
                "ttft_seconds": 2.5,
                "decode_tokens_per_s": 3.0,
            },
            {
                "status": "error",
                "error": "ignored in summary",
            },
        ]

        result = summarize_trials(trials)

        self.assertEqual(result["num_trials"], 3)
        self.assertEqual(result["successful_trials"], 2)
        self.assertAlmostEqual(result["mean_wall_time_seconds"], 12.0)
        self.assertAlmostEqual(result["median_wall_time_seconds"], 12.0)
        self.assertAlmostEqual(result["mean_output_tokens_per_s"], 3.0)
        self.assertAlmostEqual(result["mean_ttft_seconds"], 2.0)
        self.assertAlmostEqual(result["mean_decode_tokens_per_s"], 3.5)

    def test_group_examples_by_target_tokens(self) -> None:
        examples = [
            {"example_id": "a", "target_tokens": 4096},
            {"example_id": "b", "target_tokens": 8192},
            {"example_id": "c", "target_tokens": 4096},
        ]

        grouped = group_examples_by_target_tokens(examples)

        self.assertEqual(sorted(grouped.keys()), [4096, 8192])
        self.assertEqual([row["example_id"] for row in grouped[4096]], ["a", "c"])
        self.assertEqual([row["example_id"] for row in grouped[8192]], ["b"])

    def test_summarize_retrieval_scores(self) -> None:
        rows = [
            {
                "mode": "baseline",
                "tier": "A",
                "target_tokens": 8192,
                "score": {"exact_match": True},
            },
            {
                "mode": "baseline",
                "tier": "A",
                "target_tokens": 8192,
                "score": {"exact_match": False},
            },
            {
                "mode": "tq3",
                "tier": "B",
                "target_tokens": 8192,
                "score": {"exact_match": True},
            },
        ]

        summary = summarize_retrieval_scores(rows)

        self.assertEqual(
            summary["overall"]["baseline"]["8192"]["total_examples"],
            2,
        )
        self.assertEqual(
            summary["overall"]["baseline"]["8192"]["exact_matches"],
            1,
        )
        self.assertAlmostEqual(
            summary["overall"]["baseline"]["8192"]["accuracy"],
            0.5,
        )
        self.assertEqual(
            summary["by_tier"]["tq3"]["8192"]["B"]["exact_matches"],
            1,
        )

    def test_choose_effective_max_model_len_accounts_for_prompt_and_output_budget(self) -> None:
        result = choose_effective_max_model_len(
            prompt_token_counts=[10991, 11004, 11030],
            requested_target_tokens=8192,
            output_tokens=24,
            safety_margin=64,
        )

        self.assertEqual(result, 11118)

    def test_exact_match_ignores_empty_think_wrapper(self) -> None:
        prediction = "\n\n<think>\n\n</think>\n\nTQ-985440"
        self.assertTrue(exact_match(prediction, "TQ-985440"))

    def test_summarize_batch_generation(self) -> None:
        rows = [
            {"generated_tokens": 512, "decode_tokens_per_s": 14.0, "ttft_seconds": 0.4},
            {"generated_tokens": 500, "decode_tokens_per_s": 13.0, "ttft_seconds": 0.5},
            {"generated_tokens": 508, "decode_tokens_per_s": 15.0, "ttft_seconds": 0.6},
        ]

        summary = summarize_batch_generation(rows, wall_time_seconds=120.0)

        self.assertEqual(summary["batch_size"], 3)
        self.assertEqual(summary["total_generated_tokens"], 1520)
        self.assertAlmostEqual(summary["aggregate_output_tokens_per_s"], 1520 / 120.0)
        self.assertAlmostEqual(summary["mean_request_decode_tokens_per_s"], 14.0)
        self.assertAlmostEqual(summary["mean_request_ttft_seconds"], 0.5)


if __name__ == "__main__":
    unittest.main()
