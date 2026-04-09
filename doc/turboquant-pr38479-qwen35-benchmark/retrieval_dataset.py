from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmark_helpers import ensure_dir


FILLER_SENTENCE = (
    "This section contains routine operational notes about scheduling, "
    "compliance tracking, review checkpoints, and archival handling."
)


@dataclass
class RetrievalExample:
    example_id: str
    tier: str
    target_tokens: int
    seed: int
    question: str
    answer: str
    document: str


def _repeat_filler(min_words: int) -> list[str]:
    words = 0
    out: list[str] = []
    while words < min_words:
        out.append(FILLER_SENTENCE)
        words += len(FILLER_SENTENCE.split())
    return out


def _insert_at_fraction(lines: list[str], fraction: float, content: str) -> None:
    idx = min(len(lines), max(0, int(len(lines) * fraction)))
    lines.insert(idx, content)


def build_single_detail_example(target_tokens: int, seed: int) -> RetrievalExample:
    rnd = random.Random(seed)
    ticket = f"TQ-{rnd.randint(100000, 999999)}"
    lines = _repeat_filler(target_tokens)
    _insert_at_fraction(lines, 0.72, f"Audit note: the recovery ticket is {ticket}.")
    doc = "\n".join(lines)
    return RetrievalExample(
        example_id=f"tierA-{target_tokens}-{seed}",
        tier="A",
        target_tokens=target_tokens,
        seed=seed,
        question="What is the recovery ticket? Answer with the exact ticket only.",
        answer=ticket,
        document=doc,
    )


def build_multi_detail_example(target_tokens: int, seed: int) -> RetrievalExample:
    rnd = random.Random(seed)
    event_id = f"EV-{rnd.randint(1000, 9999)}"
    due_date = f"2026-{rnd.randint(1,12):02d}-{rnd.randint(1,28):02d}"
    lines = _repeat_filler(target_tokens)
    _insert_at_fraction(lines, 0.21, f"Planning note: event id {event_id}.")
    _insert_at_fraction(lines, 0.83, f"Final approval deadline: {due_date}.")
    doc = "\n".join(lines)
    return RetrievalExample(
        example_id=f"tierB-{target_tokens}-{seed}",
        tier="B",
        target_tokens=target_tokens,
        seed=seed,
        question=(
            "Return EVENT_ID|DATE using the exact values from the document only."
        ),
        answer=f"{event_id}|{due_date}",
        document=doc,
    )


def build_adversarial_example(target_tokens: int, seed: int) -> RetrievalExample:
    rnd = random.Random(seed)
    good_id = f"ZX-{rnd.randint(1000, 9999)}"
    bad_ids = [f"ZX-{rnd.randint(1000, 9999)}" for _ in range(4)]
    lines = _repeat_filler(target_tokens)
    for frac, bad in zip((0.15, 0.35, 0.55, 0.75), bad_ids):
        _insert_at_fraction(lines, frac, f"Reference candidate id {bad}.")
    _insert_at_fraction(lines, 0.62, f"Signed release identifier: {good_id}.")
    doc = "\n".join(lines)
    return RetrievalExample(
        example_id=f"tierC-{target_tokens}-{seed}",
        tier="C",
        target_tokens=target_tokens,
        seed=seed,
        question="What is the signed release identifier? Answer with the exact id only.",
        answer=good_id,
        document=doc,
    )


def build_examples(
    *,
    context_lengths: list[int],
    seeds_per_tier: int,
) -> list[RetrievalExample]:
    examples: list[RetrievalExample] = []
    for target_tokens in context_lengths:
        for seed in range(seeds_per_tier):
            examples.append(build_single_detail_example(target_tokens, seed))
            examples.append(build_multi_detail_example(target_tokens, seed))
            examples.append(build_adversarial_example(target_tokens, seed))
    return examples


def write_examples_jsonl(
    *,
    output_dir: str | Path,
    context_lengths: list[int],
    seeds_per_tier: int,
) -> Path:
    out_dir = ensure_dir(output_dir)
    out_path = out_dir / "retrieval_examples.jsonl"
    rows = build_examples(
        context_lengths=context_lengths,
        seeds_per_tier=seeds_per_tier,
    )
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json_line(asdict(row)))
    return out_path


def json_line(data: dict) -> str:
    import json

    return json.dumps(data, ensure_ascii=False) + "\n"
