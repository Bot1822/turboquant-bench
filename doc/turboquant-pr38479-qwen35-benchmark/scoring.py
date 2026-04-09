from __future__ import annotations

import re


def normalize_answer(text: str) -> str:
    text = re.sub(r"(?is)^\s*(<think>.*?</think>\s*)+", "", text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match(prediction: str, answer: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(answer)


def score_prediction(prediction: str, answer: str) -> dict[str, object]:
    return {
        "prediction": prediction,
        "answer": answer,
        "exact_match": exact_match(prediction, answer),
    }
