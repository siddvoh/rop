"""Official-compatible answer scoring for HybridQA.

Normalization and per-answer EM/F1 calculations mirror HybridQA's released
``evaluate_script.py``. Aggregate helpers add validation and deterministic,
conservative handling of missing predictions: a missing model output receives
zero EM and zero F1 even if the reference normalizes to an empty string.

All aggregate EM and F1 values are percentages in the closed interval
``[0, 100]``, matching the official evaluator's output scale.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Iterable, Mapping
import re
import string
from typing import Any


_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_ASCII_PUNCTUATION = frozenset(string.punctuation)


def normalize_answer(answer: str) -> str:
    """Apply the normalization used by the released HybridQA evaluator."""

    lowered = answer.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in _ASCII_PUNCTUATION
    )
    without_articles = _ARTICLES.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def get_tokens(answer: str) -> list[str]:
    """Return official-normalized whitespace tokens."""

    if not answer:
        return []
    return normalize_answer(answer).split()


def compute_exact(gold: str, prediction: str) -> int:
    """Return normalized exact match as ``0`` or ``1``."""

    return int(normalize_answer(gold) == normalize_answer(prediction))


def compute_f1(gold: str, prediction: str) -> float:
    """Return multiset token F1 on the official ``[0, 1]`` scale."""

    gold_tokens = get_tokens(gold)
    prediction_tokens = get_tokens(prediction)
    overlap = Counter(gold_tokens) & Counter(prediction_tokens)
    shared = sum(overlap.values())

    if not gold_tokens or not prediction_tokens:
        return float(gold_tokens == prediction_tokens)
    if shared == 0:
        return 0.0

    precision = shared / len(prediction_tokens)
    recall = shared / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_predictions(
    predictions: Mapping[str, str],
    references: Mapping[str, str],
) -> dict[str, Any]:
    """Score one prediction per question against single-answer references.

    References define the evaluation population. Predictions with unknown IDs
    are reported but do not alter the denominator. Missing predictions are
    always assigned zero, so provider or parser failures cannot become accidental
    successes through normalization.
    """

    per_item: OrderedDict[str, dict[str, Any]] = OrderedDict()
    missing_ids: list[str] = []
    exact_total = 0
    f1_total = 0.0

    for question_id, gold in references.items():
        if question_id not in predictions:
            missing_ids.append(question_id)
            exact = 0
            f1 = 0.0
            prediction: str | None = None
        else:
            prediction = predictions[question_id]
            exact = compute_exact(gold, prediction)
            f1 = compute_f1(gold, prediction)

        exact_total += exact
        f1_total += f1
        per_item[question_id] = {
            "prediction": prediction,
            "reference": gold,
            "exact": exact,
            "f1": f1,
        }

    total = len(references)
    unexpected_ids = sorted(set(predictions).difference(references))
    return {
        "exact_match": 100.0 * exact_total / total if total else 0.0,
        "f1": 100.0 * f1_total / total if total else 0.0,
        "total": total,
        "missing_count": len(missing_ids),
        "missing_ids": missing_ids,
        "unexpected_count": len(unexpected_ids),
        "unexpected_ids": unexpected_ids,
        "per_item": per_item,
    }


def _unique_submission_predictions(examples: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    predictions: dict[str, str] = {}
    for position, example in enumerate(examples):
        try:
            question_id = example["question_id"]
            prediction = example["pred"]
        except KeyError as error:
            raise ValueError(f"submission item {position} lacks {error.args[0]!r}") from error
        if not isinstance(question_id, str) or not isinstance(prediction, str):
            raise TypeError("submission question_id and pred values must be strings")
        if question_id in predictions:
            raise ValueError(f"duplicate prediction for question_id {question_id!r}")
        predictions[question_id] = prediction
    return predictions


def score_official_submission(
    examples: Iterable[Mapping[str, Any]],
    reference: Mapping[str, Any],
) -> OrderedDict[str, float | int]:
    """Score HybridQA's released submission and reference JSON structures.

    The returned field names and percentage scale match ``evaluate_script.py``.
    Unlike that script, missing predictions are conservatively scored as zero
    instead of causing a late ``KeyError``.
    """

    references = reference.get("reference")
    if not isinstance(references, Mapping):
        raise ValueError("reference JSON must contain a 'reference' mapping")
    predictions = _unique_submission_predictions(examples)
    scored = score_predictions(predictions, references)
    per_item = scored["per_item"]

    def group_average(group_name: str, metric: str) -> float:
        question_ids = reference.get(group_name)
        if not isinstance(question_ids, list):
            raise ValueError(f"reference JSON must contain a {group_name!r} list")
        if not question_ids:
            return 0.0
        unknown = [question_id for question_id in question_ids if question_id not in per_item]
        if unknown:
            raise ValueError(f"{group_name!r} contains unknown question IDs: {unknown[:3]!r}")
        return 100.0 * sum(per_item[question_id][metric] for question_id in question_ids) / len(
            question_ids
        )

    return OrderedDict(
        [
            ("table exact", group_average("table", "exact")),
            ("table f1", group_average("table", "f1")),
            ("passage exact", group_average("passage", "exact")),
            ("passage f1", group_average("passage", "f1")),
            ("total exact", scored["exact_match"]),
            ("total f1", scored["f1"]),
            ("total", scored["total"]),
        ]
    )
