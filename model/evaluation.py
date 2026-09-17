"""Offline ranking metrics for the recommendation lists."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def precision_at_k(actual: Iterable[int], predicted: Sequence[int], k: int = 5) -> float:
    """Share of the top ``k`` predictions that the user actually interacted with."""
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    relevant = set(actual)
    top_k = list(predicted)[:k]
    if not top_k:
        return 0.0
    return len([item for item in top_k if item in relevant]) / k


def recall_at_k(actual: Iterable[int], predicted: Sequence[int], k: int = 5) -> float:
    """Share of the user's relevant products that appear in the top ``k``."""
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    relevant = set(actual)
    if not relevant:
        return 0.0
    top_k = set(list(predicted)[:k])
    return len(relevant & top_k) / len(relevant)


def evaluate_rankings(
    holdout: dict[int, Iterable[int]],
    predictions: dict[int, Sequence[int]],
    k: int = 5,
) -> dict[str, float]:
    """Average precision@k and recall@k over every user in ``holdout``."""
    if not holdout:
        return {"users": 0, "precision_at_k": 0.0, "recall_at_k": 0.0, "k": k}

    precisions = []
    recalls = []
    for user_id, relevant in holdout.items():
        predicted = predictions.get(user_id, [])
        precisions.append(precision_at_k(relevant, predicted, k))
        recalls.append(recall_at_k(relevant, predicted, k))

    return {
        "users": len(holdout),
        "precision_at_k": sum(precisions) / len(precisions),
        "recall_at_k": sum(recalls) / len(recalls),
        "k": k,
    }
