"""Query-level metrics for ranked, potentially multi-positive document IDs."""

from math import log2
from typing import Sequence


CUTOFFS = (1, 5, 10)
METRIC_NAMES = tuple(f"{name}{k}" for name in
                     ("hit", "mrr", "recall", "precision", "ndcg", "relevant_count") for k in CUTOFFS)


def score_ranking(ranked: Sequence[str], positives: Sequence[str]) -> dict[str, float]:
    relevant = set(positives)
    if not relevant:
        raise ValueError("Cannot score a query with no positive URLs")
    unique = list(dict.fromkeys(ranked))
    first = next((i + 1 for i, target in enumerate(unique) if target in relevant), None)
    result = {}
    for k in CUTOFFS:
        labels = [float(target in relevant) for target in unique[:k]]
        count = sum(labels)
        dcg = sum(label / log2(i + 2) for i, label in enumerate(labels))
        ideal = sum(1 / log2(i + 2) for i in range(min(k, len(relevant))))
        result.update({f"hit{k}": float(first is not None and first <= k),
                       f"mrr{k}": 1 / first if first is not None and first <= k else 0.,
                       f"recall{k}": count / len(relevant), f"precision{k}": count / k,
                       f"ndcg{k}": dcg / ideal, f"relevant_count{k}": count})
    return result
