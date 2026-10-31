from collections.abc import Iterable


def apk(actual: Iterable[str], predicted: Iterable[str], k: int = 12) -> float:
    """AP@k; duplicate recommendations never earn additional credit."""
    truth = set(actual)
    if not truth or k <= 0:
        return 0.0
    hits, score, seen = 0, 0.0, set()
    for rank, item in enumerate(list(predicted)[:k], 1):
        if item in truth and item not in seen:
            hits += 1
            score += hits / rank
        seen.add(item)
    return score / min(len(truth), k)

