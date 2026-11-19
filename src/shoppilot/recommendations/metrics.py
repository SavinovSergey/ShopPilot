"""Per-user ranking metrics for recommender evaluation."""
import math

def calculate_user_metrics(
    relevant_items: set[str],
    ranked_items: list[str],
    k: int,
) -> dict[str, float]:
    """Calculate binary-relevance ranking metrics for one user at K.

    Args:
        relevant_items: Nonempty set of distinct target product IDs.
        ranked_items: Product IDs in recommendation order, without duplicates.
        k: Number of leading recommendations to evaluate; must be positive.

    Returns:
        A dictionary with ``recall``, ``hit_rate``, ``ndcg``, and ``mrr`` as
        floats. Only the first ``k`` recommendations affect the metrics.

    Raises:
        ValueError: If the target set is empty, ``k`` is not positive, or the
            recommendation list contains duplicate product IDs.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")
    if len(relevant_items) == 0:
        raise ValueError("relevant_items must not be empty")
    if len(ranked_items) != len(set(ranked_items)):
        raise ValueError("ranked_items must be unique")
    
    top_items = ranked_items[:k]
    scores = [1.0 if item in relevant_items else 0.0 for item in top_items]
    matched_relevant_items = relevant_items.intersection(top_items)
    if len(matched_relevant_items) == 0:
        return {
            "recall": 0.0,
            "hit_rate": 0.0,
            "ndcg": 0.0,
            "mrr": 0.0,
        }
    
    recall = len(matched_relevant_items) / len(relevant_items)
    hitrate = 1.0 if len(matched_relevant_items) > 0 else 0.0
    dcg = sum(score / math.log2(i + 2) for i, score in enumerate(scores))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_items), k)))
    ndcg = dcg / idcg
    i = 1
    mrr = 0.0
    while i <= len(top_items):
        if scores[i - 1] > 0:
            mrr = 1.0 / i
            break
        i += 1
    
    return {
        "recall": recall,
        "hit_rate": hitrate,
        "ndcg": ndcg,
        "mrr": mrr,
    }
