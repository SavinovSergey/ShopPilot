import math

import pytest

from shoppilot.recommendations.metrics import calculate_user_metrics


def test_calculate_user_metrics_for_partial_hit_at_second_rank():
    metrics = calculate_user_metrics(
        relevant_items={"B", "C"},
        ranked_items=["D", "B", "C"],
        k=2,
    )

    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["hit_rate"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(0.5)
    assert metrics["ndcg"] == pytest.approx(1 / (1 + math.log2(3)))


def test_calculate_user_metrics_for_perfect_ranking():
    metrics = calculate_user_metrics(
        relevant_items={"B", "C"},
        ranked_items=["B", "C", "D"],
        k=2,
    )

    assert metrics == {
        "recall": 1.0,
        "hit_rate": 1.0,
        "ndcg": 1.0,
        "mrr": 1.0,
    }


def test_ndcg_uses_relevant_count_when_ranking_contains_extra_items():
    metrics = calculate_user_metrics(
        relevant_items={"B"},
        ranked_items=["B", "D"],
        k=2,
    )

    assert metrics["ndcg"] == pytest.approx(1.0)


def test_ndcg_uses_full_ideal_ranking_when_fewer_than_k_items_are_returned():
    metrics = calculate_user_metrics(
        relevant_items={"B", "C"},
        ranked_items=["B"],
        k=2,
    )

    assert metrics["ndcg"] == pytest.approx(1 / (1 + 1 / math.log2(3)))


def test_calculate_user_metrics_for_no_hit_in_top_k():
    metrics = calculate_user_metrics(
        relevant_items={"B", "C"},
        ranked_items=["D", "E", "B"],
        k=2,
    )

    assert metrics == {
        "recall": 0.0,
        "hit_rate": 0.0,
        "ndcg": 0.0,
        "mrr": 0.0,
    }


@pytest.mark.parametrize(
    ("relevant_items", "ranked_items", "k"),
    [
        (set(), ["B"], 2),
        ({"B"}, ["B"], 0),
        ({"B", "C"}, ["B", "B"], 2),
    ],
)
def test_calculate_user_metrics_rejects_invalid_inputs(
    relevant_items, ranked_items, k
):
    with pytest.raises(ValueError):
        calculate_user_metrics(relevant_items, ranked_items, k)
