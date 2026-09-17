"""Data loading, feature engineering, matrix building and ranking metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from data.loader import load_interactions
from model.evaluation import evaluate_rankings, precision_at_k, recall_at_k
from model.features import EVENT_WEIGHTS, build_ratings
from model.matrix import (
    build_index_mapping,
    build_interaction_matrix,
    rank_products_by_popularity,
)
from utils.errors import DataValidationError


class TestLoader:
    def test_sample_dataset_loads_with_the_expected_columns(self) -> None:
        frame = load_interactions()

        assert not frame.empty
        assert {"user_id", "product_id", "event"} <= set(frame.columns)

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(DataValidationError):
            load_interactions(tmp_path / "does-not-exist.csv")

    def test_missing_column_raises(self, tmp_path) -> None:
        path = tmp_path / "bad.csv"
        path.write_text("user_id,product_id\n1,101\n", encoding="utf-8")

        with pytest.raises(DataValidationError):
            load_interactions(path)


class TestFeatures:
    def test_events_map_to_their_weights(self) -> None:
        frame = pd.DataFrame(
            {
                "user_id": [1, 2, 3],
                "product_id": [10, 11, 12],
                "event": ["view", "click", "purchase"],
            }
        )

        ratings = build_ratings(frame)

        assert sorted(ratings["rating"].tolist()) == sorted(EVENT_WEIGHTS.values())

    def test_repeated_pairs_are_summed(self) -> None:
        frame = pd.DataFrame(
            {"user_id": [1, 1], "product_id": [10, 10], "event": ["view", "purchase"]}
        )

        ratings = build_ratings(frame)

        assert len(ratings) == 1
        assert ratings["rating"].iloc[0] == pytest.approx(
            EVENT_WEIGHTS["view"] + EVENT_WEIGHTS["purchase"]
        )

    def test_unsupported_events_are_dropped_not_turned_into_nan(self) -> None:
        frame = pd.DataFrame(
            {"user_id": [1, 2], "product_id": [10, 11], "event": ["view", "teleported"]}
        )

        ratings = build_ratings(frame)

        assert len(ratings) == 1
        assert ratings["rating"].isna().sum() == 0

    def test_all_events_unsupported_raises(self) -> None:
        frame = pd.DataFrame({"user_id": [1], "product_id": [10], "event": ["teleported"]})

        with pytest.raises(DataValidationError):
            build_ratings(frame)


class TestMatrix:
    def test_matrix_shape_follows_the_index_mapping(self) -> None:
        ratings = build_ratings(load_interactions())
        mapping = build_index_mapping(ratings)
        matrix = build_interaction_matrix(ratings, mapping)

        assert matrix.shape == (mapping.n_users, mapping.n_products)
        assert matrix.nnz == len(ratings)

    def test_mapping_round_trips_ids(self) -> None:
        ratings = build_ratings(load_interactions())
        mapping = build_index_mapping(ratings)

        for user_id, index in mapping.user_to_index.items():
            assert mapping.index_to_user[index] == user_id

    def test_popularity_is_sorted_and_limited(self) -> None:
        ratings = build_ratings(load_interactions())
        popular = rank_products_by_popularity(ratings, limit=3)

        assert len(popular) == 3
        assert [score for _, score in popular] == sorted(
            (score for _, score in popular), reverse=True
        )


class TestMetrics:
    def test_precision_at_k(self) -> None:
        assert precision_at_k([1, 2, 3], [1, 9, 2, 8], k=4) == pytest.approx(0.5)

    def test_recall_at_k(self) -> None:
        assert recall_at_k([1, 2], [1, 9, 8], k=3) == pytest.approx(0.5)

    def test_metrics_reject_non_positive_k(self) -> None:
        with pytest.raises(ValueError):
            precision_at_k([1], [1], k=0)
        with pytest.raises(ValueError):
            recall_at_k([1], [1], k=0)

    def test_evaluate_rankings_averages_over_users(self) -> None:
        report = evaluate_rankings(
            holdout={1: [10, 11], 2: [20]},
            predictions={1: [10, 99], 2: [98, 97]},
            k=2,
        )

        assert report["users"] == 2
        assert report["precision_at_k"] == pytest.approx(0.25)
        assert report["recall_at_k"] == pytest.approx(0.25)

    def test_empty_holdout_is_handled(self) -> None:
        assert evaluate_rankings({}, {}, k=5)["users"] == 0
