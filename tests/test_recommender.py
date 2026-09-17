"""Engine-level behaviour: known users, cold start, and Top-N validation."""

from __future__ import annotations

import pytest

from model.recommender import STRATEGY_COLLABORATIVE, STRATEGY_POPULARITY, RecommendationEngine
from tests.conftest import KNOWN_USER_ID, UNKNOWN_USER_ID
from utils.errors import InvalidTopNError, UnknownUserError


class TestKnownUser:
    def test_known_user_uses_the_model(self, engine: RecommendationEngine) -> None:
        result = engine.recommend(KNOWN_USER_ID, top_n=3)

        assert engine.is_known_user(KNOWN_USER_ID)
        assert result.user_id == KNOWN_USER_ID
        assert result.strategy == STRATEGY_COLLABORATIVE
        assert result.cold_start is False
        assert len(result.recommendations) == 3

    def test_ranks_are_sequential_and_products_unique(self, engine: RecommendationEngine) -> None:
        result = engine.recommend(KNOWN_USER_ID, top_n=4)

        assert [item.rank for item in result.recommendations] == [1, 2, 3, 4]
        product_ids = [item.product_id for item in result.recommendations]
        assert len(set(product_ids)) == len(product_ids)

    def test_recommended_products_are_known_products(self, engine: RecommendationEngine) -> None:
        result = engine.recommend(KNOWN_USER_ID, top_n=5)
        catalogue = set(engine.bundle.mapping.index_to_product)

        assert {item.product_id for item in result.recommendations} <= catalogue

    def test_top_n_larger_than_catalogue_is_clamped(self, engine: RecommendationEngine) -> None:
        result = engine.recommend(KNOWN_USER_ID, top_n=engine.known_product_count + 20)

        assert len(result.recommendations) <= engine.known_product_count

    def test_default_top_n_is_used_when_omitted(self, engine: RecommendationEngine) -> None:
        result = engine.recommend(KNOWN_USER_ID)

        assert result.requested_top_n == engine.settings.default_top_n


class TestColdStart:
    def test_unknown_user_falls_back_to_popular_products(
        self, engine: RecommendationEngine
    ) -> None:
        result = engine.recommend(UNKNOWN_USER_ID, top_n=3)

        assert engine.is_known_user(UNKNOWN_USER_ID) is False
        assert result.strategy == STRATEGY_POPULARITY
        assert result.cold_start is True
        assert len(result.recommendations) == 3

    def test_fallback_matches_the_popularity_ranking(self, engine: RecommendationEngine) -> None:
        fallback = engine.recommend(UNKNOWN_USER_ID, top_n=3).recommendations
        popular = engine.popular_products(top_n=3)

        assert [item.product_id for item in fallback] == [item.product_id for item in popular]

    def test_popularity_scores_are_descending(self, engine: RecommendationEngine) -> None:
        scores = [item.score for item in engine.popular_products(top_n=5)]

        assert scores == sorted(scores, reverse=True)

    def test_unknown_user_raises_when_cold_start_disabled(
        self, engine: RecommendationEngine
    ) -> None:
        with pytest.raises(UnknownUserError):
            engine.recommend(UNKNOWN_USER_ID, top_n=3, allow_cold_start=False)

    def test_known_user_unaffected_by_cold_start_flag(self, engine: RecommendationEngine) -> None:
        result = engine.recommend(KNOWN_USER_ID, top_n=3, allow_cold_start=False)

        assert result.strategy == STRATEGY_COLLABORATIVE


class TestTopNValidation:
    @pytest.mark.parametrize("top_n", [0, -1, -100])
    def test_non_positive_top_n_is_rejected(
        self, engine: RecommendationEngine, top_n: int
    ) -> None:
        with pytest.raises(InvalidTopNError):
            engine.recommend(KNOWN_USER_ID, top_n=top_n)

    def test_top_n_above_the_maximum_is_rejected(self, engine: RecommendationEngine) -> None:
        with pytest.raises(InvalidTopNError):
            engine.recommend(KNOWN_USER_ID, top_n=engine.settings.max_top_n + 1)

    @pytest.mark.parametrize("top_n", ["5", 2.5, object()])
    def test_non_integer_top_n_is_rejected(self, engine: RecommendationEngine, top_n) -> None:
        with pytest.raises(InvalidTopNError):
            engine.recommend(KNOWN_USER_ID, top_n=top_n)

    def test_invalid_top_n_is_rejected_for_unknown_users_too(
        self, engine: RecommendationEngine
    ) -> None:
        with pytest.raises(InvalidTopNError):
            engine.recommend(UNKNOWN_USER_ID, top_n=0)
