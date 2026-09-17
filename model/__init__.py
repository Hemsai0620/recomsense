"""Feature engineering, ALS training and recommendation serving."""

from model.artifacts import ModelBundle, load_bundle, save_bundle
from model.features import EVENT_WEIGHTS, build_ratings
from model.recommender import (
    STRATEGY_COLLABORATIVE,
    STRATEGY_POPULARITY,
    RecommendationEngine,
    RecommendationResult,
    RecommendedProduct,
)

__all__ = [
    "ModelBundle",
    "load_bundle",
    "save_bundle",
    "EVENT_WEIGHTS",
    "build_ratings",
    "RecommendationEngine",
    "RecommendationResult",
    "RecommendedProduct",
    "STRATEGY_COLLABORATIVE",
    "STRATEGY_POPULARITY",
]
