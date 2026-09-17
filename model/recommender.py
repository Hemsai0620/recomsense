"""Serving-time recommendation logic, including the cold-start fallback.

Two strategies are exposed behind one call:

* ``collaborative_filtering`` - the trained ALS model scores candidate
  products using the user's own interaction history.
* ``popularity`` - used when the user is unknown or has no usable history;
  the globally most-interacted products are returned instead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from model.artifacts import ModelBundle, load_bundle
from utils.config import Settings, get_settings
from utils.errors import InvalidTopNError, UnknownUserError
from utils.logging import get_logger

logger = get_logger(__name__)

STRATEGY_COLLABORATIVE = "collaborative_filtering"
STRATEGY_POPULARITY = "popularity"


@dataclass(frozen=True)
class RecommendedProduct:
    """A single scored product in a recommendation list."""

    rank: int
    product_id: int
    score: float


@dataclass(frozen=True)
class RecommendationResult:
    """Outcome of one recommendation request."""

    user_id: int
    strategy: str
    cold_start: bool
    requested_top_n: int
    recommendations: list[RecommendedProduct]

    def as_dict(self) -> dict:
        return asdict(self)


class RecommendationEngine:
    """Answers recommendation queries from a trained :class:`ModelBundle`."""

    def __init__(self, bundle: ModelBundle, settings: Settings | None = None) -> None:
        self._bundle = bundle
        self._settings = settings or get_settings()

    @classmethod
    def from_artifact(
        cls, path: str | Path | None = None, settings: Settings | None = None
    ) -> "RecommendationEngine":
        """Build an engine by loading the bundle written by the trainer."""
        resolved = settings or get_settings()
        return cls(load_bundle(path or resolved.artifact_path), resolved)

    # -- introspection ---------------------------------------------------

    @property
    def bundle(self) -> ModelBundle:
        return self._bundle

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def known_user_count(self) -> int:
        return self._bundle.mapping.n_users

    @property
    def known_product_count(self) -> int:
        return self._bundle.mapping.n_products

    def is_known_user(self, user_id: int) -> bool:
        """True when the user appeared in the training interactions."""
        return int(user_id) in self._bundle.mapping.user_to_index

    # -- validation ------------------------------------------------------

    def validate_top_n(self, top_n: int | None) -> int:
        """Normalise and bounds-check the requested list length."""
        if top_n is None:
            return self._settings.default_top_n

        if isinstance(top_n, bool) or not isinstance(top_n, (int, np.integer)):
            raise InvalidTopNError(f"top_n must be an integer, got {top_n!r}.")

        value = int(top_n)
        if value < 1 or value > self._settings.max_top_n:
            raise InvalidTopNError(
                f"top_n must be between 1 and {self._settings.max_top_n}, got {value}."
            )
        return value

    # -- recommendation --------------------------------------------------

    def recommend(
        self,
        user_id: int,
        top_n: int | None = None,
        *,
        allow_cold_start: bool = True,
    ) -> RecommendationResult:
        """Return up to ``top_n`` products for ``user_id``.

        Args:
            user_id: Identifier from the interaction dataset.
            top_n: How many products to return. Defaults to the configured
                value and must fall between 1 and ``max_top_n``.
            allow_cold_start: When False an unknown user raises instead of
                falling back to popular products.

        Raises:
            InvalidTopNError: if ``top_n`` is not a valid list length.
            UnknownUserError: if the user is unknown and ``allow_cold_start``
                is False.
        """
        size = self.validate_top_n(top_n)
        user_key = int(user_id)
        user_index = self._bundle.mapping.user_to_index.get(user_key)

        if user_index is None:
            if not allow_cold_start:
                raise UnknownUserError(f"User {user_key} has no interaction history.")
            logger.info("Cold start for unknown user %s; serving popular products", user_key)
            return self._popular_result(user_key, size, requested=size)

        history = self._bundle.interaction_matrix[user_index]
        if history.nnz == 0:
            logger.info("Known user %s has an empty history; serving popular products", user_key)
            return self._popular_result(user_key, size, requested=size)

        scored = self._collaborative_products(user_index, history, size)
        if not scored:
            logger.info("Model returned nothing for user %s; serving popular products", user_key)
            return self._popular_result(user_key, size, requested=size)

        return RecommendationResult(
            user_id=user_key,
            strategy=STRATEGY_COLLABORATIVE,
            cold_start=False,
            requested_top_n=size,
            recommendations=scored,
        )

    def popular_products(self, top_n: int | None = None) -> list[RecommendedProduct]:
        """Expose the popularity ranking on its own."""
        size = self.validate_top_n(top_n)
        return [
            RecommendedProduct(rank=position, product_id=product_id, score=round(float(score), 6))
            for position, (product_id, score) in enumerate(
                self._bundle.popular_products[:size], start=1
            )
        ]

    # -- internals -------------------------------------------------------

    def _popular_result(self, user_id: int, size: int, requested: int) -> RecommendationResult:
        return RecommendationResult(
            user_id=user_id,
            strategy=STRATEGY_POPULARITY,
            cold_start=True,
            requested_top_n=requested,
            recommendations=self.popular_products(size),
        )

    def _collaborative_products(self, user_index, history, size: int) -> list[RecommendedProduct]:
        """Call ALS and normalise its output across implicit versions."""
        limit = min(size, self.known_product_count)
        if limit < 1:
            return []

        raw = self._bundle.model.recommend(user_index, history, N=limit)
        product_indices, scores = _unpack_recommendations(raw)

        products: list[RecommendedProduct] = []
        for position, (index, score) in enumerate(zip(product_indices, scores), start=1):
            if not 0 <= int(index) < self.known_product_count:
                continue
            products.append(
                RecommendedProduct(
                    rank=position,
                    product_id=int(self._bundle.mapping.index_to_product[int(index)]),
                    score=round(float(score), 6),
                )
            )
        return products


def _unpack_recommendations(raw) -> tuple[list, list]:
    """Accept both the ``(ids, scores)`` and ``[(id, score), ...]`` shapes."""
    if isinstance(raw, tuple) and len(raw) == 2:
        ids, scores = raw
        return list(np.atleast_1d(ids)), list(np.atleast_1d(scores))
    return [item[0] for item in raw], [item[1] for item in raw]
