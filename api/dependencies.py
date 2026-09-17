"""FastAPI dependency wiring for the shared recommendation engine."""

from __future__ import annotations

from functools import lru_cache

from model.recommender import RecommendationEngine
from utils.config import get_settings
from utils.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_engine() -> RecommendationEngine:
    """Load the trained bundle once and reuse it for every request.

    Raises:
        ArtifactNotFoundError: if the model has not been trained yet. The
            application turns this into a 503 response.
    """
    settings = get_settings()
    logger.info("Loading recommendation engine from %s", settings.artifact_path)
    return RecommendationEngine.from_artifact(settings.artifact_path, settings)


def reset_engine_cache() -> None:
    """Drop the cached engine, e.g. after retraining or inside tests."""
    get_engine.cache_clear()


def get_engine_or_none() -> RecommendationEngine | None:
    """Like :func:`get_engine` but returns ``None`` instead of raising.

    Used by ``/health``, which must stay reachable and answer truthfully even
    when the model has not been trained yet.
    """
    from utils.errors import ArtifactNotFoundError

    try:
        return get_engine()
    except ArtifactNotFoundError as exc:
        logger.warning("Recommendation engine unavailable: %s", exc.message)
        return None
