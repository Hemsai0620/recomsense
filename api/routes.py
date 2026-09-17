"""HTTP routes for RecomSense."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Path, Query

from api.dependencies import get_engine, get_engine_or_none
from api.schemas import (
    ErrorResponse,
    HealthResponse,
    RecommendationResponse,
    RecommendedProductOut,
)
from model.recommender import RecommendationEngine
from utils.config import get_settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["service"], summary="Service readiness")
def health(
    engine: RecommendationEngine | None = Depends(get_engine_or_none),
) -> HealthResponse:
    """Report whether the trained model bundle is loaded and servable."""
    if engine is None:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            detail="No trained model bundle available. Run 'python -m model.trainer'.",
        )

    return HealthResponse(
        status="ok",
        model_loaded=True,
        trained_at=engine.bundle.trained_at,
        known_users=engine.known_user_count,
        known_products=engine.known_product_count,
    )


@router.get(
    "/recommendations/{user_id}",
    response_model=RecommendationResponse,
    tags=["recommendations"],
    summary="Top-N product recommendations for a user",
    responses={
        404: {"model": ErrorResponse, "description": "Unknown user with cold start disabled."},
        422: {"model": ErrorResponse, "description": "Invalid user_id or top_n."},
        503: {"model": ErrorResponse, "description": "Model bundle not trained yet."},
    },
)
def get_recommendations(
    user_id: int = Path(..., ge=0, description="User identifier from the interaction dataset."),
    top_n: int = Query(
        default=None,
        ge=1,
        le=get_settings().max_top_n,
        description="How many products to return. Defaults to the configured Top-N.",
    ),
    allow_cold_start: bool = Query(
        default=True,
        description="When false, an unknown user returns 404 instead of popular products.",
    ),
    engine: RecommendationEngine = Depends(get_engine),
) -> RecommendationResponse:
    """Return ranked product recommendations for ``user_id``.

    Known users are scored by the trained ALS model. Users with no interaction
    history fall back to the most popular products, and the response says which
    strategy produced the list.
    """
    result = engine.recommend(user_id, top_n, allow_cold_start=allow_cold_start)

    return RecommendationResponse(
        user_id=result.user_id,
        strategy=result.strategy,
        cold_start=result.cold_start,
        requested_top_n=result.requested_top_n,
        count=len(result.recommendations),
        recommendations=[
            RecommendedProductOut(rank=item.rank, product_id=item.product_id, score=item.score)
            for item in result.recommendations
        ],
    )


@router.get(
    "/products/popular",
    response_model=list[RecommendedProductOut],
    tags=["recommendations"],
    summary="Most popular products",
    responses={
        422: {"model": ErrorResponse, "description": "Invalid top_n."},
        503: {"model": ErrorResponse, "description": "Model bundle not trained yet."},
    },
)
def popular_products(
    top_n: int = Query(default=None, ge=1, le=get_settings().max_top_n),
    engine: RecommendationEngine = Depends(get_engine),
) -> list[RecommendedProductOut]:
    """Expose the ranking that backs the cold-start fallback."""
    return [
        RecommendedProductOut(rank=item.rank, product_id=item.product_id, score=item.score)
        for item in engine.popular_products(top_n)
    ]
