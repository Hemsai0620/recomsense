"""Pydantic response models - the documented contract of the API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RecommendedProductOut(BaseModel):
    """One scored product inside a recommendation list."""

    model_config = ConfigDict(json_schema_extra={"example": {"rank": 1, "product_id": 101, "score": 0.84}})

    rank: int = Field(..., ge=1, description="1-based position in the returned list.")
    product_id: int = Field(..., description="Identifier of the recommended product.")
    score: float = Field(..., description="Model score; higher means a stronger match.")


class RecommendationResponse(BaseModel):
    """Payload returned by the recommendation endpoint."""

    user_id: int = Field(..., description="User the recommendations were produced for.")
    strategy: str = Field(
        ...,
        description="'collaborative_filtering' for known users, 'popularity' for the cold-start fallback.",
    )
    cold_start: bool = Field(
        ..., description="True when the popularity fallback was used instead of the ALS model."
    )
    requested_top_n: int = Field(..., ge=1, description="Top-N value applied to this request.")
    count: int = Field(..., ge=0, description="Number of products actually returned.")
    recommendations: list[RecommendedProductOut] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Service and model readiness report."""

    status: str = Field(..., description="'ok' when the model bundle is loaded.")
    model_loaded: bool
    trained_at: str | None = None
    known_users: int | None = None
    known_products: int | None = None
    detail: str | None = None


class ErrorResponse(BaseModel):
    """Uniform error body for every failure the API reports."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": {
                    "code": "invalid_top_n",
                    "message": "top_n must be between 1 and 50, got 0.",
                }
            }
        }
    )

    error: dict
