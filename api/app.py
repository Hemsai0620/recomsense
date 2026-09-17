"""RecomSense FastAPI application.

The app is built by a factory so tests can create isolated instances, and the
model bundle is loaded lazily: the service still starts (and reports its state
on ``/health``) when the model has not been trained yet.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.routes import router
from utils.errors import RecomSenseError
from utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

API_TITLE = "RecomSense API"
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "Personalized product recommendations from user-product interaction history.\n\n"
    "Known users are served by an Alternating Least Squares collaborative "
    "filtering model. Users without usable history fall back to the most "
    "popular products, and every response reports which strategy was used."
)


def _error_body(code: str, message: str, **extra) -> dict:
    return {"error": {"code": code, "message": message, **extra}}


def create_app() -> FastAPI:
    """Build and configure the application."""
    configure_logging()

    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        contact={"name": "RecomSense"},
    )

    @app.exception_handler(RecomSenseError)
    def handle_domain_error(_: Request, exc: RecomSenseError) -> JSONResponse:
        """Map every domain error onto its declared HTTP status code."""
        logger.warning("%s: %s", exc.code, exc.message)
        return JSONResponse(
            status_code=exc.http_status,
            content=_error_body(exc.code, exc.message),
        )

    @app.exception_handler(RequestValidationError)
    def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        """Return FastAPI's parameter validation failures in the same shape."""
        details = [
            {
                "field": ".".join(str(part) for part in error.get("loc", []) if part != "query"),
                "message": error.get("msg", "invalid value"),
            }
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content=_error_body(
                "invalid_request",
                "One or more request parameters are invalid.",
                details=details,
            ),
        )

    @app.get("/", tags=["service"], summary="Service metadata")
    def index() -> dict:
        return {
            "service": API_TITLE,
            "version": API_VERSION,
            "docs": "/docs",
            "endpoints": ["/health", "/recommendations/{user_id}", "/products/popular"],
        }

    app.include_router(router)
    return app


app = create_app()
