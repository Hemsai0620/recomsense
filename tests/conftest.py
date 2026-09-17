"""Shared fixtures: one small model trained once per test session.

Training happens against a temporary artifact path so running the tests never
overwrites the artifact produced by ``python -m model.trainer``.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.app import create_app  # noqa: E402
from api.dependencies import get_engine, get_engine_or_none  # noqa: E402
from model.recommender import RecommendationEngine  # noqa: E402
from model.trainer import train  # noqa: E402
from utils.config import Settings  # noqa: E402

#: A user that appears in the sample interaction dataset.
KNOWN_USER_ID = 1
#: A user that does not appear anywhere in the dataset.
UNKNOWN_USER_ID = 999_999


@pytest.fixture(scope="session")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    artifact_dir = tmp_path_factory.mktemp("artifacts")
    return replace(
        Settings(),
        artifact_path=artifact_dir / "recommender.pkl",
        factors=8,
        iterations=5,
    )


@pytest.fixture(scope="session")
def engine(settings: Settings) -> RecommendationEngine:
    bundle = train(settings)
    return RecommendationEngine(bundle, settings)


@pytest.fixture()
def client(engine: RecommendationEngine):
    """A TestClient whose engine dependency is the session-trained engine."""
    from fastapi.testclient import TestClient

    app = create_app()
    app.dependency_overrides[get_engine] = lambda: engine
    app.dependency_overrides[get_engine_or_none] = lambda: engine
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
