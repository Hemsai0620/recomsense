"""Runtime configuration for RecomSense.

All values have working defaults, so the project runs with zero setup, and
every value can be overridden through a ``RECOMSENSE_*`` environment variable
for Docker or CI without editing code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INTERACTIONS_PATH = PROJECT_ROOT / "data" / "interactions.csv"
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "recommender.pkl"


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser().resolve() if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - configuration typo
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover - configuration typo
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the project configuration."""

    interactions_path: Path = DEFAULT_INTERACTIONS_PATH
    artifact_path: Path = DEFAULT_ARTIFACT_PATH
    factors: int = 50
    iterations: int = 20
    regularization: float = 0.05
    random_state: int = 42
    default_top_n: int = 5
    max_top_n: int = 50
    popular_pool_size: int = 100

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            interactions_path=_env_path("RECOMSENSE_INTERACTIONS_PATH", DEFAULT_INTERACTIONS_PATH),
            artifact_path=_env_path("RECOMSENSE_ARTIFACT_PATH", DEFAULT_ARTIFACT_PATH),
            factors=_env_int("RECOMSENSE_FACTORS", 50),
            iterations=_env_int("RECOMSENSE_ITERATIONS", 20),
            regularization=_env_float("RECOMSENSE_REGULARIZATION", 0.05),
            random_state=_env_int("RECOMSENSE_RANDOM_STATE", 42),
            default_top_n=_env_int("RECOMSENSE_DEFAULT_TOP_N", 5),
            max_top_n=_env_int("RECOMSENSE_MAX_TOP_N", 50),
            popular_pool_size=_env_int("RECOMSENSE_POPULAR_POOL_SIZE", 100),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, read from the environment once."""
    return Settings.from_env()
