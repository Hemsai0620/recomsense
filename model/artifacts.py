"""Persist and reload everything inference needs as one bundle.

ALS scoring needs the user's own interaction row at prediction time, so the
trained factors alone are not enough. Saving the model, the matrix, the id
mappings and the popularity ranking together keeps training and serving from
drifting apart.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scipy.sparse import csr_matrix

from model.matrix import IndexMapping
from utils.errors import ArtifactNotFoundError
from utils.logging import get_logger

logger = get_logger(__name__)

#: Bumped whenever the bundle layout changes, so stale files fail loudly.
ARTIFACT_FORMAT_VERSION = 2


@dataclass
class ModelBundle:
    """Everything required to serve recommendations."""

    model: Any
    interaction_matrix: csr_matrix
    mapping: IndexMapping
    popular_products: list[tuple[int, float]]
    trained_at: str = ""
    format_version: int = ARTIFACT_FORMAT_VERSION
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.trained_at:
            self.trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_bundle(bundle: ModelBundle, path: str | Path) -> Path:
    """Write ``bundle`` to ``path``, creating the parent directory."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved model bundle to %s", target)
    return target


def load_bundle(path: str | Path) -> ModelBundle:
    """Read a bundle previously written by :func:`save_bundle`.

    Raises:
        ArtifactNotFoundError: if the file is missing, unreadable, or was
            produced by an incompatible bundle layout.
    """
    source = Path(path)
    if not source.exists():
        raise ArtifactNotFoundError(
            f"No trained model at {source}. Run 'python -m model.trainer' first."
        )

    try:
        with source.open("rb") as handle:
            bundle = pickle.load(handle)
    except (pickle.UnpicklingError, EOFError, AttributeError, ModuleNotFoundError) as exc:
        raise ArtifactNotFoundError(f"Model bundle at {source} could not be read: {exc}") from exc

    if not isinstance(bundle, ModelBundle):
        raise ArtifactNotFoundError(f"Model bundle at {source} has an unexpected type.")

    if bundle.format_version != ARTIFACT_FORMAT_VERSION:
        raise ArtifactNotFoundError(
            f"Model bundle at {source} uses format v{bundle.format_version}, "
            f"but v{ARTIFACT_FORMAT_VERSION} is required. Retrain the model."
        )

    logger.info("Loaded model bundle trained at %s from %s", bundle.trained_at, source)
    return bundle
