"""Read the raw user-product interaction log from disk.

The loader is deliberately strict: a malformed CSV should fail during training
with a clear message rather than silently producing an empty model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.config import get_settings
from utils.errors import DataValidationError
from utils.logging import get_logger

logger = get_logger(__name__)

#: Columns the pipeline cannot work without. ``timestamp`` is optional.
REQUIRED_COLUMNS: tuple[str, ...] = ("user_id", "product_id", "event")


def load_interactions(path: str | Path | None = None) -> pd.DataFrame:
    """Load the interaction log as a :class:`pandas.DataFrame`.

    Args:
        path: CSV location. Defaults to the configured interactions path.

    Raises:
        DataValidationError: if the file is missing, empty, or lacks a
            required column.
    """
    csv_path = Path(path) if path is not None else get_settings().interactions_path

    if not csv_path.exists():
        raise DataValidationError(f"Interaction file not found: {csv_path}")

    try:
        frame = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"Interaction file is empty: {csv_path}") from exc
    except pd.errors.ParserError as exc:
        raise DataValidationError(f"Interaction file is not valid CSV: {csv_path}") from exc

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DataValidationError(
            f"Interaction file {csv_path} is missing required column(s): {', '.join(missing)}"
        )

    if frame.empty:
        raise DataValidationError(f"Interaction file contains no rows: {csv_path}")

    logger.info("Loaded %d interactions from %s", len(frame), csv_path)
    return frame
