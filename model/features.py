"""Turn raw events into the implicit-feedback ratings ALS consumes.

Each event type carries a different amount of purchase intent, so events are
mapped to a numeric weight and all weights for the same (user, product) pair
are summed into a single confidence score.
"""

from __future__ import annotations

import pandas as pd

from utils.errors import DataValidationError
from utils.logging import get_logger

logger = get_logger(__name__)

#: Relative strength of each supported interaction type.
EVENT_WEIGHTS: dict[str, float] = {
    "view": 1.0,
    "click": 2.0,
    "purchase": 3.0,
}

RATING_COLUMNS = ["user_id", "product_id", "rating"]


def build_ratings(frame: pd.DataFrame) -> pd.DataFrame:
    """Map events to weights and aggregate them per user-product pair.

    Rows with an unrecognised event type are dropped instead of becoming
    ``NaN`` ratings, which previously poisoned the interaction matrix.

    Returns:
        A frame with ``user_id``, ``product_id`` and a summed ``rating``.

    Raises:
        DataValidationError: if nothing survives the mapping.
    """
    ratings = frame.loc[:, ["user_id", "product_id", "event"]].copy()

    normalised_events = ratings["event"].astype(str).str.strip().str.lower()
    ratings["rating"] = normalised_events.map(EVENT_WEIGHTS)

    unknown = ratings["rating"].isna().sum()
    if unknown:
        logger.warning(
            "Dropped %d interaction(s) with an unsupported event type (known: %s)",
            unknown,
            ", ".join(sorted(EVENT_WEIGHTS)),
        )
        ratings = ratings.dropna(subset=["rating"])

    ratings = ratings.dropna(subset=["user_id", "product_id"])
    if ratings.empty:
        raise DataValidationError(
            "No usable interactions after feature engineering; "
            f"supported event types are: {', '.join(sorted(EVENT_WEIGHTS))}"
        )

    aggregated = (
        ratings.groupby(["user_id", "product_id"], as_index=False)["rating"]
        .sum()
        .astype({"rating": "float32"})
    )

    logger.info(
        "Built %d user-product ratings from %d raw interactions",
        len(aggregated),
        len(frame),
    )
    return aggregated.loc[:, RATING_COLUMNS]
