"""Training pipeline: CSV -> ratings -> sparse matrix -> ALS -> artifact.

Run it with ``python -m model.trainer``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from implicit.als import AlternatingLeastSquares
from threadpoolctl import threadpool_limits

from data.loader import load_interactions
from model.artifacts import ModelBundle, save_bundle
from model.features import build_ratings
from model.matrix import (
    build_index_mapping,
    build_interaction_matrix,
    rank_products_by_popularity,
)
from utils.config import Settings, get_settings
from utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def train(settings: Settings | None = None) -> ModelBundle:
    """Train the ALS model and persist the serving bundle."""
    config = settings or get_settings()

    interactions = load_interactions(config.interactions_path)
    ratings = build_ratings(interactions)

    mapping = build_index_mapping(ratings)
    interaction_matrix = build_interaction_matrix(ratings, mapping)
    logger.info(
        "Interaction matrix: %d users x %d products, %d non-zero cells",
        mapping.n_users,
        mapping.n_products,
        interaction_matrix.nnz,
    )

    # ALS cannot learn more latent factors than the smaller matrix dimension.
    factors = max(1, min(config.factors, mapping.n_users, mapping.n_products))
    if factors != config.factors:
        logger.warning(
            "Reduced factors from %d to %d to fit a %dx%d matrix",
            config.factors,
            factors,
            mapping.n_users,
            mapping.n_products,
        )

    model = AlternatingLeastSquares(
        factors=factors,
        iterations=config.iterations,
        regularization=config.regularization,
        random_state=config.random_state,
    )
    # implicit's ALS is already parallel; letting BLAS spawn its own pool on top
    # of that degrades performance and makes the library emit a warning.
    with threadpool_limits(limits=1, user_api="blas"):
        model.fit(interaction_matrix, show_progress=False)

    bundle = ModelBundle(
        model=model,
        interaction_matrix=interaction_matrix,
        mapping=mapping,
        popular_products=rank_products_by_popularity(ratings, config.popular_pool_size),
        params={
            "factors": factors,
            "iterations": config.iterations,
            "regularization": config.regularization,
            "random_state": config.random_state,
            "n_users": mapping.n_users,
            "n_products": mapping.n_products,
            "n_interactions": int(interaction_matrix.nnz),
        },
    )

    save_bundle(bundle, config.artifact_path)
    logger.info("Training complete: %s", bundle.params)
    return bundle


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the RecomSense ALS recommender.")
    parser.add_argument("--interactions", type=Path, help="Path to the interactions CSV.")
    parser.add_argument("--artifact", type=Path, help="Where to write the model bundle.")
    parser.add_argument("--factors", type=int, help="Number of ALS latent factors.")
    parser.add_argument("--iterations", type=int, help="Number of ALS iterations.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = _parse_args(argv)

    overrides = {}
    if args.interactions:
        overrides["interactions_path"] = args.interactions
    if args.artifact:
        overrides["artifact_path"] = args.artifact
    if args.factors:
        overrides["factors"] = args.factors
    if args.iterations:
        overrides["iterations"] = args.iterations

    train(replace(get_settings(), **overrides) if overrides else None)


if __name__ == "__main__":
    main()
