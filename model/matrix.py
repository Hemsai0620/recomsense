"""Build the sparse user-item matrix and the id/index mappings around it.

The original pipeline used the raw ids as matrix coordinates. Dense integer
mappings are used instead so that the matrix stays compact and, more
importantly, so the service can tell an unknown user apart from a known one -
which is what the cold-start fallback depends on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix


@dataclass(frozen=True)
class IndexMapping:
    """Bidirectional mapping between dataset ids and matrix positions."""

    user_to_index: dict[int, int] = field(default_factory=dict)
    product_to_index: dict[int, int] = field(default_factory=dict)
    index_to_user: list[int] = field(default_factory=list)
    index_to_product: list[int] = field(default_factory=list)

    @property
    def n_users(self) -> int:
        return len(self.index_to_user)

    @property
    def n_products(self) -> int:
        return len(self.index_to_product)


def build_index_mapping(ratings: pd.DataFrame) -> IndexMapping:
    """Assign a contiguous matrix index to every user and product."""
    users = sorted(int(value) for value in ratings["user_id"].unique())
    products = sorted(int(value) for value in ratings["product_id"].unique())
    return IndexMapping(
        user_to_index={user: position for position, user in enumerate(users)},
        product_to_index={product: position for position, product in enumerate(products)},
        index_to_user=users,
        index_to_product=products,
    )


def build_interaction_matrix(ratings: pd.DataFrame, mapping: IndexMapping) -> csr_matrix:
    """Return the ``users x products`` CSR matrix of aggregated ratings."""
    rows = ratings["user_id"].astype(int).map(mapping.user_to_index).to_numpy()
    columns = ratings["product_id"].astype(int).map(mapping.product_to_index).to_numpy()
    values = ratings["rating"].astype("float32").to_numpy()

    matrix = coo_matrix(
        (values, (rows, columns)),
        shape=(mapping.n_users, mapping.n_products),
        dtype="float32",
    ).tocsr()
    matrix.sum_duplicates()
    return matrix


def rank_products_by_popularity(ratings: pd.DataFrame, limit: int) -> list[tuple[int, float]]:
    """Rank products by total interaction weight, strongest first.

    This ranking is the cold-start fallback: it is the best guess available
    when a user has no history for the collaborative filtering model to use.
    """
    totals = (
        ratings.groupby("product_id")["rating"]
        .sum()
        .sort_values(ascending=False)
        .head(max(int(limit), 0))
    )
    return [(int(product_id), float(score)) for product_id, score in totals.items()]
