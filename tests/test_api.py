"""HTTP contract: status codes, response shape, and cold-start signalling."""

from __future__ import annotations

import pytest

from model.recommender import STRATEGY_COLLABORATIVE, STRATEGY_POPULARITY
from tests.conftest import KNOWN_USER_ID, UNKNOWN_USER_ID

RECOMMENDATION_KEYS = {
    "user_id",
    "strategy",
    "cold_start",
    "requested_top_n",
    "count",
    "recommendations",
}
ITEM_KEYS = {"rank", "product_id", "score"}


class TestResponseFormat:
    def test_known_user_returns_the_documented_payload(self, client) -> None:
        response = client.get(f"/recommendations/{KNOWN_USER_ID}", params={"top_n": 3})

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")

        body = response.json()
        assert set(body) == RECOMMENDATION_KEYS
        assert body["user_id"] == KNOWN_USER_ID
        assert body["strategy"] == STRATEGY_COLLABORATIVE
        assert body["cold_start"] is False
        assert body["requested_top_n"] == 3
        assert body["count"] == len(body["recommendations"]) == 3

    def test_every_item_has_the_documented_fields_and_types(self, client) -> None:
        body = client.get(f"/recommendations/{KNOWN_USER_ID}", params={"top_n": 4}).json()

        for position, item in enumerate(body["recommendations"], start=1):
            assert set(item) == ITEM_KEYS
            assert item["rank"] == position
            assert isinstance(item["product_id"], int)
            assert isinstance(item["score"], float)

    def test_top_n_controls_the_list_length(self, client) -> None:
        for top_n in (1, 2, 5):
            body = client.get(f"/recommendations/{KNOWN_USER_ID}", params={"top_n": top_n}).json()
            assert body["requested_top_n"] == top_n
            assert body["count"] <= top_n

    def test_top_n_is_optional(self, client, engine) -> None:
        body = client.get(f"/recommendations/{KNOWN_USER_ID}").json()

        assert body["requested_top_n"] == engine.settings.default_top_n


class TestColdStartEndpoint:
    def test_unknown_user_returns_popular_products(self, client) -> None:
        response = client.get(f"/recommendations/{UNKNOWN_USER_ID}", params={"top_n": 3})

        assert response.status_code == 200
        body = response.json()
        assert body["strategy"] == STRATEGY_POPULARITY
        assert body["cold_start"] is True
        assert body["count"] == 3

    def test_cold_start_list_matches_the_popular_products_endpoint(self, client) -> None:
        fallback = client.get(
            f"/recommendations/{UNKNOWN_USER_ID}", params={"top_n": 3}
        ).json()["recommendations"]
        popular = client.get("/products/popular", params={"top_n": 3}).json()

        assert [item["product_id"] for item in fallback] == [
            item["product_id"] for item in popular
        ]

    def test_unknown_user_returns_404_when_cold_start_disabled(self, client) -> None:
        response = client.get(
            f"/recommendations/{UNKNOWN_USER_ID}",
            params={"top_n": 3, "allow_cold_start": "false"},
        )

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "unknown_user"


class TestErrorHandling:
    @pytest.mark.parametrize("top_n", [0, -5, 10_000])
    def test_out_of_range_top_n_returns_422(self, client, top_n: int) -> None:
        response = client.get(f"/recommendations/{KNOWN_USER_ID}", params={"top_n": top_n})

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_request"

    def test_non_numeric_top_n_returns_422(self, client) -> None:
        response = client.get(f"/recommendations/{KNOWN_USER_ID}", params={"top_n": "many"})

        assert response.status_code == 422
        body = response.json()
        assert body["error"]["code"] == "invalid_request"
        assert any("top_n" in detail["field"] for detail in body["error"]["details"])

    def test_non_numeric_user_id_returns_422(self, client) -> None:
        response = client.get("/recommendations/not-a-user", params={"top_n": 3})

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_request"

    def test_errors_use_a_consistent_envelope(self, client) -> None:
        body = client.get(f"/recommendations/{KNOWN_USER_ID}", params={"top_n": 0}).json()

        assert set(body) == {"error"}
        assert {"code", "message"} <= set(body["error"])

    def test_unknown_route_returns_404(self, client) -> None:
        assert client.get("/no-such-endpoint").status_code == 404


class TestServiceEndpoints:
    def test_health_reports_a_loaded_model(self, client, engine) -> None:
        body = client.get("/health").json()

        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["known_users"] == engine.known_user_count
        assert body["known_products"] == engine.known_product_count

    def test_index_lists_the_endpoints(self, client) -> None:
        body = client.get("/").json()

        assert body["service"] == "RecomSense API"
        assert "/recommendations/{user_id}" in body["endpoints"]

    def test_openapi_schema_is_generated(self, client) -> None:
        schema = client.get("/openapi.json").json()

        assert schema["info"]["title"] == "RecomSense API"
        assert "/recommendations/{user_id}" in schema["paths"]
