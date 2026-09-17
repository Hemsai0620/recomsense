# RecomSense

A personalized product recommendation service. It learns from past user activity
(views, clicks, purchases) and answers one question over HTTP: *which products
should we show this user next?*

RecomSense is built around collaborative filtering with **Alternating Least
Squares (ALS)** and served through a **FastAPI** JSON API. Users the model has
never seen are not left empty-handed — they fall back to a popularity ranking.

---

## The problem in plain English

An online store has thousands of products and a limited amount of screen space.
Showing everyone the same shelf wastes that space: a customer who keeps buying
coffee equipment does not need the same homepage as one who only browses books.

The store does, however, have a log of what each customer did — which items they
looked at, which they clicked, which they bought. That log is enough to find
useful patterns without knowing anything about the products themselves:

> People who interacted with the same things tend to like the same things next.

RecomSense turns that idea into a service. It reads the interaction log, learns
the pattern offline, and then serves a ranked list of product IDs for any user
you ask about.

Two situations have to be handled differently:

| Situation | What we know | What RecomSense does |
| --- | --- | --- |
| Returning user | They have interaction history | Score products with the trained ALS model |
| New / unknown user | Nothing at all | Return the most popular products instead |

The second case is the **cold-start problem**, and it is handled explicitly
rather than failing with an error.

---

## How the recommendation works

### 1. Events become numbers

Not all activity means the same thing. A purchase is a stronger signal of
interest than a click, and a click is stronger than a view. Each event type is
mapped to a weight (`model/features.py`):

| Event | Weight |
| --- | --- |
| `view` | 1.0 |
| `click` | 2.0 |
| `purchase` | 3.0 |

All weights for the same `(user, product)` pair are summed into one confidence
score. Rows with an unrecognised event type are dropped with a warning.

### 2. Ratings become a sparse matrix

The weighted ratings are arranged into a `users x products` matrix. Almost every
cell is empty — a typical shopper touches a handful of products out of the whole
catalogue — so it is stored as a SciPy **CSR** (compressed sparse row) matrix.

Users and products are given contiguous internal indexes, and the mapping from
real IDs to those indexes is stored alongside the model. That mapping is what
lets the service answer "have I ever seen this user before?" — the question the
cold-start fallback depends on.

### 3. ALS learns latent factors

`implicit`'s `AlternatingLeastSquares` factorises the interaction matrix into
user and product factor vectors. ALS alternates between holding the product
factors fixed while solving for user factors and vice versa, until the product
of the two matrices reconstructs the observed interactions as closely as
possible. The dot product of a user vector and a product vector is that user's
predicted affinity for the product.

Products the user has already interacted with are excluded from their own
recommendations, so the list is always something new.

> The number of latent factors is automatically reduced if the interaction
> matrix is smaller than the configured value — you cannot learn 50 factors from
> a 15 × 10 matrix.

### 4. Popularity is computed as a safety net

At the same time, products are ranked by their total interaction weight across
all users. This ranking is stored with the model and used whenever the ALS model
cannot help.

### 5. Everything is saved as one bundle

ALS needs the user's own interaction row at prediction time, so the trained
factors alone are not enough to serve traffic. Training writes a single pickled
bundle containing the model, the interaction matrix, the ID mappings, the
popularity ranking and the training parameters. The bundle carries a format
version, so a stale artifact fails loudly instead of misbehaving quietly.

---

## Architecture

```
data/interactions.csv                 raw event log
        |
        v
data/loader.py                        read + validate the CSV
        |
        v
model/features.py                     events -> weighted ratings
        |
        v
model/matrix.py                       ID mappings, CSR matrix, popularity ranking
        |
        v
model/trainer.py                      fit ALS
        |
        v
model/artifacts.py                    artifacts/recommender.pkl  (one bundle)
        |
        v
model/recommender.py                  RecommendationEngine: ALS or popularity
        |
        v
api/routes.py + api/app.py            FastAPI JSON endpoints
```

Layers only depend downwards. `utils/` (configuration, logging, error types) is
shared by all of them.

**Design notes**

- The engine is loaded **lazily and cached**, so the API starts even when no
  model has been trained yet and reports that truthfully on `/health`.
- Every domain error carries its own HTTP status code, so the API layer contains
  no status-code guesswork.
- All configuration has working defaults and can be overridden through
  `RECOMSENSE_*` environment variables — no code edits needed for Docker or CI.

---

## Project structure

```
recomsense/
├── api/                         FastAPI service
│   ├── __init__.py
│   ├── app.py                   app factory, error handlers, root endpoint
│   ├── dependencies.py          cached engine dependency
│   ├── routes.py                /health, /recommendations, /products/popular
│   └── schemas.py               Pydantic request/response contracts
├── data/                        dataset + its loader
│   ├── __init__.py
│   ├── interactions.csv         sample user-product event log
│   └── loader.py                CSV reading and validation
├── model/                       ML pipeline and serving logic
│   ├── __init__.py
│   ├── artifacts.py             bundle definition, save/load
│   ├── evaluation.py            precision@k, recall@k
│   ├── features.py              event weighting and aggregation
│   ├── matrix.py                ID mappings, CSR matrix, popularity ranking
│   ├── recommender.py           RecommendationEngine + cold-start fallback
│   └── trainer.py               training entry point
├── tests/                       pytest suite
│   ├── __init__.py
│   ├── conftest.py              shared fixtures (trained once per session)
│   ├── test_api.py              HTTP contract and error handling
│   ├── test_pipeline.py         loading, features, matrix, metrics
│   └── test_recommender.py      known users, cold start, Top-N validation
├── utils/                       shared helpers
│   ├── __init__.py
│   ├── config.py                settings and environment overrides
│   ├── errors.py                error types with HTTP status codes
│   └── logging.py               logging setup
├── artifacts/                   generated model bundle (git-ignored)
├── .github/workflows/ci.yml     train, test and build the image on push
├── .dockerignore
├── .env.example                 documented configuration template
├── .gitignore
├── Dockerfile
├── pyproject.toml               project metadata and pytest configuration
├── requirements.txt             runtime dependencies
├── requirements-dev.txt         runtime + test dependencies
└── README.md
```

---

## Setup

Requires **Python 3.10 or newer**.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt          # runtime only
pip install -r requirements-dev.txt      # runtime + test tools
```

Configuration is optional. To change anything, copy the template and edit it:

```bash
cp .env.example .env
```

`.env` is git-ignored; the variables can equally be exported directly in your
shell or set in your deployment environment.

---

## Training the model

```bash
python -m model.trainer
```

This reads `data/interactions.csv`, builds the interaction matrix, fits ALS and
writes `artifacts/recommender.pkl`. The trainer logs the resulting matrix shape
and the parameters actually used.

Command-line overrides are available for one-off runs:

```bash
python -m model.trainer --interactions data/interactions.csv \
                        --artifact artifacts/recommender.pkl \
                        --factors 32 --iterations 30
```

The artifact is not committed to version control — retrain whenever you change
the dataset or the hyperparameters.

### Using your own data

Point `RECOMSENSE_INTERACTIONS_PATH` at any CSV with these columns:

| Column | Required | Notes |
| --- | --- | --- |
| `user_id` | yes | integer |
| `product_id` | yes | integer |
| `event` | yes | one of `view`, `click`, `purchase` |
| `timestamp` | no | carried in the file but not used for training |

The bundled `data/interactions.csv` is a small synthetic sample (58 events,
15 users, 10 products) meant for demonstrating and testing the pipeline — not a
benchmark dataset.

---

## Starting the API

```bash
uvicorn api.app:app --reload           # development
uvicorn api.app:app --host 0.0.0.0 --port 8000   # production-style
```

Interactive docs: <http://127.0.0.1:8000/docs>

### Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Service metadata and endpoint list |
| `GET` | `/health` | Whether the trained model is loaded |
| `GET` | `/recommendations/{user_id}` | Top-N recommendations for a user |
| `GET` | `/products/popular` | The popularity ranking behind the fallback |

`/recommendations/{user_id}` accepts two query parameters:

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `top_n` | integer, 1–50 | 5 | How many products to return |
| `allow_cold_start` | boolean | `true` | If `false`, unknown users return `404` instead of popular products |

---

## Example requests and responses

### Known user — served by the model

```bash
curl "http://127.0.0.1:8000/recommendations/1?top_n=5"
```

```json
{
  "user_id": 1,
  "strategy": "collaborative_filtering",
  "cold_start": false,
  "requested_top_n": 5,
  "count": 5,
  "recommendations": [
    { "rank": 1, "product_id": 105, "score": 0.014463 },
    { "rank": 2, "product_id": 107, "score": 0.012313 },
    { "rank": 3, "product_id": 108, "score": 0.009197 },
    { "rank": 4, "product_id": 110, "score": 0.003634 },
    { "rank": 5, "product_id": 106, "score": -0.000093 }
  ]
}
```

### Unknown user — cold-start fallback

```bash
curl "http://127.0.0.1:8000/recommendations/9999?top_n=5"
```

```json
{
  "user_id": 9999,
  "strategy": "popularity",
  "cold_start": true,
  "requested_top_n": 5,
  "count": 5,
  "recommendations": [
    { "rank": 1, "product_id": 102, "score": 17.0 },
    { "rank": 2, "product_id": 101, "score": 16.0 },
    { "rank": 3, "product_id": 103, "score": 10.0 },
    { "rank": 4, "product_id": 104, "score": 10.0 },
    { "rank": 5, "product_id": 105, "score": 10.0 }
  ]
}
```

### Health check

```bash
curl "http://127.0.0.1:8000/health"
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "trained_at": "2026-09-17T21:10:47+00:00",
  "known_users": 15,
  "known_products": 10,
  "detail": null
}
```

---

## Cold-start behaviour

When a request arrives for a user the model has never seen — or a user whose
interaction row turns out to be empty — RecomSense does **not** fail. It returns
the products with the highest total interaction weight across all users, and
labels the response so the caller can tell the difference:

```json
{ "strategy": "popularity", "cold_start": true }
```

Known users always keep the ALS path (`"strategy": "collaborative_filtering"`,
`"cold_start": false`). The fallback is exactly the ranking exposed by
`/products/popular`, so the two can be compared directly.

If you would rather treat an unknown user as an error — for example when a
caller should only ever pass real, existing user IDs — disable the fallback per
request:

```bash
curl "http://127.0.0.1:8000/recommendations/9999?allow_cold_start=false"
```

```json
{
  "error": {
    "code": "unknown_user",
    "message": "User 9999 has no interaction history."
  }
}
```

---

## Error handling

Every failure returns the same JSON envelope, with a stable machine-readable
`code`:

| Status | `code` | When |
| --- | --- | --- |
| `404` | `unknown_user` | Unknown user while `allow_cold_start=false` |
| `422` | `invalid_request` | `top_n` out of range or non-numeric, `user_id` not an integer |
| `422` | `invalid_dataset` | The interaction CSV is missing, empty or malformed |
| `503` | `model_unavailable` | No trained bundle yet, or the bundle cannot be read |

Invalid `top_n`:

```bash
curl "http://127.0.0.1:8000/recommendations/1?top_n=0"
```

```json
{
  "error": {
    "code": "invalid_request",
    "message": "One or more request parameters are invalid.",
    "details": [
      { "field": "top_n", "message": "Input should be greater than or equal to 1" }
    ]
  }
}
```

Non-integer `user_id`:

```bash
curl "http://127.0.0.1:8000/recommendations/abc"
```

```json
{
  "error": {
    "code": "invalid_request",
    "message": "One or more request parameters are invalid.",
    "details": [
      {
        "field": "path.user_id",
        "message": "Input should be a valid integer, unable to parse string as an integer"
      }
    ]
  }
}
```

If the API is started before any training run, `/health` reports
`"status": "degraded"` with `"model_loaded": false`, and the recommendation
endpoints return `503 model_unavailable`.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite trains a small model once per session into a temporary directory, so
running the tests never overwrites `artifacts/recommender.pkl`. It covers:

- known-user recommendations via the ALS model
- unknown-user and empty-history cold-start fallback
- `allow_cold_start=false` producing `404`
- Top-N validation: zero, negative, above the maximum, and non-integer values
- the exact JSON response shape, field types and rank ordering
- the shared error envelope and its status codes
- data loading, event weighting, matrix construction and ranking metrics

---

## Docker

```bash
docker build -t recomsense .
docker run -p 8000:8000 recomsense
```

The image installs the dependencies, trains the model during the build and
starts the API as a non-root user. The service is ready to answer requests as
soon as the container starts; `docker run` needs no extra training step.

To serve a different dataset, mount it and retrain inside the container:

```bash
docker run -p 8000:8000 \
  -v "$(pwd)/my-data.csv:/app/data/interactions.csv:ro" \
  recomsense sh -c "python -m model.trainer && uvicorn api.app:app --host 0.0.0.0 --port 8000"
```

---

## Configuration reference

All variables are optional; the defaults below are what the project uses.

| Variable | Default | Purpose |
| --- | --- | --- |
| `RECOMSENSE_INTERACTIONS_PATH` | `data/interactions.csv` | Dataset location |
| `RECOMSENSE_ARTIFACT_PATH` | `artifacts/recommender.pkl` | Model bundle location |
| `RECOMSENSE_FACTORS` | `50` | ALS latent factors (auto-capped by matrix size) |
| `RECOMSENSE_ITERATIONS` | `20` | ALS iterations |
| `RECOMSENSE_REGULARIZATION` | `0.05` | ALS regularization |
| `RECOMSENSE_RANDOM_STATE` | `42` | Seed for reproducible training |
| `RECOMSENSE_DEFAULT_TOP_N` | `5` | Top-N used when the request omits it |
| `RECOMSENSE_MAX_TOP_N` | `50` | Largest accepted Top-N |
| `RECOMSENSE_POPULAR_POOL_SIZE` | `100` | How many popular products to store for fallback |
| `RECOMSENSE_LOG_LEVEL` | `INFO` | Logging verbosity |

---

## Tech stack

Python · FastAPI · Pydantic · Uvicorn · implicit (ALS) · SciPy sparse matrices ·
pandas · NumPy · pytest · Docker

---

## Possible next steps

Not implemented — listed as directions, not features:

- content-based signals blended with collaborative filtering
- time decay so recent interactions count for more
- incremental retraining instead of full rebuilds
- an experiment tracker for hyperparameter runs
- request metrics and recommendation quality monitoring in production
