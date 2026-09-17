# RecomSense - training + API service image.
#
# The model is trained during the build so the container is ready to serve as
# soon as it starts. Mount a different dataset and rerun the trainer inside the
# container to refresh the model without rebuilding.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OPENBLAS_NUM_THREADS=1 \
    RECOMSENSE_ARTIFACT_PATH=/app/artifacts/recommender.pkl

WORKDIR /app

# Dependencies first so code edits do not invalidate the pip layer.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Bake a trained model bundle into the image.
RUN python -m model.trainer

# Run as a non-root user.
RUN useradd --create-home --uid 10001 recomsense \
    && chown -R recomsense:recomsense /app
USER recomsense

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status == 200 else 1)"

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
