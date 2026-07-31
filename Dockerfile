# Dockerfile — zero-cost deploy on Render.com (Free tier)
# Build context: this folder. Render reads render.yaml -> dockerfilePath ./Dockerfile.
FROM python:3.11-slim

WORKDIR /app

# Install deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Render passes FRED_API_KEY as a *build* env var. Docker does NOT auto-expose
# env vars at build time — we must declare it as ARG then promote it to ENV so
# the RUN steps below can see it. Without this, the build-time pre-fetch never
# receives the key, so EMVMACROBUS (F6) and the other FRED features are NOT
# baked into the image (they would silently fall back to AAII/runtime fetch).
ARG FRED_API_KEY
ENV FRED_API_KEY=$FRED_API_KEY

# Pre-fetch & bake data caches into the image so the FIRST request is instant
# and free-tier health checks don't time out on a slow live data pull.
# Network at build is best-effort: if it fails we just fall back to a runtime
# fetch (the `|| echo` keeps the build green).
RUN python pipeline.py \
    && python -c "import pipeline as p; [p.get_price_series(t, refresh=True) for t in ['^GSPC','^IXIC','SPY','SHY','QQQ']]" \
    || echo "pre-fetch skipped (will fetch live at runtime)"

EXPOSE 8501

# Render injects $PORT at runtime; Streamlit must bind to it.
# ${PORT:-8501} keeps a sane default if PORT is somehow unset.
CMD streamlit run app.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true
