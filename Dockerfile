# ---------- auton-agent: production image ----------
# Slim base because Render Free gives 512 MB RAM / 0.1 CPU; every megabyte and
# every second of cold start matters. No build toolchain is installed on
# purpose: the agent can install what it needs at runtime, inside its sandbox.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    WORKSPACE_ROOT=/app/workspace \
    STATE_ROOT=/app/.agentstate

WORKDIR /app

# Minimal OS tooling the agent is allowed to rely on. procps gives it `ps` so it
# can inspect its own processes; curl/ca-certificates are needed for HTTPS.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl git procps jq \
 && rm -rf /var/lib/apt/lists/*

# Dependencies first, so the layer is cached across code-only redeploys.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runtime (test-only deps are excluded by not copying the test suite's extras).
COPY src/ ./src/

# Non-root runtime user: the agent executes commands, so it must not be root.
RUN useradd --create-home --uid 10001 agent \
 && mkdir -p /app/workspace /app/.agentstate \
 && chown -R agent:agent /app
USER agent

EXPOSE 10000

# Render sets PORT; the app reads it. Health check hits /health.
CMD ["python", "-m", "agentcore.main", "service"]
