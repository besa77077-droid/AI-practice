FROM python:3.11-slim

# ffmpeg is required by faster-whisper to decode mp3/mp4/m4a — its absence
# was the most likely reason audio uploads silently failed in the previous
# prototype. Install it explicitly and verify it before shipping the image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ffmpeg -version

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[audio]"

ENV HOST=0.0.0.0 \
    PORT=8000 \
    INSIGHT_ENGINE_DB=/data/insights.db \
    INSIGHT_ENGINE_LLM=ollama

VOLUME ["/data"]
EXPOSE 8000

CMD ["insight-engine-serve"]
