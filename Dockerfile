FROM python:3.11-slim

WORKDIR /app

# System deps for scipy/audioop
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)"

EXPOSE $PORT

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
