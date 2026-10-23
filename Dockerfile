FROM python:3.11-slim

# Install system dependencies including FFmpeg and curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Railway dynamically injects $PORT (defaults to 8080)
ENV PORT=8080
EXPOSE ${PORT}

# Launch Uvicorn using shell form so $PORT is evaluated at container runtime
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
