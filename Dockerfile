FROM python:3.10-slim-bullseye

WORKDIR /app

# Install system dependencies for sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the Flask port
EXPOSE 5000

# Set environment defaults (override at runtime with -e or .env)
ENV FLASK_SECRET_KEY="change-me-in-production"
ENV PORT=5000

CMD ["python3", "app.py"]