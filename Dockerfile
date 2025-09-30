FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data
COPY app/ ./app/
COPY felipe_data/ ./felipe_data/
COPY app/dash_data/ ./app/dash_data/
COPY parquet_data/ ./parquet_data/

# Set Python path to include the current directory
ENV PYTHONPATH=/app

# Expose port (Cloud Run provides $PORT; 8080 is conventional)
EXPOSE 8080

# Use gunicorn for production (bind to $PORT provided by Cloud Run)
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 app.dash_app:server