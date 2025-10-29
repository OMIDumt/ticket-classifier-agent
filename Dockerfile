FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependencies and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model files
COPY src /app/src
COPY models /app/models

# Environment variables
ENV PYTHONPATH=/app/src
ENV MODEL_DIR=/app/models/model-1
ENV PORT=8000

EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
