
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
# gcc might be needed for some python packages, but for these it's usually fine.
# keeping it minimal.
# RUN apt-get update && apt-get install -y gcc

# Copy requirements first to leverage cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create models directory (optional as code does it, but good for permissions)
RUN mkdir -p models && chmod 777 models

# Expose port
EXPOSE 8000

# Run command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
