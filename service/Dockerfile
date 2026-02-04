FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libnetcdf-dev \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Library and install it
# Copy Library (Root Context) and install it
# Assumes build context is the repository root (containing src/ and service/)
COPY pyproject.toml README.md /tmp/library/
COPY src /tmp/library/src
RUN pip install /tmp/library

# Copy Service dependencies
COPY service/pyproject.toml /app/
RUN pip install .

# Copy App Source
COPY service/app /app/app

# Run
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
