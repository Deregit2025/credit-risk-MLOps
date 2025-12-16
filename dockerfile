# Use Python 3.13 slim image to match conda.yml
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy your environment files
COPY mlruns/1/models/m-cacd6654018946898329870611bbc8e7/artifacts/conda.yaml ./conda.yml
COPY requirements.txt ./

# Install dependencies using pip (as per your conda.yml)
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir mlflow==3.7.0 \
                       cloudpickle==3.1.2 \
                       numpy==2.3.5 \
                       pandas==2.3.3 \
                       psutil==7.1.3 \
                       pyarrow==22.0.0 \
                       scikit-learn==1.8.0 \
                       scipy==1.16.3 \
                       fastapi==0.111.1 \
                       uvicorn==0.38.0 \
                       pydantic==2.3.0

# Copy the project code
COPY . .

# Expose port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
