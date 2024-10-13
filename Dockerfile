# Use an official Python runtime as the base image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file into the container
COPY requirements.txt .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libc6-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt
# Copy only the necessary source files into the container
COPY src/ ./src/

# Set the Python path to include the src directory
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the FastAPI application when the container launches
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]