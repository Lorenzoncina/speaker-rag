# Base image
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Working directory
WORKDIR /app

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# 2. Install Python Requirements (Cached Layer)
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# 3. Download Models (Cached Layer)
# We copy the script and run it so models are saved into the image filesystem
COPY download_models.py /app/
COPY age_model/age_probe.pt /app/age_model/
RUN python3 download_models.py

# 4. Copy Application Code
# This is the layer that changes most often
COPY . /app

# Expose port
EXPOSE 8006

# Start Server
CMD ["gunicorn", "--config", "gunicorn_conf.py", "endpoints:app"]