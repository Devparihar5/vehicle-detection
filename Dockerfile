FROM python:3.10-slim

# System dependencies (for OpenCV)
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create uploads folder
RUN mkdir -p uploads

# Expose port
EXPOSE 8080

# Gunicorn launch
CMD ["gunicorn", "-b", "0.0.0.0:8080", "--threads=4", "app:app"]
