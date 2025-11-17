# Use a slim Python image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies needed for OpenCV, mediapipe, etc.
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose Render port
ENV PORT=8000

# Run migrations + start server
CMD ["sh", "-c", "python manage.py migrate && daphne -b 0.0.0.0 -p 8000 asl_lite.asgi:application"]

ENV DJANGO_SETTINGS_MODULE=asl_lite.settings
