# Base image with Python 3.10
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential

# Copy all project files
COPY . .

# Install Python packages
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Use Render's PORT environment variable
ENV PORT=8000

# Expose port (for Docker, optional for Render)
EXPOSE $PORT

# Run the app using gunicorn and dynamic port
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:$PORT"]
