# Use official Python 3.10 image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 8000

# Start the app using gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000"]
