# Use full Python 3.10 image (NOT slim)
FROM python:3.10

# Set work directory
WORKDIR /app

# Copy everything
COPY . .

# Upgrade pip and install all packages
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Use environment port
ENV PORT=8000
EXPOSE $PORT

# Run app using gunicorn
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:$PORT"]
