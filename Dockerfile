# Use python:3.10-slim as base image
FROM python:3.10-slim

# Set working directory to /app
WORKDIR /app

# Copy requirements.txt first (for better caching)
COPY requirements.txt .

# Install dependencies using pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY tts_app.py .

# Create the generated_audio directory for temporary audio files
RUN mkdir -p generated_audio

# Expose port 5000
EXPOSE 5000

# Use environment variable for PORT if available, default to 5000
ENV PORT=5000

# Start the app using gunicorn
# gunicorn -w 1 -b 0.0.0.0:$PORT tts_app:app
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:$PORT", "tts_app:app"]

