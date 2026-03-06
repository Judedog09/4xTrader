FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader vader_lexicon

# Copy application
COPY server_multitenant.py .

# Expose port
EXPOSE 8000

# Run the app
CMD ["python", "server_multitenant.py"]