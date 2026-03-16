FROM python:3.11-slim

# 1. Set environment variables to ensure Playwright knows where to look
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PLAYWRIGHT_BROWSERS_PATH=/usr/local/bin/playwright-browsers

WORKDIR /app

# 2. Install minimal system deps first
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# 3. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Install Playwright browsers AND their system dependencies
# We use the 'with-deps' flag to automatically pull in the 20+ Linux libs 
# (like libnss3, libatk, etc.) that Chromium needs to run.
RUN playwright install chromium --with-deps

COPY . .

EXPOSE 8080

# 5. Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]