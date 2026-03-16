FROM python:3.11-slim

# 1. Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PLAYWRIGHT_BROWSERS_PATH=/usr/local/bin/playwright-browsers

# FIXED: Change /root/.cargo/bin to /root/.local/bin
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# 2. Install system deps + browser runtime libs + curl to get 'uv'
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    curl \
    ca-certificates \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libxshmfence1 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxext6 \
    libxrender1 \
    libglib2.0-0 \
    libpango-1.0-0 \
    libcairo2 \
    libatspi2.0-0 \
    libgtk-3-0 \
    fonts-liberation \
    fonts-noto-color-emoji \
    fonts-unifont \
    && rm -rf /var/lib/apt/lists/*

# 3. Install 'uv' (Fastest Python package installer)
ADD https://astral.sh/uv/install.sh /install.sh
RUN chmod +x /install.sh && /install.sh && rm /install.sh

# 4. Install Python dependencies using uv
COPY requirements.txt .
# We upgrade pip/setuptools first to avoid that 'use_2to3' error
RUN uv pip install --system --no-cache-dir --upgrade pip setuptools wheel
RUN uv pip install --system --no-cache-dir -r requirements.txt

# 5. Install Playwright browser (deps already installed above)
RUN playwright install chromium

COPY . .

EXPOSE 8080

# 6. Start FastAPI
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
