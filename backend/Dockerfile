FROM python:3.11-slim

# 1. Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PLAYWRIGHT_BROWSERS_PATH=/usr/local/bin/playwright-browsers

# FIXED: Change /root/.cargo/bin to /root/.local/bin
ENV PATH="/root/.local/bin:${PATH}"

# Tell SeleniumBase / undetected-chromedriver to use the system Chromium binary
# so it never tries to download a driver at runtime (no network access needed).
ENV SB_BINARY_LOCATION=/usr/bin/chromium
ENV CHROME_EXECUTABLE_PATH=/usr/bin/chromium

# Cloud Run / Docker: Chromium needs a writable /tmp for crash dumps and user data.
# Also disable GPU and sandbox at the env level so every Chrome invocation inherits them.
ENV CHROME_FLAGS="--no-sandbox --disable-dev-shm-usage --disable-gpu --single-process --no-zygote"
ENV DBUS_SESSION_BUS_ADDRESS="unix:path=/dev/null"

# ── Stealth browser configuration ──
# Use Playwright's bundled Chromium (NOT system chromium) for the stealth browser.
# The system chromium does NOT support Playwright's headless protocol on Cloud Run.
# Playwright's bundled Chromium has proper headless support built in.
ENV USE_STEALTH_BROWSER=true
ENV BROWSER_HEADLESS=true

WORKDIR /app

# 2. Install system deps + Chromium browser + Xvfb (virtual display) + all runtime libs
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    curl \
    ca-certificates \
    chromium \
    dbus \
    xvfb \
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

# 3. Create a larger /dev/shm alternative for Chromium (Cloud Run has tiny /dev/shm)
# Chromium uses shared memory for rendering; without this it crashes on Cloud Run.
RUN mkdir -p /tmp/shm && chmod 1777 /tmp/shm

# 4. Verify Chromium is available and print its path (fails build if missing)
RUN which chromium && chromium --version

# 5. Install Python dependencies
COPY requirements.txt .
# We upgrade pip/setuptools first to avoid that 'use_2to3' error
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# 6. Install Playwright's Chromium browser + system dependencies
# playwright install --with-deps installs both the browser AND all required system libs
# This is the CRITICAL step — without it, Playwright's headless Chromium won't launch
RUN playwright install chromium --with-deps

COPY . .

EXPOSE 8080

# 7. Start Xvfb (virtual display) + FastAPI
# Xvfb provides a virtual X server so SeleniumBase CDP can launch Chromium
# even on headless Cloud Run containers. The display is set to :99.
# If Xvfb fails to start (non-fatal), Playwright fallback still works headlessly.
ENV DISPLAY=:99
CMD Xvfb :99 -screen 0 1280x800x24 -nolisten tcp -nolisten unix &>/dev/null & \
    sleep 1 && \
    uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
