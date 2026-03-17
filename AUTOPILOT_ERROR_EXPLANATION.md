# Autopilot Error: "Could not find a valid chrome browser binary"

## Error Message
```
Autopilot error: Could not find a valid chrome browser binary. 
Please make sure Chrome is installed. 
Or use the keyword argument: 'browser_executable_path=/path/to/your/browser'.
```

## Why This Error Occurs in Production

### Root Cause
This error occurs when **Playwright cannot find the Chrome/Chromium browser binary** in the production environment. The error typically comes from Playwright's browser launch mechanism when it tries to execute the browser executable.

### Why It Happens in Production (But Not Locally)

| Aspect | Local Development | Production (Docker) |
|--------|-------------------|-------------------|
| **Browser Installation** | Chrome/Chromium already installed on your OS | Must be installed during Docker build |
| **Browser Path** | System PATH includes browser location | Must be explicitly installed via package manager |
| **Dependencies** | System libraries already present | Must install all runtime dependencies |
| **Environment** | Full OS with all tools | Minimal container (python:3.11-slim) |

---

## Your Codebase Analysis

### 1. **Browser Launch Flow**

Your code uses **two browser controllers**:

#### **BrowserController** (Standard Mode)
```python
# backend/core/browser.py
self.browser = await self._playwright.chromium.launch(
    headless=True,
    args=["--disable-blink-features=AutomationControlled"]
)
```
- Launches Chromium via Playwright
- Requires Chromium binary to be installed

#### **StealthBrowserController** (Stealth Mode)
```python
# backend/core/stealth_browser.py
from seleniumbase.undetected.cdp_driver.browser import Browser as SBBrowser
self.sb_driver = await SBBrowser.create(headless=headless)
```
- Uses SeleniumBase's CDP driver for anti-bot evasion
- Also requires Chrome/Chromium binary

### 2. **Autopilot Initialization**
```python
# backend/core/autopilot/autopilot_mode.py
if settings.USE_STEALTH_BROWSER:
    browser = StealthBrowserController()
else:
    browser = BrowserController()

await browser.start(headless=settings.BROWSER_HEADLESS)
```

---

## Why Production Fails

### Problem 1: Missing Browser Binary in Docker
Your `Dockerfile` installs system dependencies but **may not have Chromium properly installed**:

```dockerfile
# Current Dockerfile
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    curl \
    ca-certificates \
    libnss3 \
    libnspr4 \
    # ... other libs ...
    && rm -rf /var/lib/apt/lists/*

# ✅ This installs Playwright browsers
RUN playwright install chromium
```

**Issue**: While `playwright install chromium` is present, it may fail silently if:
- System dependencies are incomplete
- Playwright installation fails
- Browser binary doesn't get downloaded properly

### Problem 2: Playwright Browser Installation Timing
```dockerfile
# Current order:
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt
RUN playwright install chromium  # ← Happens AFTER pip install
```

If `pip install` fails or Playwright isn't properly installed, the `playwright install chromium` command may not work correctly.

### Problem 3: SeleniumBase CDP Driver
If using `USE_STEALTH_BROWSER=True`, SeleniumBase also needs:
- Chrome/Chromium binary
- Additional system libraries
- Proper initialization

---

## Solutions

### ✅ Solution 1: Fix Dockerfile (Recommended)

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PLAYWRIGHT_BROWSERS_PATH=/usr/local/bin/playwright-browsers
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# 1. Install ALL system dependencies FIRST (including chromium)
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    curl \
    ca-certificates \
    chromium-browser \
    chromium \
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

# 2. Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# 3. Install Playwright browsers (should find system chromium)
RUN playwright install chromium

# 4. Verify browser installation
RUN which chromium-browser || which chromium || echo "Warning: chromium not found in PATH"

COPY . .

EXPOSE 8080

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
```

**Key Changes:**
- Added `chromium-browser` and `chromium` packages to apt-get
- Verification step to confirm browser is installed
- Ensures system chromium is available before Playwright tries to use it

---

### ✅ Solution 2: Use Explicit Browser Path

If you want to use the system-installed Chromium:

```python
# backend/core/browser.py
import shutil

class BrowserController:
    async def start(self, headless=True):
        # Find chromium in system PATH
        chromium_path = shutil.which("chromium") or shutil.which("chromium-browser")
        
        if chromium_path:
            logger.info(f"Using system Chromium: {chromium_path}")
            self.browser = await self._playwright.chromium.launch(
                executable_path=chromium_path,
                headless=headless,
                args=["--disable-blink-features=AutomationControlled"]
            )
        else:
            # Fallback to Playwright's bundled browser
            logger.warning("System Chromium not found, using Playwright bundled browser")
            self.browser = await self._playwright.chromium.launch(
                headless=headless,
                args=["--disable-blink-features=AutomationControlled"]
            )
```

---

### ✅ Solution 3: Use Headless Mode with Fallback

```python
# backend/config.py
import os

class Settings:
    # ... existing settings ...
    
    # Browser configuration
    BROWSER_HEADLESS = os.getenv("BROWSER_HEADLESS", "true").lower() == "true"
    BROWSER_EXECUTABLE_PATH = os.getenv("BROWSER_EXECUTABLE_PATH", None)
    USE_STEALTH_BROWSER = os.getenv("USE_STEALTH_BROWSER", "false").lower() == "true"
    
    # Fallback: disable stealth mode in production if browser issues occur
    ENABLE_BROWSER_FALLBACK = os.getenv("ENABLE_BROWSER_FALLBACK", "true").lower() == "true"
```

Then in `autopilot_mode.py`:

```python
async def run_autopilot(...):
    try:
        if settings.USE_STEALTH_BROWSER:
            browser = StealthBrowserController()
        else:
            browser = BrowserController()
        
        await browser.start(headless=settings.BROWSER_HEADLESS)
    
    except Exception as e:
        if "chrome browser binary" in str(e).lower() and settings.ENABLE_BROWSER_FALLBACK:
            logger.warning(f"Browser launch failed: {e}. Attempting fallback...")
            # Try standard browser if stealth failed
            browser = BrowserController()
            await browser.start(headless=settings.BROWSER_HEADLESS)
        else:
            raise
```

---

### ✅ Solution 4: Docker Compose with Browser Service

For more complex setups, use a separate browser container:

```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8080:8080"
    environment:
      - BROWSER_EXECUTABLE_PATH=http://browser:3000
      - USE_STEALTH_BROWSER=false
    depends_on:
      - browser

  browser:
    image: mcr.microsoft.com/playwright:v1.45.0-jammy
    ports:
      - "3000:3000"
    environment:
      - BROWSER_HEADLESS=true
```

---

## Debugging Steps

### 1. Check Docker Build Logs
```bash
docker build --progress=plain -t autopilot-backend ./backend 2>&1 | grep -i "chromium\|playwright\|browser"
```

### 2. Verify Browser Installation in Container
```bash
docker run -it autopilot-backend bash
# Inside container:
which chromium
which chromium-browser
playwright install --list
```

### 3. Check Playwright Browser Cache
```bash
docker run -it autopilot-backend bash
# Inside container:
ls -la /usr/local/bin/playwright-browsers/
```

### 4. Enable Debug Logging
```python
# backend/main.py
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("playwright")
logger.setLevel(logging.DEBUG)
```

---

## Summary

| Issue | Cause | Solution |
|-------|-------|----------|
| **Browser binary not found** | Chromium not installed in Docker | Add `chromium-browser` to apt-get |
| **Playwright install fails** | Missing system dependencies | Install all required libs before pip install |
| **Stealth mode fails** | SeleniumBase needs Chrome | Use standard BrowserController as fallback |
| **Path issues** | Browser not in system PATH | Use explicit `executable_path` parameter |
| **Container size** | Chromium is large | Use multi-stage build or slim image |

---

## Recommended Fix (Quick Implementation)

**Update your Dockerfile:**

```dockerfile
# Add chromium-browser to the apt-get install line
RUN apt-get update && apt-get install -y \
    chromium-browser \  # ← ADD THIS LINE
    wget \
    gnupg \
    curl \
    # ... rest of dependencies ...
```

**Add fallback in autopilot_mode.py:**

```python
try:
    await browser.start(headless=settings.BROWSER_HEADLESS)
except Exception as e:
    if "chrome browser binary" in str(e).lower():
        logger.error(f"Browser launch failed in production: {e}")
        await _send_error(websocket, 
            "Browser initialization failed. Please check server logs.")
        return {"status": "error", "message": "Browser not available"}
    raise
```

This ensures graceful error handling and clear feedback to users when browser issues occur.
