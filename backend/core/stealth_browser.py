"""
stealth_browser.py — Stealthy browser controller using SeleniumBase CDP + Playwright.

Architecture:
  1. SeleniumBase launches a stealthy Chrome browser in CDP mode
  2. Playwright connects to it via connect_over_cdp
  3. Mix and match: Use Playwright for automation + SeleniumBase for CAPTCHA solving

This approach bypasses most anti-bot detection systems including:
  - Cloudflare
  - reCAPTCHA
  - hCaptcha
  - DataDome
  - PerimeterX

Optimizations (v2):
  - Fast screenshot path: disables font-wait and visual-stability for PDF pages
  - Instant capture mode: 1-2s static delay instead of font/stability polling
  - Reduced screenshot timeout to 15s with aggressive window.stop() recovery
"""

import asyncio
import base64
import logging
import os
import shutil
import time
from io import BytesIO
from PIL import Image, ImageChops
from playwright.async_api import async_playwright, Page, Browser, Playwright

logger = logging.getLogger(__name__)


def _find_chromium_binary() -> str | None:
    """
    Locate the Chromium/Chrome binary on the current system.

    Search order:
      1. SB_BINARY_LOCATION / CHROME_EXECUTABLE_PATH env vars  (set in Dockerfile)
      2. Common Linux paths used by Debian/Ubuntu packages
      3. shutil.which() for anything on PATH

    Returns the first path that exists, or None if nothing is found.
    """
    # 1. Explicit env-var overrides (set in Dockerfile for production)
    for env_var in ("SB_BINARY_LOCATION", "CHROME_EXECUTABLE_PATH"):
        path = os.environ.get(env_var, "")
        if path and os.path.isfile(path):
            logger.info(f"Chromium binary from env {env_var}: {path}")
            return path

    # 2. Well-known fixed paths (Debian/Ubuntu apt packages)
    fixed_paths = [
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/usr/local/bin/chromium",
    ]
    for path in fixed_paths:
        if os.path.isfile(path):
            logger.info(f"Chromium binary found at: {path}")
            return path

    # 3. PATH lookup
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        path = shutil.which(name)
        if path:
            logger.info(f"Chromium binary found via PATH ({name}): {path}")
            return path

    logger.warning("No Chromium binary found on this system.")
    return None

# ─── Screenshot timeout tiers ───
# PDF/heavy pages: use a short timeout + instant capture (no font wait)
# Normal pages: standard timeout
SCREENSHOT_TIMEOUT_FAST_MS = 15_000   # 15s — for PDF/paper pages (fast path)
SCREENSHOT_TIMEOUT_NORMAL_MS = 30_000  # 30s — for regular pages
SCREENSHOT_TIMEOUT_LAST_RESORT_MS = 45_000  # 45s — last resort (was 120s, now 45s)


class StealthBrowserController:
    """
    Stealthy browser controller combining SeleniumBase CDP + Playwright.
    
    Features:
      - Automatic anti-bot evasion via SeleniumBase
      - Full Playwright API for automation
      - Built-in CAPTCHA solving capabilities
      - Undetectable by most anti-bot systems
      - Fast screenshot path for PDF/paper pages (no font-wait hang)
    """
    
    def __init__(self):
        self.browser: Browser = None
        self.context = None
        self.page: Page = None
        self._playwright: Playwright = None
        self.sb_driver = None  # SeleniumBase driver
        self._cdp_url = None
        
    async def start(self, headless: bool = True):
        """
        Launch stealthy browser using SeleniumBase cdp_driver (WebDriver-less mode) + Playwright.

        Production fix: detects the system Chromium binary via _find_chromium_binary()
        and passes it explicitly to SBBrowser.create() so SeleniumBase never tries to
        download a driver at runtime (which fails in locked-down Docker containers).

        Cloud Run / Docker fallback: if SeleniumBase CDP fails (common in restricted
        container environments like Cloud Run where /dev/shm is tiny and process
        sandboxing is limited), automatically falls back to Playwright-only mode
        with stealth patches applied. This ensures the browser always starts.
        """
        # ── Attempt 1: SeleniumBase CDP (maximum stealth) ──
        try:
            await self._start_seleniumbase_cdp(headless)
            return self
        except Exception as e:
            logger.warning(
                f"SeleniumBase CDP failed: {e}. "
                "Falling back to Playwright-only stealth mode (Cloud Run compatible)."
            )
            # Clean up any partial state from the failed attempt
            await self._cleanup_partial()

        # ── Attempt 2: Playwright-only with stealth patches (Cloud Run safe) ──
        try:
            await self._start_playwright_stealth(headless)
            return self
        except Exception as e2:
            logger.error(f"Playwright stealth fallback also failed: {e2}", exc_info=True)
            await self._cleanup_partial()
            raise

    async def _start_seleniumbase_cdp(self, headless: bool):
        """
        Primary launch path: SeleniumBase CDP + Playwright overlay.
        Maximum anti-bot evasion but requires a working Chrome/Chromium
        process with full CDP support (may fail on Cloud Run).
        """
        # Import SeleniumBase cdp_driver Browser class
        try:
            from seleniumbase.undetected.cdp_driver.browser import Browser as SBBrowser
        except ImportError:
            raise ImportError(
                "SeleniumBase not installed or cdp_driver not found. "
                "Install with: pip install seleniumbase"
            )

        # ── Locate Chromium binary (critical for Docker / production) ──
        chromium_path = _find_chromium_binary()
        if chromium_path:
            logger.info(f"Stealth browser: using Chromium at {chromium_path}")
        else:
            logger.warning(
                "Stealth browser: no Chromium binary found — "
                "SeleniumBase will attempt to auto-download (may fail in production). "
                "Set SB_BINARY_LOCATION env var or install chromium via apt."
            )

        logger.info("Starting SeleniumBase cdp_driver (WebDriver-less stealth browser)...")

        # PRODUCTION / DOCKER / CLOUD RUN CRITICAL FLAGS:
        create_kwargs: dict = {
            "headless": headless,
            "sandbox": False,
            "extra_chromium_args": [
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-setuid-sandbox",
                "--disable-software-rasterizer",
                "--disable-extensions",
                "--disable-background-networking",
                "--no-first-run",
                "--no-zygote",
                "--single-process",
                "--disable-features=VizDisplayCompositor",
                "--shm-size=128m",
            ],
        }
        if chromium_path:
            create_kwargs["browser_executable_path"] = chromium_path

        self.sb_driver = await SBBrowser.create(**create_kwargs)

        # Get the CDP endpoint URL
        self._cdp_url = self.sb_driver.websocket_url
        logger.info(f"CDP Stealth endpoint: {self._cdp_url}")

        # Connect Playwright to the running stealthy browser
        logger.info("Connecting Playwright to CDP Stealth browser...")
        self._playwright = await async_playwright().start()

        self.browser = await self._playwright.chromium.connect_over_cdp(
            endpoint_url=self._cdp_url
        )

        # Get the default context and page
        contexts = self.browser.contexts
        if contexts:
            self.context = contexts[0]
            pages = self.context.pages
            if pages:
                self.page = pages[0]
            else:
                self.page = await self.context.new_page()
        else:
            self.context = await self.browser.new_context()
            self.page = await self.context.new_page()

        # Set viewport
        await self.page.set_viewport_size({"width": 1280, "height": 800})

        # Handle new tabs
        self.context.on("page", self._on_page_created)

        logger.info("✅ CDP Stealth browser started (No WebDriver + Playwright)")
        logger.info(f"   - Headless: {headless}")
        logger.info(f"   - Binary:   {chromium_path or 'auto-detected by SeleniumBase'}")
        logger.info("   - Anti-bot evasion: MAXIMUM")

    async def _start_playwright_stealth(self, headless: bool):
        """
        Fallback launch path: Playwright-only with MAXIMUM stealth patches.
        Works reliably on Cloud Run / Docker / any restricted container.
        
        Enhanced anti-bot evasion via:
        - playwright-stealth patches
        - Comprehensive Chrome flags to mask automation signals
        - Deep JS overrides for navigator, WebGL, canvas, audio fingerprinting
        - Realistic browser profile (timezone, locale, screen, fonts)
        - Random human-like delays injected into page interactions

        CRITICAL: We do NOT pass executable_path here. Playwright's own bundled
        Chromium (installed via `playwright install chromium`) has proper headless
        support built in. The system /usr/bin/chromium (apt package) does NOT
        support Playwright's headless protocol and crashes with "Missing X server"
        on Cloud Run. Letting Playwright use its own binary fixes this.
        """
        logger.info("Starting Playwright MAXIMUM stealth browser (Cloud Run compatible)...")

        self._playwright = await async_playwright().start()

        # Comprehensive Chrome flags for maximum anti-detection
        stealth_args = [
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--disable-setuid-sandbox",
            "--disable-software-rasterizer",
            "--disable-extensions",
            "--disable-background-networking",
            "--no-first-run",
            "--no-zygote",
            "--single-process",
            "--disable-features=VizDisplayCompositor",
            # ── Anti-detection flags ──
            "--disable-blink-features=AutomationControlled",
            "--disable-infobars",
            "--disable-component-update",
            "--disable-default-apps",
            "--disable-domain-reliability",
            "--disable-features=AudioServiceOutOfProcess",
            "--disable-hang-monitor",
            "--disable-ipc-flooding-protection",
            "--disable-popup-blocking",
            "--disable-prompt-on-repost",
            "--disable-renderer-backgrounding",
            "--disable-sync",
            "--disable-translate",
            "--metrics-recording-only",
            "--no-default-browser-check",
            "--password-store=basic",
            "--use-mock-keychain",
            # ── Fingerprint masking ──
            "--disable-features=IsolateOrigins,site-per-process",
            "--flag-switches-begin",
            "--flag-switches-end",
            "--window-size=1280,800",
        ]

        launch_kwargs = {
            "headless": headless,
            "args": stealth_args,
        }

        self.browser = await self._playwright.chromium.launch(**launch_kwargs)

        # Realistic user agent — use a recent, common Chrome version
        import random
        _chrome_versions = [
            "131.0.6778.85", "131.0.6778.108", "130.0.6723.116",
            "130.0.6723.91", "129.0.6668.100", "128.0.6613.137",
        ]
        _chrome_ver = random.choice(_chrome_versions)
        _user_agent = (
            f"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            f"(KHTML, like Gecko) Chrome/{_chrome_ver} Safari/537.36"
        )

        self.context = await self.browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=_user_agent,
            java_script_enabled=True,
            locale="en-US",
            timezone_id="America/New_York",
            color_scheme="light",
            # Realistic screen dimensions
            screen={"width": 1920, "height": 1080},
            # Accept common headers
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "Sec-Ch-Ua": f'"Chromium";v="{_chrome_ver.split(".")[0]}", "Google Chrome";v="{_chrome_ver.split(".")[0]}", "Not?A_Brand";v="99"',
                "Sec-Ch-Ua-Mobile": "?0",
                "Sec-Ch-Ua-Platform": '"Linux"',
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Upgrade-Insecure-Requests": "1",
            },
        )

        # Apply playwright-stealth patches if available
        try:
            from playwright_stealth import stealth_async
            self.page = await self.context.new_page()
            await stealth_async(self.page)
            logger.info("   - playwright-stealth patches applied")
        except ImportError:
            self.page = await self.context.new_page()
            logger.info("   - playwright-stealth not available, using comprehensive JS evasion")

        # ── COMPREHENSIVE anti-detection JS overrides ──
        # These are injected into EVERY frame (including iframes like reCAPTCHA)
        await self.context.add_init_script("""
            // ═══ 1. Navigator overrides ═══
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            
            // Remove webdriver from prototype chain
            delete navigator.__proto__.webdriver;
            
            // Chrome runtime — must look like a real Chrome extension environment
            window.chrome = {
                runtime: {
                    onMessage: { addListener: function() {} },
                    sendMessage: function() {},
                    connect: function() { return { onMessage: { addListener: function() {} } }; },
                    PlatformOs: { MAC: 'mac', WIN: 'win', ANDROID: 'android', CROS: 'cros', LINUX: 'linux', OPENBSD: 'openbsd' },
                    PlatformArch: { ARM: 'arm', X86_32: 'x86-32', X86_64: 'x86-64', MIPS: 'mips', MIPS64: 'mips64' },
                },
                loadTimes: function() {
                    return {
                        requestTime: Date.now() / 1000 - Math.random() * 5,
                        startLoadTime: Date.now() / 1000 - Math.random() * 3,
                        commitLoadTime: Date.now() / 1000 - Math.random() * 2,
                        finishDocumentLoadTime: Date.now() / 1000 - Math.random(),
                        finishLoadTime: Date.now() / 1000,
                        firstPaintTime: Date.now() / 1000 - Math.random() * 0.5,
                        firstPaintAfterLoadTime: 0,
                        navigationType: 'Other',
                        wasFetchedViaSpdy: false,
                        wasNpnNegotiated: true,
                        npnNegotiatedProtocol: 'h2',
                        wasAlternateProtocolAvailable: false,
                        connectionInfo: 'h2',
                    };
                },
                csi: function() { return { pageT: Date.now(), startE: Date.now(), onloadT: Date.now() }; },
                app: {
                    isInstalled: false,
                    InstallState: { INSTALLED: 'installed', NOT_INSTALLED: 'not_installed' },
                    RunningState: { CANNOT_RUN: 'cannot_run', READY_TO_RUN: 'ready_to_run', RUNNING: 'running' },
                    getDetails: function() { return null; },
                    getIsInstalled: function() { return false; },
                },
            };
            
            // ═══ 2. Permissions API ═══
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) =>
                parameters.name === 'notifications'
                    ? Promise.resolve({ state: Notification.permission })
                    : originalQuery(parameters);
            
            // ═══ 3. Plugins — realistic Chrome plugin list ═══
            Object.defineProperty(navigator, 'plugins', {
                get: () => {
                    const plugins = [
                        { name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer', description: 'Portable Document Format' },
                        { name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai', description: '' },
                        { name: 'Native Client', filename: 'internal-nacl-plugin', description: '' },
                    ];
                    plugins.length = 3;
                    return plugins;
                },
            });
            
            // ═══ 4. Languages ═══
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en'],
            });
            Object.defineProperty(navigator, 'language', {
                get: () => 'en-US',
            });
            
            // ═══ 5. Platform ═══
            Object.defineProperty(navigator, 'platform', {
                get: () => 'Linux x86_64',
            });
            
            // ═══ 6. Hardware concurrency (realistic) ═══
            Object.defineProperty(navigator, 'hardwareConcurrency', {
                get: () => 4,
            });
            
            // ═══ 7. Device memory (realistic) ═══
            Object.defineProperty(navigator, 'deviceMemory', {
                get: () => 8,
            });
            
            // ═══ 8. Connection API ═══
            if (navigator.connection) {
                Object.defineProperty(navigator.connection, 'rtt', { get: () => 50 });
            }
            
            // ═══ 9. WebGL fingerprint masking ═══
            const getParameterOrig = WebGLRenderingContext.prototype.getParameter;
            WebGLRenderingContext.prototype.getParameter = function(parameter) {
                if (parameter === 37445) return 'Intel Inc.';
                if (parameter === 37446) return 'Intel Iris OpenGL Engine';
                return getParameterOrig.call(this, parameter);
            };
            
            // ═══ 10. Canvas fingerprint noise ═══
            const toDataURLOrig = HTMLCanvasElement.prototype.toDataURL;
            HTMLCanvasElement.prototype.toDataURL = function(type) {
                if (type === 'image/png' || type === undefined) {
                    const context = this.getContext('2d');
                    if (context) {
                        const imageData = context.getImageData(0, 0, this.width, this.height);
                        for (let i = 0; i < imageData.data.length; i += 4) {
                            // Add tiny noise to RGB channels (imperceptible but changes fingerprint)
                            imageData.data[i] = imageData.data[i] ^ (Math.random() > 0.99 ? 1 : 0);
                        }
                        context.putImageData(imageData, 0, 0);
                    }
                }
                return toDataURLOrig.apply(this, arguments);
            };
            
            // ═══ 11. Prevent iframe detection of automation ═══
            Object.defineProperty(document, 'hidden', { get: () => false });
            Object.defineProperty(document, 'visibilityState', { get: () => 'visible' });
            
            // ═══ 12. Notification constructor (some sites check this) ═══
            if (!window.Notification) {
                window.Notification = { permission: 'default' };
            }
            
            // ═══ 13. Remove automation-related properties ═══
            const automationProps = [
                '_phantom', '__nightmare', '_selenium', 'callPhantom',
                '__phantomas', 'Buffer', 'emit', 'spawn',
                'domAutomation', 'domAutomationController',
                '_Selenium_IDE_Recorder', '__webdriver_evaluate',
                '__selenium_evaluate', '__webdriver_script_function',
                '__webdriver_script_func', '__webdriver_script_fn',
                '__fxdriver_evaluate', '__driver_unwrapped',
                '__webdriver_unwrapped', '__driver_evaluate',
                '__selenium_unwrapped', '__fxdriver_unwrapped',
            ];
            automationProps.forEach(prop => {
                if (prop in window) {
                    try { delete window[prop]; } catch(e) {}
                }
            });
        """)

        await self.page.set_viewport_size({"width": 1280, "height": 800})

        # Handle new tabs
        self.context.on("page", self._on_page_created)

        logger.info("✅ Playwright MAXIMUM stealth browser started (Cloud Run compatible)")
        logger.info(f"   - Headless: {headless}")
        logger.info(f"   - User-Agent: Chrome/{_chrome_ver}")
        logger.info("   - Binary:   Playwright bundled Chromium")
        logger.info("   - Anti-bot evasion: MAXIMUM (stealth patches + deep JS overrides + fingerprint masking)")

    async def _cleanup_partial(self):
        """Clean up any partially initialized state from a failed start attempt."""
        try:
            if self.browser:
                await self.browser.close()
                self.browser = None
        except Exception:
            pass
        try:
            if self._playwright:
                await self._playwright.stop()
                self._playwright = None
        except Exception:
            pass
        try:
            if self.sb_driver:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.sb_driver.stop)
                self.sb_driver = None
        except Exception:
            pass
        self.context = None
        self.page = None
        self._cdp_url = None
    
    async def _on_page_created(self, page: Page):
        """Automatically switch focus to newly opened tabs."""
        logger.info(f"New tab created: {page.url} — switching Active Page focus.")
        await page.wait_for_load_state()
        self.page = page

    def _is_pdf_or_paper_page(self) -> bool:
        """
        Return True if the current page is a PDF or heavy academic paper page.
        These pages hang on font loading — we use the fast screenshot path for them.
        """
        if not self.page:
            return False
        url = self.page.url.lower()
        pdf_patterns = [".pdf", "/pdf/", "arxiv.org/abs/", "/article/", "/paper/",
                        "pmc/articles/", "europepmc.org", "semanticscholar.org/paper/"]
        return any(pat in url for pat in pdf_patterns)

    async def screenshot_b64(self, timeout_ms: int = None, retries: int = 3) -> str:
        """
        Capture current page as base64 PNG (in-memory, never written to disk).

        OPTIMIZED v2:
        - PDF/paper pages use FAST PATH: stop font loading, 1s static delay, 15s timeout.
          This eliminates the "waiting for fonts to load" hang seen in error logs.
        - Normal pages use standard 30s timeout with retry logic.
        - Last-resort timeout reduced from 120s → 45s to fail fast and recover.

        The root cause of the timeouts in error.txt was Playwright waiting for
        custom academic fonts (LaTeX/MathJax) to load before taking a screenshot.
        By calling window.stop() before capturing on PDF pages, we cut off that
        wait entirely and get a clean screenshot of whatever is already rendered.
        """
        is_pdf = self._is_pdf_or_paper_page()

        if timeout_ms is None:
            timeout_ms = SCREENSHOT_TIMEOUT_FAST_MS if is_pdf else SCREENSHOT_TIMEOUT_NORMAL_MS

        last_error = None

        for attempt in range(1, retries + 1):
            try:
                if is_pdf:
                    # ── FAST PATH for PDF/paper pages ──
                    # Stop any pending font/resource loading BEFORE screenshotting.
                    # This is the key fix: Playwright's screenshot hangs because it
                    # waits for fonts to finish loading. window.stop() cuts that off.
                    try:
                        await self.page.evaluate("window.stop()")
                    except Exception:
                        pass
                    # Brief static delay (1s) to let already-loaded content render
                    await asyncio.sleep(1.0)

                screenshot_bytes = await self.page.screenshot(
                    type="png",
                    timeout=timeout_ms,
                )
                return base64.b64encode(screenshot_bytes).decode("utf-8")

            except Exception as e:
                last_error = e
                logger.warning(f"Screenshot attempt {attempt}/{retries} failed: {e}")
                if attempt < retries:
                    await asyncio.sleep(1.0)
                    try:
                        await self.page.evaluate("window.stop()")
                        await asyncio.sleep(0.5)
                    except Exception:
                        pass

        # All retries exhausted — last-resort with reduced timeout (45s, not 120s)
        logger.error(f"All {retries} screenshot attempts failed. Trying last-resort capture...")
        try:
            await self.page.evaluate("window.stop()")
            await asyncio.sleep(1.0)
            screenshot_bytes = await self.page.screenshot(
                type="png",
                timeout=SCREENSHOT_TIMEOUT_LAST_RESORT_MS,
            )
            return base64.b64encode(screenshot_bytes).decode("utf-8")
        except Exception as e2:
            logger.error(f"Last-resort screenshot also failed: {e2}")
            raise last_error
    
    def is_on_captcha_page(self) -> bool:
        """
        Detect if the browser is currently on a CAPTCHA / bot-detection page.
        
        Checks for:
        - Google's /sorry/ CAPTCHA page (the most common case on Cloud Run)
        - Cloudflare challenge pages
        - Generic CAPTCHA indicators in the URL
        
        Returns True if CAPTCHA/block page is detected.
        """
        if not self.page:
            return False
        url = (self.page.url or "").lower()
        captcha_url_patterns = [
            "google.com/sorry",
            "/sorry/index",
            "recaptcha",
            "captcha",
            "challenge-platform",
            "challenges.cloudflare.com",
            "/cdn-cgi/challenge",
            "hcaptcha.com",
            "arkoselabs.com",
        ]
        return any(pattern in url for pattern in captcha_url_patterns)

    async def detect_captcha_in_page(self) -> bool:
        """
        CAPTCHA detection using URL analysis and page title only.
        
        Strictly avoids DOM/CSS selectors — uses only:
        1. URL pattern matching (is_on_captcha_page)
        2. document.title text check (single property, no selector)
        3. document.body.innerText substring check (no selector, read-only)
        
        Returns True if CAPTCHA is detected.
        """
        # Quick URL check first — covers Google /sorry/, Cloudflare challenge URLs
        if self.is_on_captcha_page():
            return True
        
        # Page title check — no DOM selector, just document.title property
        try:
            title_and_text = await self.page.evaluate("""
                () => {
                    const title = (document.title || '').toLowerCase();
                    // Only read the first 2000 chars of body text to avoid performance hit
                    const bodyText = document.body
                        ? (document.body.innerText || '').substring(0, 2000).toLowerCase()
                        : '';
                    const captchaSignals = [
                        'unusual traffic',
                        'not a robot',
                        'verify you are human',
                        'prove you are not a robot',
                        'automated queries',
                        'systems have detected unusual traffic',
                        'before you continue',
                        'checking your browser',
                    ];
                    const inTitle = captchaSignals.some(s => title.includes(s));
                    const inBody  = captchaSignals.some(s => bodyText.includes(s));
                    return inTitle || inBody;
                }
            """)
            return bool(title_and_text)
        except Exception as e:
            logger.debug(f"CAPTCHA page-text detection failed (non-fatal): {e}")
            return False

    async def solve_captcha(self, timeout: int = 30) -> bool:
        """
        Attempt to solve CAPTCHA using SeleniumBase's built-in solver.
        """
        try:
            logger.info("Attempting to solve CAPTCHA with SeleniumBase (CDP Mode)...")
            
            # cdp_driver has solve_captcha built-in
            # It's an async call in the newer CDP mode
            if hasattr(self.sb_driver, 'solve_captcha'):
                await self.sb_driver.solve_captcha()
                logger.info("✅ CAPTCHA solved successfully!")
                await asyncio.sleep(2)
                await self.page.wait_for_load_state("networkidle", timeout=10000)
                return True
            else:
                logger.warning("cdp_driver doesn't have solve_captcha method")
                return False
            
        except Exception as e:
            logger.error(f"Error during CAPTCHA solving: {e}")
            return False
    
    async def execute_action(self, action: dict) -> str:
        """
        Execute a single action from Gemini's output.
        Returns 'OK', 'DONE', or 'ERROR: <message>'.
        
        Supports all Google Computer Use predefined actions:
        - click / double_click / right_click / long_press
        - type (with optional press_enter)
        - scroll / scroll_at
        - navigate / go_back / go_forward
        - press (key combos)
        - hover_at / drag
        - wait / done

        Reference: https://ai.google.dev/gemini-api/docs/computer-use
        """
        act = action.get("action")
        x = action.get("x", 0)
        y = action.get("y", 0)
        text = action.get("text", "")

        try:
            if act == "click":
                await self.page.mouse.click(x, y, delay=100)
                
                # Adaptive wait: click may trigger navigation
                try:
                    await self.page.wait_for_load_state(
                        "domcontentloaded", timeout=3000
                    )
                except Exception:
                    pass  # Click didn't trigger navigation — that's fine
                
                await asyncio.sleep(1.0)

            elif act == "double_click":
                logger.info(f"Double-clicking at [{x}, {y}]")
                await self.page.mouse.dblclick(x, y, delay=80)
                await asyncio.sleep(0.5)

            elif act == "right_click":
                logger.info(f"Right-clicking at [{x}, {y}]")
                await self.page.mouse.click(x, y, button="right", delay=100)
                await asyncio.sleep(0.5)

            elif act == "long_press":
                duration = action.get("duration", 1)
                logger.info(f"Long-pressing at [{x}, {y}] for {duration}s")
                await self.page.mouse.move(x, y)
                await self.page.mouse.down()
                await asyncio.sleep(min(duration, 5))
                await self.page.mouse.up()
                await asyncio.sleep(0.3)

            elif act == "type":
                # ── Pure Vision / Coordinate-based Focus Strategy ──
                # No DOM traversal, no CSS selectors. Only:
                #   1. mouse.click(x, y)  — pixel-coordinate clicks
                #   2. activeElement.tagName — single property to verify click landed
                #
                # If the model's coordinates miss, we retry with coordinate
                # offsets (scanning upward, since search bars are typically
                # above the model's predicted point).

                _input_focused = False
                if "x" in action and "y" in action:
                    # Coordinate retry pattern: original point, then scan upward
                    # in increasing steps, then try some downward/lateral offsets.
                    _offsets = [
                        (0, 0),       # original coordinates
                        (0, -30),     # 30px above
                        (0, -60),     # 60px above
                        (0, -90),     # 90px above
                        (0, -120),    # 120px above
                        (0, 30),      # 30px below
                        (-50, -60),   # left + above
                        (50, -60),    # right + above
                    ]

                    for _dx, _dy in _offsets:
                        _cx = max(0, min(1280, x + _dx))
                        _cy = max(0, min(800, y + _dy))

                        logger.info(f"Trying click at [{_cx}, {_cy}] (offset [{_dx},{_dy}])...")
                        await self.page.mouse.click(_cx, _cy, delay=50)
                        await asyncio.sleep(0.3)

                        # Single property check: did an input/textarea get focus?
                        _focused_tag = await self.page.evaluate(
                            "() => document.activeElement?.tagName || 'NONE'"
                        )
                        _input_focused = _focused_tag in ("INPUT", "TEXTAREA")

                        if _input_focused:
                            logger.info(
                                f"✅ Input focused at [{_cx},{_cy}] "
                                f"(offset [{_dx},{_dy}] from model's [{x},{y}])"
                            )
                            break

                    if not _input_focused:
                        logger.warning(
                            f"All coordinate retries failed to focus an input. "
                            f"Last focused: {_focused_tag}. Will type anyway."
                        )

                # Only Ctrl+A if an input is actually focused — prevents whole-page select
                clear_before_type = action.get("clear", True)
                if clear_before_type and _input_focused:
                    await self.page.keyboard.press("Control+A")
                    await asyncio.sleep(0.1)
                    await self.page.keyboard.press("Backspace")
                    await asyncio.sleep(0.1)
                elif clear_before_type and not _input_focused:
                    logger.warning("Skipping Ctrl+A — no input focused (would select entire page)")

                # Slow Humanized Typing
                logger.info(f"Typing text: '{text}'")
                import random
                for char in text:
                    await self.page.keyboard.type(char)
                    await asyncio.sleep(random.uniform(0.03, 0.08))

                # Press Enter if requested
                press_enter = action.get("press_enter", False)
                if press_enter:
                    await asyncio.sleep(0.2)
                    await self.page.keyboard.press("Enter")
                    logger.info("Pressed Enter after typing")

                await asyncio.sleep(0.5) # Wait for text to appear and UI to react

            elif act == "scroll":
                delta = action.get("delta", 300)
                await self.page.mouse.wheel(0, delta)
                await asyncio.sleep(0.5)

            elif act == "navigate":
                try:
                    await self.page.goto(
                        text,
                        wait_until="networkidle",
                        timeout=15000,
                    )
                except Exception as e:
                    logger.warning(f"Navigation to '{text}' partial: {e}")
                    try:
                        await self.page.wait_for_load_state(
                            "domcontentloaded", timeout=5000
                        )
                    except Exception:
                        pass

            elif act == "press":
                def _normalise_key(k: str) -> str:
                    return k[0].upper() + k[1:] if k else k

                key = "+".join(_normalise_key(part) for part in text.split("+"))
                await self.page.keyboard.press(key)
                await asyncio.sleep(0.3)

            elif act == "wait":
                seconds = action.get("seconds", 1)
                await asyncio.sleep(min(seconds, 10))

            elif act == "done":
                return "DONE"
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Google Computer Use Actions
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            
            elif act == "hover_at":
                # Hover over element to reveal dropdowns/menus
                logger.info(f"Hovering at [{x}, {y}]")
                await self.page.mouse.move(x, y)
                await asyncio.sleep(0.5)  # Wait for hover effects (dropdowns, tooltips)
            
            elif act == "go_back":
                # Browser back navigation
                logger.info("Navigating back")
                try:
                    await self.page.go_back(wait_until="domcontentloaded", timeout=5000)
                except Exception as e:
                    logger.warning(f"Go back failed: {e}")
                    # Return ERROR so the vision loop sends accurate feedback to Gemini
                    # instead of falsely reporting success and corrupting its page model.
                    return f"ERROR: go_back failed — {e}. Use navigate() with the previous URL instead."
                await asyncio.sleep(1.0)
            
            elif act == "go_forward":
                # Browser forward navigation
                logger.info("Navigating forward")
                try:
                    await self.page.go_forward(wait_until="domcontentloaded", timeout=5000)
                except Exception as e:
                    logger.warning(f"Go forward failed: {e}")
                    return f"ERROR: go_forward failed — {e}. Use navigate() with the target URL instead."
                await asyncio.sleep(1.0)
            
            elif act == "scroll_at":
                # Scroll at specific element coordinates
                direction = action.get("direction", "down")
                magnitude = action.get("magnitude", 300)
                
                logger.info(f"Scrolling {direction} at [{x}, {y}] with magnitude {magnitude}")
                
                # Move mouse to the element first
                await self.page.mouse.move(x, y)
                await asyncio.sleep(0.1)
                
                # Determine scroll direction
                if direction == "down":
                    await self.page.mouse.wheel(0, magnitude)
                elif direction == "up":
                    await self.page.mouse.wheel(0, -magnitude)
                elif direction == "right":
                    await self.page.mouse.wheel(magnitude, 0)
                elif direction == "left":
                    await self.page.mouse.wheel(-magnitude, 0)
                else:
                    return f"ERROR: Invalid scroll direction '{direction}'"
                
                await asyncio.sleep(0.5)

            elif act == "drag":
                # Mouse drag from start to end coordinates
                start_x = action.get("start_x", x)
                start_y = action.get("start_y", y)
                end_x = action.get("end_x", 0)
                end_y = action.get("end_y", 0)
                logger.info(f"Dragging from [{start_x}, {start_y}] to [{end_x}, {end_y}]")
                await self.page.mouse.move(start_x, start_y)
                await self.page.mouse.down()
                await asyncio.sleep(0.1)
                # Smooth drag with intermediate steps
                steps = 10
                for i in range(1, steps + 1):
                    ix = start_x + (end_x - start_x) * i / steps
                    iy = start_y + (end_y - start_y) * i / steps
                    await self.page.mouse.move(ix, iy)
                    await asyncio.sleep(0.02)
                await self.page.mouse.up()
                await asyncio.sleep(0.3)

            # ── Safe Jump: large viewport-based scroll ──
            # Used by the optimized paper-reading strategy.
            # Jumps 90% of viewport height (720px) in one action.
            elif act == "safe_jump":
                direction = action.get("direction", "down")
                # 90% of 800px viewport = 720px jump with 80px overlap
                jump_px = action.get("jump_px", 720)
                delta = jump_px if direction == "down" else -jump_px
                logger.info(f"Safe jump {direction} by {jump_px}px")
                await self.page.mouse.wheel(0, delta)
                # Brief static delay — no font-wait, no stability check
                await asyncio.sleep(1.2)

            else:
                return f"ERROR: Unknown action '{act}'"

        except Exception as e:
            error_msg = f"ERROR: {act} failed — {e}"
            logger.error(error_msg)
            return error_msg

        return "OK"
    
    async def extract_page_text(self, max_length: int = 50000) -> str:
        """
        Extract visible text content from the current page using DOM APIs.
        
        This is used to read paper content when the agent navigates to a paper page.
        Extracts text from the body, stripping navigation, headers, footers, and scripts.
        
        Args:
            max_length: Maximum characters to extract (default 50K to avoid memory issues)
            
        Returns:
            Extracted text content, or empty string on failure.
        """
        try:
            text = await self.page.evaluate("""
                () => {
                    // Remove non-content elements
                    const removeSelectors = [
                        'nav', 'header', 'footer', 'script', 'style', 'noscript',
                        '[role="navigation"]', '[role="banner"]', '[role="contentinfo"]',
                        '.cookie-banner', '.consent-banner', '#cookie-notice',
                        '.sidebar', '.nav', '.menu', '.toolbar', '.ad', '.advertisement'
                    ];
                    
                    // Clone body to avoid modifying the actual page
                    const clone = document.body.cloneNode(true);
                    removeSelectors.forEach(sel => {
                        clone.querySelectorAll(sel).forEach(el => el.remove());
                    });
                    
                    // Get text content
                    let text = clone.innerText || clone.textContent || '';
                    
                    // Clean up whitespace
                    text = text.replace(/\\n{3,}/g, '\\n\\n').trim();
                    
                    return text;
                }
            """)
            
            if text and len(text) > max_length:
                text = text[:max_length] + "\n\n[... content truncated ...]"
            
            logger.info(f"Extracted {len(text)} chars of page text from {self.page.url}")
            return text or ""
            
        except Exception as e:
            logger.warning(f"Failed to extract page text: {e}")
            return ""

    async def get_scroll_position(self) -> dict:
        """
        Get current scroll position and total page height.
        Used to detect when the agent has reached the bottom of a paper.
        Returns: {scrollY, scrollHeight, clientHeight, atBottom, progress_pct}
        """
        try:
            pos = await self.page.evaluate("""
                () => {
                    const scrollY = window.scrollY || document.documentElement.scrollTop || 0;
                    const scrollHeight = document.documentElement.scrollHeight || document.body.scrollHeight || 0;
                    const clientHeight = window.innerHeight || document.documentElement.clientHeight || 800;
                    const atBottom = (scrollY + clientHeight) >= (scrollHeight - 50);
                    const progress = scrollHeight > clientHeight
                        ? Math.round((scrollY / (scrollHeight - clientHeight)) * 100)
                        : 100;
                    return {
                        scrollY: Math.round(scrollY),
                        scrollHeight: Math.round(scrollHeight),
                        clientHeight: Math.round(clientHeight),
                        atBottom: atBottom,
                        progress_pct: Math.min(progress, 100)
                    };
                }
            """)
            return pos or {"scrollY": 0, "scrollHeight": 0, "clientHeight": 800, "atBottom": False, "progress_pct": 0}
        except Exception as e:
            logger.debug(f"get_scroll_position failed (non-fatal): {e}")
            return {"scrollY": 0, "scrollHeight": 0, "clientHeight": 800, "atBottom": False, "progress_pct": 0}

    async def extract_page_metadata(self) -> dict:
        """
        Extract metadata (title, description, URL) from the current page.
        Useful for identifying what paper/article the browser is currently viewing.
        """
        try:
            metadata = await self.page.evaluate("""
                () => {
                    const getMeta = (name) => {
                        const el = document.querySelector(`meta[name="${name}"], meta[property="${name}"]`);
                        return el ? el.getAttribute('content') : '';
                    };
                    return {
                        title: document.title || '',
                        url: window.location.href,
                        description: getMeta('description') || getMeta('og:description') || '',
                        authors: getMeta('citation_author') || getMeta('author') || '',
                        doi: getMeta('citation_doi') || getMeta('DOI') || '',
                        published_date: getMeta('citation_publication_date') || getMeta('citation_date') || '',
                        journal: getMeta('citation_journal_title') || '',
                        pdf_url: (document.querySelector('a[href*=".pdf"]') || {}).href || '',
                        abstract_text: (document.querySelector('.abstract, #abstract, [class*="abstract"]') || {}).innerText || '',
                    };
                }
            """)
            return metadata or {}
        except Exception as e:
            logger.warning(f"Failed to extract page metadata: {e}")
            return {}

    async def inject_grid(self, cell_size=100):
        """Draw coordinate grid overlay for vision-based automation."""
        try:
            script = f"""
            (() => {{
                if (document.querySelector('[data-vision-grid]')) return;
                const prevFocus = document.activeElement;

                const canvas = document.createElement('canvas');
                canvas.setAttribute('data-vision-grid', '1');
                canvas.style.position  = 'fixed';
                canvas.style.top       = '0';
                canvas.style.left      = '0';
                canvas.style.width     = '100vw';
                canvas.style.height    = '100vh';
                canvas.style.pointerEvents = 'none';
                canvas.style.zIndex    = '999999';

                const w = window.innerWidth;
                const h = window.innerHeight;
                canvas.width  = w;
                canvas.height = h;

                const ctx  = canvas.getContext('2d');
                const size = {cell_size};

                for (let yp = size; yp < h; yp += size) {{
                    ctx.beginPath();
                    ctx.setLineDash([5, 5]);
                    ctx.strokeStyle = 'rgba(255, 0, 0, 0.55)';
                    ctx.lineWidth   = 1;
                    ctx.moveTo(0, yp);
                    ctx.lineTo(w, yp);
                    ctx.stroke();

                    ctx.font         = 'bold 13px monospace';
                    ctx.fillStyle    = 'red';
                    ctx.strokeStyle  = 'white';
                    ctx.lineWidth    = 3;
                    ctx.setLineDash([]);
                    ctx.strokeText(`Y:${{yp}}`, 5, yp - 4);
                    ctx.fillText(`Y:${{yp}}`, 5, yp - 4);
                }}

                for (let xp = size; xp < w; xp += size) {{
                    ctx.beginPath();
                    ctx.setLineDash([5, 5]);
                    ctx.strokeStyle = 'rgba(0, 0, 255, 0.55)';
                    ctx.lineWidth   = 1;
                    ctx.moveTo(xp, 0);
                    ctx.lineTo(xp, h);
                    ctx.stroke();

                    ctx.font         = 'bold 13px monospace';
                    ctx.fillStyle    = 'blue';
                    ctx.strokeStyle  = 'white';
                    ctx.lineWidth    = 3;
                    ctx.setLineDash([]);
                    ctx.strokeText(`X:${{xp}}`, xp + 4, 15);
                    ctx.fillText(`X:${{xp}}`, xp + 4, 15);
                }}

                document.body.appendChild(canvas);

                if (prevFocus && prevFocus !== document.body &&
                        typeof prevFocus.focus === 'function') {{
                    prevFocus.focus();
                }}
            }})();
            """
            await self.page.evaluate(script)
        except Exception as e:
            logger.warning(f"Failed to inject grid: {e}")
    
    async def remove_grid(self):
        """Remove the coordinate grid canvas overlay."""
        try:
            await self.page.evaluate("""
                const canvas = document.querySelector('[data-vision-grid]');
                if (canvas) canvas.remove();
            """)
        except Exception:
            pass
    
    async def wait_for_visual_stability(
        self,
        max_wait_sec: int = 5,
        min_stable_frames: int = 3,
        change_area_threshold: int = 500,
    ) -> bool:
        """
        Wait until the page is visually stable (no significant changes).

        OPTIMIZED v2:
        - max_wait_sec reduced from 10s → 5s (was causing 10s+ delays per step)
        - PDF/paper pages skip stability check entirely (use static 1s delay instead)
          because PDF rendering never truly "stabilizes" — fonts keep loading.
        - This directly fixes the "Visual stability check timed out" warnings in error.txt
        """
        # Fast path: PDF/paper pages — skip stability polling, use static delay
        if self._is_pdf_or_paper_page():
            await asyncio.sleep(1.0)
            return True

        start = time.time()
        prev_img = None
        stable_count = 0

        while time.time() - start < max_wait_sec:
            try:
                b64 = await self.screenshot_b64()
                img_data = base64.b64decode(b64)
                curr_img = Image.open(BytesIO(img_data)).convert("RGB")

                if prev_img:
                    diff = ImageChops.difference(prev_img, curr_img)
                    bbox = diff.getbbox()

                    significant = False
                    if bbox:
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        if area > change_area_threshold:
                            significant = True
                            logger.debug(f"Page still rendering — changed area: {area}px²")

                    if not significant:
                        stable_count += 1
                        if stable_count >= min_stable_frames:
                            logger.info(
                                f"Page is visually stable ({min_stable_frames} consecutive frames)."
                            )
                            return True
                    else:
                        stable_count = 0

                prev_img = curr_img
                await asyncio.sleep(0.3)

            except Exception as e:
                logger.warning(f"Error during visual stability check: {e}")
                await asyncio.sleep(0.3)

        logger.warning("Visual stability check timed out.")
        return False
    
    async def close(self):
        """Shut down browser and cleanup."""
        try:
            if self.browser:
                await self.browser.close()
        except Exception as e:
            logger.warning(f"Error closing Playwright browser: {e}")
        
        try:
            if self._playwright:
                await self._playwright.stop()
        except Exception as e:
            logger.warning(f"Error stopping Playwright: {e}")
        
        try:
            if self.sb_driver:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.sb_driver.stop)
        except Exception as e:
            logger.warning(f"Error closing SeleniumBase driver: {e}")
        
        logger.info("Stealth browser closed")
