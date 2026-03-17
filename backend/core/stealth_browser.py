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
        """
        try:
            # Import SeleniumBase cdp_driver Browser class
            try:
                from seleniumbase.undetected.cdp_driver.browser import Browser as SBBrowser
            except ImportError:
                logger.error("SeleniumBase not installed or cdp_driver not found. Install with: pip install seleniumbase")
                raise ImportError("SeleniumBase is required for stealth mode")

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

            # Step 1: Launch SeleniumBase browser in CDP mode (no WebDriver/Chromedriver)
            # This is the most stealthy mode available.
            # Browser.create() is the correct async API in SeleniumBase 4.x
            # Pass browser_executable_path when we have a known binary so SeleniumBase
            # doesn't try to download Chrome at runtime inside the container.
            create_kwargs: dict = {"headless": headless}
            if chromium_path:
                create_kwargs["browser_executable_path"] = chromium_path

            self.sb_driver = await SBBrowser.create(**create_kwargs)

            # Step 2: Get the CDP endpoint URL
            # websocket_url is a property returning the webSocketDebuggerUrl
            self._cdp_url = self.sb_driver.websocket_url
            logger.info(f"CDP Stealth endpoint: {self._cdp_url}")

            # Step 3: Connect Playwright to the running stealthy browser
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

            return self

        except Exception as e:
            logger.error(f"Failed to start CDP stealth browser: {e}", exc_info=True)
            await self.close()
            raise
    
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
