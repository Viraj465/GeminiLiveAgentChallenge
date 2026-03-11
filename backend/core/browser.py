"""
browser.py — Playwright-based headless browser controller.

Features:
  - Async Playwright with headless Chromium
  - Base64 screenshot capture (in-memory only)
  - Action executor with adaptive waits per action type
  - Structured error returns (never throws)
"""

import asyncio
import base64
import logging
import sys
import threading
import time
from io import BytesIO
from PIL import Image, ImageChops
from playwright.async_api import async_playwright, Page, Browser
from playwright.sync_api import sync_playwright

try:
    from playwright_stealth import stealth_async
except ImportError:
    stealth_async = None

logger = logging.getLogger(__name__)


def _start_playwright_sync():
    """
    Start Playwright's server process using the sync API on a dedicated thread.
    This works around the Windows SelectorEventLoop limitation where
    asyncio.create_subprocess_exec raises NotImplementedError.
    Returns the sync Playwright connection object.
    """
    return sync_playwright().start()


class BrowserController:
    def __init__(self):
        self.browser: Browser = None
        self.context = None
        self.page: Page = None
        self._playwright = None
        self._sync_playwright = None

    async def start(self):
        """Launch headless Chromium and open a blank page."""
        try:
            self._playwright = await async_playwright().start()
        except NotImplementedError:
            # Windows + uvicorn --reload uses SelectorEventLoop which can't
            # spawn subprocesses. Fall back to sync Playwright started in a
            # thread with subprocess support.
            logger.warning(
                "async_playwright failed (SelectorEventLoop on Windows) — "
                "falling back to sync_playwright in thread"
            )
            loop = asyncio.get_event_loop()
            self._sync_playwright = await loop.run_in_executor(
                None, _start_playwright_sync
            )
            # Wrap the sync browser in async-compatible interface
            self._playwright = self._sync_playwright

        self.browser = await self._playwright.chromium.launch(
            headless=False,  # True for Cloud Run, False for local debug
            args=[
                "--no-sandbox", 
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled"
            ]
        )
        self.context = await self.browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        
        # Multi-tab handler
        self.context.on("page", self._on_page_created)
        
        self.page = await self.context.new_page()
        if stealth_async:
            await stealth_async(self.page)
        logger.info("Browser started (1280×800)")
        return self

    async def _on_page_created(self, page: Page):
        """Automatically switch focus to newly opened tabs."""
        logger.info(f"New tab created: {page.url} — switching Active Page focus.")
        if stealth_async:
            await stealth_async(page)
        await page.wait_for_load_state()
        self.page = page

    async def screenshot_b64(self) -> str:
        """Capture current page as base64 PNG (in-memory, never written to disk)."""
        screenshot_bytes = await self.page.screenshot(type="png")
        return base64.b64encode(screenshot_bytes).decode("utf-8")

    async def execute_action(self, action: dict) -> str:
        """
        Execute a single action from Gemini's output.
        Returns 'OK', 'DONE', or 'ERROR: <message>'.
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
                await asyncio.sleep(0.3)

            elif act == "type":
                await self.page.keyboard.type(text, delay=100) # Increased delay for realism
                await self.page.keyboard.press("Enter")
                await asyncio.sleep(0.3)

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
                    # Navigation may partially load — log but don't fail
                    logger.warning(f"Navigation to '{text}' partial: {e}")
                    try:
                        await self.page.wait_for_load_state(
                            "domcontentloaded", timeout=5000
                        )
                    except Exception:
                        pass

            elif act == "press":
                # Playwright expects PascalCase key names e.g. "Enter", "Tab", "Escape"
                key = text
                if len(key) > 1 and key[0].islower():
                    key = key.capitalize()
                
                await self.page.keyboard.press(key)  # e.g. "Enter"
                await asyncio.sleep(0.3)

            elif act == "wait":
                seconds = action.get("seconds", 1)
                await asyncio.sleep(min(seconds, 10))  # Cap at 10s

            elif act == "done":
                return "DONE"

            else:
                return f"ERROR: Unknown action '{act}'"

        except Exception as e:
            error_msg = f"ERROR: {act} failed — {e}"
            logger.error(error_msg)
            return error_msg

        return "OK"

    async def inject_grid(self, cell_size=100):
        """Draws a responsive coordinate grid directly into the page."""
        try:
            script = f"""
            (() => {{
                if (document.getElementById('__vision_grid_overlay__')) return;
                const overlay = document.createElement('div');
                overlay.id = '__vision_grid_overlay__';
                overlay.style.position = 'fixed';
                overlay.style.top = '0';
                overlay.style.left = '0';
                overlay.style.width = '100vw';
                overlay.style.height = '100vh';
                overlay.style.pointerEvents = 'none'; // Clicks pass through
                overlay.style.zIndex = '999999';
                
                // SVG Grid
                const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
                svg.setAttribute('width', '100%');
                svg.setAttribute('height', '100%');
                svg.style.position = 'absolute';
                svg.style.top = '0';
                svg.style.left = '0';
                
                const w = window.innerWidth;
                const h = window.innerHeight;
                const size = {cell_size};
                
                // Draw horizontal lines & labels
                for (let y = size; y < h; y += size) {{
                    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    line.setAttribute('x1', '0');
                    line.setAttribute('y1', y);
                    line.setAttribute('x2', w);
                    line.setAttribute('y2', y);
                    line.setAttribute('stroke', 'rgba(255, 0, 0, 0.5)');
                    line.setAttribute('stroke-dasharray', '5,5');
                    svg.appendChild(line);
                    
                    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    text.setAttribute('x', 5);
                    text.setAttribute('y', y - 5);
                    text.setAttribute('fill', 'red');
                    text.setAttribute('font-size', '14px');
                    text.setAttribute('font-weight', 'bold');
                    text.setAttribute('text-shadow', '1px 1px 0 #fff');
                    text.textContent = `Y:${{y}}`;
                    svg.appendChild(text);
                }}
                
                // Draw vertical lines & labels
                for (let x = size; x < w; x += size) {{
                    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    line.setAttribute('x1', x);
                    line.setAttribute('y1', '0');
                    line.setAttribute('x2', x);
                    line.setAttribute('y2', h);
                    line.setAttribute('stroke', 'rgba(0, 0, 255, 0.5)');
                    line.setAttribute('stroke-dasharray', '5,5');
                    svg.appendChild(line);
                    
                    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    text.setAttribute('x', x + 5);
                    text.setAttribute('y', 15);
                    text.setAttribute('fill', 'blue');
                    text.setAttribute('font-size', '14px');
                    text.setAttribute('font-weight', 'bold');
                    text.setAttribute('text-shadow', '1px 1px 0 #fff');
                    text.textContent = `X:${{x}}`;
                    svg.appendChild(text);
                }}
                
                overlay.appendChild(svg);
                document.body.appendChild(overlay);
            }})();
            """
            await self.page.evaluate(script)
        except Exception as e:
            logger.warning(f"Failed to inject grid: {e}")

    async def remove_grid(self):
        """Removes the coordinate grid overlay."""
        try:
            await self.page.evaluate("""
                const overlay = document.getElementById('__vision_grid_overlay__');
                if (overlay) overlay.remove();
            """)
        except Exception:
            pass
            
    async def wait_for_visual_stability(self, max_wait_sec=10) -> bool:
        """
        Takes screenshots in a loop until two consecutive screenshots 
        are visually identical, indicating the page has stopped rendering.
        """
        start = time.time()
        prev_img = None
        
        while time.time() - start < max_wait_sec:
            try:
                b64 = await self.screenshot_b64()
                img_data = base64.b64decode(b64)
                curr_img = Image.open(BytesIO(img_data)).convert('RGB')
                
                if prev_img:
                    # Calculate difference
                    diff = ImageChops.difference(prev_img, curr_img)
                    bbox = diff.getbbox()
                    
                    if not bbox:
                        logger.info("Page is visually stable.")
                        return True
                    else:
                        logger.debug("Page is still rendering visually...")
                
                prev_img = curr_img
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.warning(f"Error during visual stability check: {e}")
                await asyncio.sleep(0.5)
                
        logger.warning("Visual stability check timed out.")
        return False

    async def close(self):
        """Shut down browser and Playwright."""
        if self.browser:
            await self.browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info("Browser closed")