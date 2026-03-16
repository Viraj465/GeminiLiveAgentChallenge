"""
browser.py — Playwright-based headless browser controller.

Features:
  - Async Playwright connecting via CDP to SeleniumBase (Stealth Mode)
  - Base64 screenshot capture (in-memory only)
  - Action executor with adaptive waits per action type
  - Structured error returns (never throws)
"""

import asyncio
import base64
import logging
import time
from io import BytesIO
from PIL import Image, ImageChops
from playwright.async_api import async_playwright, Page, Browser
from playwright.sync_api import sync_playwright

try:
    from playwright_stealth import stealth_async
except ImportError:
    stealth_async = None

try:
    from seleniumbase import Driver as SBDriver
except ImportError:
    SBDriver = None

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
        self.sb_driver = None

    async def start(self):
        """Launch stealthy Chromium via Playwright."""
        try:
            self._playwright = await async_playwright().start()
            is_async = True
        except NotImplementedError:
            # Windows + uvicorn --reload uses SelectorEventLoop which can't
            # spawn subprocesses. Fall back to sync Playwright started in a
            # thread with subprocess support.
            logger.warning(
                "async_playwright failed (SelectorEventLoop on Windows) — "
                "falling back to sync_playwright in thread"
            )
            loop = asyncio.get_running_loop()
            self._sync_playwright = await loop.run_in_executor(
                None, _start_playwright_sync
            )
            self._playwright = self._sync_playwright
            is_async = False

        # ---------------------------------------------------------
        # Launch Chromium via Playwright (with stealth if available)
        # ---------------------------------------------------------
        logger.info("Starting Playwright browser...")
        
        if is_async:
            self.browser = await self._playwright.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled"]
            )
            self.context = await self.browser.new_context()
            self.page = await self.context.new_page()
            await self.page.set_viewport_size({"width": 1280, "height": 800})
        else:
            self.browser = self._playwright.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled"]
            )
            self.context = self.browser.new_context()
            self.page = self.context.new_page()
            self.page.set_viewport_size({"width": 1280, "height": 800})

        # Multi-tab handler
        self.context.on("page", self._on_page_created)
        
        if stealth_async and is_async:
            await stealth_async(self.page)
            
        logger.info("Browser started (Stealth Mode, 1280×800)")
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

        Supports all Google Computer Use predefined actions.
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
                # Vision-pure type handler — zero DOM/CSS selectors.
                if "x" in action and "y" in action:
                    await self.page.mouse.click(x, y, delay=100)
                    await asyncio.sleep(0.3)   # let focus settle

                clear_before_type = action.get("clear", True)
                if clear_before_type:
                    await self.page.keyboard.press("Control+A")
                    await asyncio.sleep(0.05)
                    await self.page.keyboard.press("Backspace")
                    await asyncio.sleep(0.05)

                await self.page.keyboard.type(text, delay=50)
                await asyncio.sleep(0.3)

                # Press Enter if requested
                press_enter = action.get("press_enter", False)
                if press_enter:
                    await asyncio.sleep(0.2)
                    await self.page.keyboard.press("Enter")
                    logger.info("Pressed Enter after typing")

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
                logger.info(f"Hovering at [{x}, {y}]")
                await self.page.mouse.move(x, y)
                await asyncio.sleep(0.5)
            
            elif act == "go_back":
                logger.info("Navigating back")
                try:
                    await self.page.go_back(wait_until="domcontentloaded", timeout=5000)
                except Exception as e:
                    logger.warning(f"Go back failed: {e}")
                await asyncio.sleep(1.0)
            
            elif act == "go_forward":
                logger.info("Navigating forward")
                try:
                    await self.page.go_forward(wait_until="domcontentloaded", timeout=5000)
                except Exception as e:
                    logger.warning(f"Go forward failed: {e}")
                await asyncio.sleep(1.0)
            
            elif act == "scroll_at":
                direction = action.get("direction", "down")
                magnitude = action.get("magnitude", 300)
                
                logger.info(f"Scrolling {direction} at [{x}, {y}] with magnitude {magnitude}")
                await self.page.mouse.move(x, y)
                await asyncio.sleep(0.1)
                
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
                start_x = action.get("start_x", x)
                start_y = action.get("start_y", y)
                end_x = action.get("end_x", 0)
                end_y = action.get("end_y", 0)
                logger.info(f"Dragging from [{start_x}, {start_y}] to [{end_x}, {end_y}]")
                await self.page.mouse.move(start_x, start_y)
                await self.page.mouse.down()
                await asyncio.sleep(0.1)
                steps = 10
                for i in range(1, steps + 1):
                    ix = start_x + (end_x - start_x) * i / steps
                    iy = start_y + (end_y - start_y) * i / steps
                    await self.page.mouse.move(ix, iy)
                    await asyncio.sleep(0.02)
                await self.page.mouse.up()
                await asyncio.sleep(0.3)

            else:
                return f"ERROR: Unknown action '{act}'"

        except Exception as e:
            error_msg = f"ERROR: {act} failed — {e}"
            logger.error(error_msg)
            return error_msg

        return "OK"

    async def inject_grid(self, cell_size=100):
        """
        Draws a responsive coordinate grid directly into the page.

        Uses a canvas element painted entirely via the 2D drawing API — no CSS
        class/style queries, no DOM selectors for finding page elements.

        Focus-safe: records which element had focus before the canvas is
        appended and restores it afterward, so a focused search bar is never
        silently blurred between a vision-loop step and its screenshot.
        """
        try:
            script = f"""
            (() => {{
                // Guard: skip if already injected (data-attr, no getElementById)
                if (document.querySelector('[data-vision-grid]')) return;

                // Save focused element BEFORE any DOM mutation
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

                // Horizontal grid lines + Y labels
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

                // Vertical grid lines + X labels
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

                // Restore focus — appendChild can blur the active element in
                // some Chromium builds, which would silently lose the typed text
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
        """Removes the coordinate grid canvas overlay."""
        try:
            await self.page.evaluate("""
                const canvas = document.querySelector('[data-vision-grid]');
                if (canvas) canvas.remove();
            """)
        except Exception:
            pass
            
    async def wait_for_visual_stability(
        self,
        max_wait_sec: int = 10,
        min_stable_frames: int = 3,
        change_area_threshold: int = 500,
    ) -> bool:
        """
        Takes screenshots until N consecutive frames are visually stable.

        Ignores micro-changes whose bounding-box area is smaller than
        `change_area_threshold` pixels² so that a blinking text cursor
        (typically ~2×14 = 28 px²) or a favicon animation never prevents
        stability from being declared.  A real page transition changes
        thousands of pixels and always exceeds the threshold.
        """
        start = time.time()
        prev_img = None
        stable_count = 0

        while time.time() - start < max_wait_sec:
            try:
                b64      = await self.screenshot_b64()
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
        """Shut down browser and Playwright."""
        if self.browser:
            await self.browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info("Browser closed")