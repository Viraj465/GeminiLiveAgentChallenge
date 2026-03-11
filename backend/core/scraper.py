"""
scraper.py — Playwright-based academic scraper to gather PDFs.

Features:
  - Takes a topic string
  - Searches Google Scholar
  - Extracts Title, Authors, and PDF link for the top 5 results
  - Bypasses Windows async loop limitations via _start_playwright_sync thread routing
"""

import asyncio
import logging
import json
import base64
from playwright.async_api import async_playwright, Page, Browser
from playwright.sync_api import sync_playwright
from google import genai
from google.genai import types
from config import settings

logger = logging.getLogger(__name__)


def _start_playwright_sync():
    """
    Start Playwright's server process using the sync API on a dedicated thread.
    This works around the Windows SelectorEventLoop limitation where
    asyncio.create_subprocess_exec raises NotImplementedError.
    Returns the sync Playwright connection object.
    """
    return sync_playwright().start()


async def scrape_google_scholar(topic: str, max_results: int = 5) -> dict:
    """
    ADK Tool: Searches Google Scholar for a given topic and extracts the top results.
    Takes a 'topic' string.
    Returns a dict with 'status' and a 'data' list containing structured paper dictionaries.
    """
    logger.info(f"Scraper started for topic: '{topic}'")
    
    _playwright = None
    _sync_playwright = None
    browser = None

    try:
        try:
            _playwright = await async_playwright().start()
        except NotImplementedError:
            # Windows fallback
            logger.warning("async_playwright failed (SelectorEventLoop) — falling back to sync_playwright scraper in thread")
            loop = asyncio.get_event_loop()
            _sync_playwright = await loop.run_in_executor(None, _start_playwright_sync)
            _playwright = _sync_playwright

        browser = await _playwright.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox", 
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled"
            ]
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        # Try to avoid detection
        try:
            from playwright_stealth import stealth_async
            await stealth_async(page)
        except ImportError:
            pass

        # 1. Navigate to Google Scholar
        search_url = f"https://scholar.google.com/scholar?q={topic.replace(' ', '+')}"
        logger.info(f"Navigating to {search_url}")
        await page.goto(search_url, wait_until="domcontentloaded", timeout=15000)
        
        # Give it a second to render
        await asyncio.sleep(2)

        # 2. Extract Results Visually (Rule 2: Never use DOM Access)
        papers = []
        
        # Take a screenshot
        screenshot_bytes = await page.screenshot(type="png", full_page=True)
        screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
        
        # Pass to Gemini Vision for extraction via Vertex AI
        from core.vision_loop import _get_client
        try:
            client = _get_client()
        except Exception as e:
            raise EnvironmentError(f"Failed to initialize Vertex AI client: {e}")
            
        model_name = settings.GOOGLE_VISION_MODEL if hasattr(settings, "GOOGLE_VISION_MODEL") else "gemini-2.5-flash"
        
        prompt = f"""
        Look at this screenshot of a Google Scholar search results page for '{topic}'.
        Extract the top {max_results} academic papers shown on the page.
        For each paper, you must find:
        1. "title": The title of the paper.
        2. "authors": The authors and publication info snippet shown in green.
        3. "pdf_url": The direct URL to the PDF file, usually found clicking the [PDF] link on the right side. Only include the URL if it appears directly on the screen or can be deduced from the visual layout.
        
        Return ONLY a raw JSON array of dictionaries with theses keys. If no PDF url is visible, omit that paper.
        """
        
        logger.info("Sending screenshot to Gemini for visual extraction...")
        
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=model_name,
                contents=[
                    types.Content(role="user", parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(data=screenshot_bytes, mime_type="image/png")
                    ])
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            ),
            timeout=45.0
        )
        
        try:
            papers = json.loads(response.text.strip())
            # Ensure it's a list
            if not isinstance(papers, list):
                if isinstance(papers, dict) and 'data' in papers:
                    papers = papers['data']
                else:
                    papers = [papers]
                    
            # Basic validation
            valid_papers = []
            for p in papers:
                if isinstance(p, dict) and p.get("title") and p.get("pdf_url"):
                    valid_papers.append(p)
            papers = valid_papers[:max_results]
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini output: {response.text}")
            raise Exception("Gemini returned invalid JSON") from e
            

        logger.info(f"Successfully extracted {len(papers)} papers with PDFs visually.")
        
        return {
            "status": "success",
            "data": papers
        }

    except Exception as e:
        logger.error(f"Scraper completely failed: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }
        
    finally:
        if browser:
            await browser.close()
        if _playwright:
            await _playwright.stop()
