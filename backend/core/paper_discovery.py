"""
paper_discovery.py — Extract paper metadata from pages visited during autopilot browsing.

After the vision loop browses Google and navigates through search results,
this module uses Gemini Vision to identify academic papers visible on
the browser's current page (and optionally from visited URLs history).

This bridges the gap between:
  - Mode 2 (Autopilot vision loop) — finds papers via browser
  - ADK Pipeline (SearchAgent → ExtractionAgent → ...) — processes them

The discovered papers are in the same format expected by the ADK agents.
"""

import asyncio
import base64
import json
import logging
from io import BytesIO

from google import genai
from google.genai import types
from PIL import Image

from config import settings

logger = logging.getLogger(__name__)

# ─── Screenshot compression for Vision API calls ───
_MAX_WIDTH = 1280
_MAX_HEIGHT = 960
_JPEG_QUALITY = 80


def _compress_for_vision(screenshot_b64: str) -> bytes:
    """Compress screenshot for Gemini Vision extraction."""
    raw = base64.b64decode(screenshot_b64)
    img = Image.open(BytesIO(raw))
    if img.width > _MAX_WIDTH or img.height > _MAX_HEIGHT:
        img.thumbnail((_MAX_WIDTH, _MAX_HEIGHT), Image.LANCZOS)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
    return buf.getvalue()


def _get_client():
    """Get a Gemini client for paper extraction."""
    import os
    from dotenv import load_dotenv
    load_dotenv()

    project_id = settings.VERTEX_AI_PROJECT or settings.PROJECT_ID
    if not project_id:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    location = settings.VERTEX_AI_LOCATION or "global"

    if project_id:
        return genai.Client(vertexai=True, project=project_id, location=location)

    api_key = settings.GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)

    raise EnvironmentError("No Gemini API credentials available")


async def extract_papers_from_screenshot(
    screenshot_b64: str,
    current_url: str = "",
    task: str = "",
) -> list[dict]:
    """
    Use Gemini Vision to extract paper metadata from a browser screenshot.

    This is called after the vision loop completes browsing. The screenshot
    typically shows Google search results or an academic page listing papers.

    Returns a list of paper dicts compatible with the ADK pipeline:
      [{"title": "...", "url": "...", "authors": [...], "snippet": "...", "year": ...}, ...]
    """
    client = _get_client()
    compressed = _compress_for_vision(screenshot_b64)

    model_name = settings.GOOGLE_VISION_MODEL

    prompt = f"""You are looking at a browser screenshot. The user was researching: "{task}"
The current URL is: {current_url}

Extract ALL academic papers/articles visible on this page. For each paper found, provide:
- "title": The full paper title
- "authors": List of author names (as strings)
- "url": The URL/link to the paper if visible
- "snippet": A brief description or abstract snippet if visible
- "year": Publication year if visible (as integer)
- "source": Where this paper was found (e.g., "google_search", "arxiv", "scholar")

Return ONLY a valid JSON array of paper objects. If no papers are visible, return an empty array [].
Do NOT include any markdown formatting, code fences, or explanation text.
"""

    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=model_name,
                contents=[
                    types.Content(role="user", parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(data=compressed, mime_type="image/jpeg"),
                    ])
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            ),
            timeout=30.0,
        )

        text = response.text.strip()
        papers = json.loads(text)

        if not isinstance(papers, list):
            if isinstance(papers, dict) and "papers" in papers:
                papers = papers["papers"]
            else:
                papers = [papers] if isinstance(papers, dict) else []

        # Validate and clean
        valid = []
        for p in papers:
            if isinstance(p, dict) and p.get("title"):
                valid.append({
                    "title": p.get("title", ""),
                    "url": p.get("url", ""),
                    "authors": p.get("authors", []),
                    "snippet": p.get("snippet", p.get("abstract", "")),
                    "year": p.get("year"),
                    "source": p.get("source", "vision_discovery"),
                })
        
        logger.info(f"Extracted {len(valid)} papers from screenshot")
        return valid

    except asyncio.TimeoutError:
        logger.warning("Paper extraction from screenshot timed out")
        return []
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse paper extraction response: {e}")
        return []
    except Exception as e:
        logger.warning(f"Paper extraction failed: {e}")
        return []


async def _extract_papers_from_url(
    browser,
    url: str,
    task: str,
    seen_titles: set,
) -> list[dict]:
    """
    Navigate the browser to `url`, take a screenshot, and extract papers.
    Returns only papers whose titles haven't been seen before.
    """
    papers_found = []
    try:
        await browser.page.goto(url, wait_until="domcontentloaded", timeout=15000)
        await asyncio.sleep(1.0)
        screenshot_b64 = await browser.screenshot_b64()
        papers = await extract_papers_from_screenshot(screenshot_b64, url, task)
        for p in papers:
            key = p["title"].lower().strip()[:80]
            if key and key not in seen_titles:
                seen_titles.add(key)
                papers_found.append(p)
    except Exception as e:
        logger.warning(f"Paper extraction from URL {url} failed: {e}")
    return papers_found


async def extract_papers_from_page_content(
    browser,
    task: str = "",
    visited_urls: list[str] | None = None,
) -> list[dict]:
    """
    Extract papers by taking screenshots of the browser's current state AND
    all previously visited URLs collected during the vision loop.

    This fixes the "only captures final page" bug: if the agent browsed 8 pages
    and the final page is a single paper's abstract, we still discover all papers
    seen on the earlier search-results pages.

    Args:
        browser:      Active browser instance (StealthBrowserController or BrowserController).
        task:         The research query string (used in the Gemini Vision prompt).
        visited_urls: Ordered list of URLs visited during the vision loop.
                      Collected from ``{"action": "url_visited"}`` events.
                      If None or empty, falls back to current-page-only extraction.
    """
    all_papers = []
    seen_titles: set[str] = set()

    # ── Step 1: Extract from the browser's CURRENT viewport (final page state) ──
    try:
        screenshot_b64 = await browser.screenshot_b64()
        current_url = browser.page.url if browser.page else ""
        papers = await extract_papers_from_screenshot(screenshot_b64, current_url, task)
        for p in papers:
            key = p["title"].lower().strip()[:80]
            if key and key not in seen_titles:
                seen_titles.add(key)
                all_papers.append(p)
        logger.info(f"Current page ({current_url}): {len(papers)} papers found")
    except Exception as e:
        logger.warning(f"Current viewport paper extraction failed: {e}")

    # Scroll down once on the current page to capture below-the-fold results
    try:
        await browser.page.evaluate("window.scrollBy(0, 800)")
        await asyncio.sleep(1.0)
        screenshot_b64 = await browser.screenshot_b64()
        current_url = browser.page.url if browser.page else ""
        papers = await extract_papers_from_screenshot(screenshot_b64, current_url, task)
        for p in papers:
            key = p["title"].lower().strip()[:80]
            if key and key not in seen_titles:
                seen_titles.add(key)
                all_papers.append(p)
    except Exception as e:
        logger.warning(f"Scrolled viewport paper extraction failed: {e}")

    # ── Step 2: Re-visit previously browsed URLs and extract papers from each ──
    # Limit to the most recent 10 URLs to avoid excessive API calls.
    # Skip the current URL (already extracted above) and PDF/binary URLs.
    _skip_extensions = (".pdf", ".zip", ".gz", ".tar", ".png", ".jpg", ".jpeg")
    current_url_now = browser.page.url if browser.page else ""
    urls_to_revisit: list[str] = []

    if visited_urls:
        # Process in reverse order (most recently visited first — most relevant)
        urls_to_revisit = [
            u for u in reversed(visited_urls)
            if u != current_url_now
            and not any(u.lower().endswith(ext) for ext in _skip_extensions)
        ][:10]  # Cap at 10 to limit Vision API calls

        if urls_to_revisit:
            logger.info(
                f"Re-visiting {len(urls_to_revisit)} previously browsed URLs "
                f"for multi-page paper discovery…"
            )
            for url in urls_to_revisit:
                new_papers = await _extract_papers_from_url(browser, url, task, seen_titles)
                if new_papers:
                    logger.info(f"  {url}: +{len(new_papers)} new papers")
                    all_papers.extend(new_papers)

    pages_visited = 1 + len(urls_to_revisit)
    logger.info(
        f"Total papers discovered from browser session: {len(all_papers)} "
        f"(from {pages_visited} pages)"
    )
    return all_papers
