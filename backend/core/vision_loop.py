"""
vision_loop.py — Core ReAct vision loop for browser automation.

Screenshot → Gemini Vision → JSON action → Playwright executes → repeat.

Features:
  - Retry with exponential backoff on Gemini API errors
  - Timeout on each Gemini call (configurable)
  - Enforced JSON output via response_mime_type
  - Action schema validation before execution
  - Rolling history with last N screenshots for visual grounding
"""

import json
import asyncio
import logging
import math
import os
import base64
import re
from collections import Counter   # ── Bug 6: loop detection
from io import BytesIO
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from PIL import Image, ImageChops


# HOW TO SWITCH BACK TO AI STUDIO

# 1. Update `_get_client()` below to return `genai.Client(api_key=...)`.
# 2. Ensure your .env file has GOOGLE_API_KEY or GEMINI_API_KEY set.


# --- Shared Imports ---
from google import genai
from google.genai import types

from core.browser import BrowserController
from core.stealth_browser import StealthBrowserController
from config import settings
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

# ─── Modular Action Reliability Pipeline ───
from core.action_validator import validate_action
from core.action_corrector import correct_action

from prompts import VISION_LOOP_SYSTEM_PROMPT

SYSTEM_PROMPT = VISION_LOOP_SYSTEM_PROMPT

# ─── Retry Configuration ───
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds
GEMINI_TIMEOUT = 30.0   # seconds per call
# ── Bug 5 Fix: 0.003 (0.3%) is too sensitive — a blinking cursor or spinner
# fires a false stall after just 2 steps, wasting the step budget on fallbacks.
# Raised threshold to 0.5% and repeat limit to 3 for more tolerance.
VISUAL_STALL_THRESHOLD = 0.005  # 0.5% — avoids false stalls on idle/slow pages
STALL_REPEAT_LIMIT = 3          # Require 3 identical frames before declaring stall


# --- Vertex AI Implementation (Active) ---
def _configure_genai():
    """Lazy-init: configure the SDK only once if needed."""
    pass


_genai_configured = False


def _get_client():
    """Return a Vertex AI Client."""
    from dotenv import load_dotenv
    load_dotenv()
    
    project_id = settings.VERTEX_AI_PROJECT or settings.PROJECT_ID
    location = settings.VERTEX_AI_LOCATION
    
    if not project_id:
        # os.getenv("GOOGLE_CLOUD_PROJECT") or
        project_id =  os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        if not project_id:
            raise EnvironmentError("GOOGLE_CLOUD_PROJECT (or GOOGLE_CLOUD_PROJECT_ID) is not set — cannot initialise Vertex AI")
            
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        logger.warning("GOOGLE_APPLICATION_CREDENTIALS is not set; assuming Application Default Credentials are configured.")
            
    logger.info(f"Initializing Vertex AI for project '{project_id}' in '{location}'")
    return genai.Client(vertexai=True, project=project_id, location=location)


# Removed: validate_action is now imported from core.action_validator


def try_repair_json(raw: str) -> dict | None:
    """
    Attempt to repair common Gemini truncation issues in JSON.
    """
    raw = raw.strip()
    if not raw:
        return None
    
    # ─── 0. Extract potential JSON block (handles preamble text) ───
    # Find the FIRST '{' and the LAST '}' (if it exists)
    start_idx = raw.find('{')
    if start_idx == -1:
        # If no '{', Gemini might have started the JSON without the opening brace
        # but inside markdown backticks.
        if "```" in raw:
            # Try to strip backticks first then search again
            temp = raw
            if "```json" in temp:
                temp = temp.split("```json")[1]
            elif "```" in temp:
                temp = temp.split("```")[1]
            temp = temp.split("```")[0].strip()
            start_idx = temp.find('{')
            if start_idx != -1:
                raw = temp[start_idx:]
            else:
                # Still no brace? Check if it looks like a truncated JSON starting with keys
                if '"action":' in temp:
                    raw = "{" + temp
                else:
                    return None
        else:
            return None
    else:
        raw = raw[start_idx:]

    # ─── 1. Strip markdown backticks if present ───
    # If the response is wrapped in backticks, but the last one is missing (truncation)
    if "```" in raw:
        raw = raw.split("```")[0].strip()

    # ─── 2. Extract first complete JSON object (handles trailing garbage) ───
    # Gemini sometimes echoes part of the reason string after the closing brace,
    # e.g. {"action":"scroll","reason":"search for XAI papers"} papers"}
    # We find the first balanced JSON object and discard trailing text.
    brace_count = 0
    in_string = False
    escaped = False
    first_json_end = -1

    for i, char in enumerate(raw):
        if char == '\\' and not escaped:
            escaped = True
            continue

        if char == '"' and not escaped:
            in_string = not in_string
        elif char == '{' and not in_string:
            brace_count += 1
        elif char == '}' and not in_string:
            brace_count -= 1
            if brace_count == 0:
                first_json_end = i + 1
                break

        escaped = False

    if first_json_end > 0:
        raw = raw[:first_json_end]
        logger.debug(f"Extracted first complete JSON object (length: {first_json_end})")

    # Try standard parse (catches complete JSON or the extracted object)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # ─── 3. Repair Truncation ───
    repaired = raw.strip()
    
    # Balance quotes (ignoring escaped ones)
    in_string = False
    escaped = False
    balance_repaired = ""
    
    for char in repaired:
        if char == '\\' and not escaped:
            escaped = True
        elif char == '"' and not escaped:
            in_string = not in_string
            escaped = False
        else:
            escaped = False
        balance_repaired += char
        
    repaired = balance_repaired
    if in_string:
        repaired += '"'
        
    # Balance braces
    open_braces = 0
    in_string = False
    escaped = False
    
    for char in repaired:
        if char == '\\' and not escaped:
            escaped = True
        elif char == '"' and not escaped:
            in_string = not in_string
            escaped = False
        elif char == '{' and not in_string:
            open_braces += 1
            escaped = False
        elif char == '}' and not in_string:
            open_braces -= 1
            escaped = False
        else:
            escaped = False
            
    if open_braces > 0:
        repaired += "}" * open_braces

    try:
        data = json.loads(repaired)
        logger.info("Successfully repaired truncated Gemini JSON.")
        return data
    except Exception as e:
        logger.debug(f"JSON repair failed: {e}. Repaired string: {repaired}")
        return None


def _normalize_reason(reason: str) -> str:
    return (reason or "").strip().lower()


def _classify_intent(action: dict) -> str:
    act = (action or {}).get("action", "")
    reason = _normalize_reason((action or {}).get("reason", ""))
    text = f"{act} {reason}"

    tools_keywords = (
        "tools",
        "any time",
        "date filter",
        "time filter",
        "custom range",
        "time range",
    )
    search_keywords = (
        "search bar",
        "search input",
        "search field",
        "query input",
        "entering query",
        "focus search",
    )

    if any(keyword in text for keyword in tools_keywords):
        return "google_tools_filter"
    if any(keyword in text for keyword in search_keywords):
        return "search_input_focus"
    return "generic"


def _infer_google_qdr(task: str) -> str | None:
    prompt = (task or "").lower()
    checks = [
        (r"(?:past|last)\s+(\d+)\s+hours?|past\s+hour|last\s+hour", "h"),
        (r"(?:past|last)\s+(\d+)\s+weeks?|past\s+week|last\s+week", "w"),
        (r"(?:past|last)\s+(\d+)\s+months?|past\s+month|last\s+month", "m"),
        (r"(?:past|last)\s+(\d+)\s+years?|past\s+year|last\s+year", "y"),
        (r"(?:past|last)\s+(\d+)\s+days?|past\s+day|last\s+day|today", "d"),
    ]
    for pattern, prefix in checks:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            # Check if there's a capture group and it has a value (e.g. "2" in "past 2 years")
            # If the value is "1", we treat it as singular (no digit suffix) to match "h", "d", etc.
            digits = match.group(1) if match.lastindex and match.group(1) else ""
            if digits == "1":
                digits = ""
            return f"{prefix}{digits}"
    return None


def _infer_google_time_phrase(task: str) -> str | None:
    prompt = (task or "").lower()
    checks = [
        (r"(?:past|last)\s+1\s+hour|past\s+hour", "past hour"),
        (r"(?:past|last)\s+24\s+hours?|past\s+day|last\s+day|today", "past 24 hours"),
        (r"(?:past|last)\s+week|past\s+7\s+days", "past week"),
        (r"(?:past|last)\s+month|past\s+30\s+days", "past month"),
        (r"(?:past|last)\s+year|past\s+12\s+months", "past year"),
        (r"(?:from|between)?\s*((\d{4})\s*(?:to|and|–|-)\s*(\d{4}))", None),
        (r"(since\s+(\d{4}))", None),
    ]
    for pattern, phrase in checks:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if not match:
            continue
        if phrase:
            return phrase
        # If group(1) exists, it's the date range without "from/between"
        return match.group(1).strip() if match.lastindex and match.group(1) else match.group(0).strip()
    return None


def _build_google_filtered_url(current_url: str, task: str) -> str | None:
    if not current_url:
        return None

    parsed = urlparse(current_url)
    if "google." not in parsed.netloc.lower() or not parsed.path.startswith("/search"):
        return None

    query = parse_qs(parsed.query, keep_blank_values=True)
    existing_tbs = query.get("tbs", [""])[0]
    
    # ─── Case 1: Detect explicit year range (e.g. "2020 to 2022") ───
    range_match = re.search(r"(\d{4})\s*(?:to|and|–|-)\s*(\d{4})", task)
    if range_match:
        y1, y2 = range_match.groups()
        # Google CDR format: cdr:1,cd_min:1/1/YYYY,cd_max:12/31/YYYY
        range_tbs = f"cdr:1,cd_min:1/1/{y1},cd_max:12/31/{y2}"
        if existing_tbs:
            # Replace existing cdr or qdr if present
            parts = [p for p in existing_tbs.split(",") if not (p.startswith("qdr:") or p.startswith("cdr:"))]
            parts.append(range_tbs)
            query["tbs"] = [",".join(parts)]
        else:
            query["tbs"] = [range_tbs]
        return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))

    # ─── Case 2: Detect "since YYYY" (e.g. "since 2020") ───
    since_match = re.search(r"since\s+(\d{4})", task)
    if since_match:
        year = since_match.group(1)
        # We set max to current year if possible, or just leave it open if Google supports it.
        # Most reliable is to set a far-future max.
        current_year = 2026 # From system context
        since_tbs = f"cdr:1,cd_min:1/1/{year},cd_max:12/31/{current_year}"
        if existing_tbs:
            parts = [p for p in existing_tbs.split(",") if not (p.startswith("qdr:") or p.startswith("cdr:"))]
            parts.append(since_tbs)
            query["tbs"] = [",".join(parts)]
        else:
            query["tbs"] = [since_tbs]
        return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))

    # ─── Case 3: Relative QDR (past year, etc.) ───
    qdr = _infer_google_qdr(task)
    if not qdr:
        return None

    if existing_tbs:
        parts = [part for part in existing_tbs.split(",") if part]
        replaced = False
        for index, part in enumerate(parts):
            if part.startswith("qdr:") or part.startswith("cdr:"):
                parts[index] = f"qdr:{qdr}"
                replaced = True
                break
        if not replaced:
            parts.append(f"qdr:{qdr}")
        query["tbs"] = [",".join(parts)]
    else:
        query["tbs"] = [f"qdr:{qdr}"]

    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


def _build_google_query_with_time_phrase(current_url: str, task: str) -> str | None:
    if not current_url:
        return None

    parsed = urlparse(current_url)
    if "google." not in parsed.netloc.lower() or not parsed.path.startswith("/search"):
        return None

    time_phrase = _infer_google_time_phrase(task)
    if not time_phrase:
        return None

    query = parse_qs(parsed.query, keep_blank_values=True)
    existing_q = (query.get("q") or [""])[0].strip()
    if not existing_q:
        return None

    if time_phrase.lower() in existing_q.lower():
        return current_url

    query["q"] = [f"{existing_q} {time_phrase}".strip()]
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


def _build_direct_time_query_url(current_url: str, task: str) -> str | None:
    """
    Build a direct time-qualified search URL when toolbar-based time filtering fails.
    Priority:
      1) Native Google /search qdr filter
      2) Google/Scholar query phrase augmentation
      3) Hard fallback to a Google search query including time phrase
    """
    filtered_url = _build_google_filtered_url(current_url, task)
    if filtered_url:
        return filtered_url

    query_with_time = _build_google_query_with_time_phrase(current_url, task)
    if query_with_time:
        return query_with_time

    time_phrase = _infer_google_time_phrase(task)
    if not time_phrase:
        return None

    parsed = urlparse(current_url) if current_url else None
    base_query = (task or "").strip()

    if parsed and "google." in parsed.netloc.lower():
        query = parse_qs(parsed.query, keep_blank_values=True)
        existing_q = (query.get("q") or [""])[0].strip()
        if existing_q:
            base_query = existing_q

        # Scholar path fallback: keep the same endpoint, rewrite q with explicit time phrase.
        if parsed.path.startswith("/scholar"):
            # Check for year range in task
            range_match = re.search(r"(\d{4})\s*(?:to|and|–|-)\s*(\d{4})", task)
            if range_match:
                ylo, yhi = range_match.groups()
                query["as_ylo"] = [ylo]
                query["as_yhi"] = [yhi]
                return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))
            
            since_match = re.search(r"since\s+(\d{4})", task)
            if since_match:
                query["as_ylo"] = [since_match.group(1)]
                return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))

            combined_query = base_query
            if time_phrase.lower() not in combined_query.lower():
                combined_query = f"{combined_query} {time_phrase}".strip()
            query["q"] = [combined_query]
            return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))

    if not base_query:
        return None

    if time_phrase.lower() not in base_query.lower():
        base_query = f"{base_query} {time_phrase}".strip()

    return f"https://www.google.com/search?{urlencode({'q': base_query})}"


def _visual_change_ratio(prev_frame_b64: str, current_frame_b64: str) -> float:
    try:
        previous = Image.open(BytesIO(base64.b64decode(prev_frame_b64))).convert("L").resize((320, 200))
        current = Image.open(BytesIO(base64.b64decode(current_frame_b64))).convert("L").resize((320, 200))
        diff = ImageChops.difference(previous, current)
        histogram = diff.histogram()
        total_pixels = 320 * 200
        unchanged_pixels = histogram[0] if histogram else 0
        return 1.0 - (unchanged_pixels / max(total_pixels, 1))
    except Exception:
        return 1.0


def _build_vision_fallback_action(
    intent: str,
    task: str,
    current_url: str,
    search_stage: int,
) -> tuple[dict | None, int]:
    if intent == "google_tools_filter":
        direct_url = _build_direct_time_query_url(current_url, task)
        if direct_url and direct_url != current_url:
            return (
                {
                    "action": "navigate",
                    "text": direct_url,
                    "reason": "Applying date filter via URL fallback after repeated Tools failure.",
                },
                search_stage,
            )
        return (
            {
                "action": "press",
                "text": "Escape",
                "reason": "Closing possible toolbar overlay before retrying date filter.",
            },
            search_stage,
        )

    if intent == "search_input_focus":
        if search_stage == 0:
            return (
                {
                    "action": "press",
                    "text": "/",
                    "reason": "Keyboard fallback to focus search input.",
                },
                1,
            )
        if search_stage == 1:
            return (
                {
                    "action": "press",
                    "text": "Control+K",
                    "reason": "Secondary keyboard fallback to focus search input.",
                },
                2,
            )
        return (
            {
                "action": "scroll",
                "delta": -500,
                "reason": "Reframing top search region after repeated focus failures.",
            },
            0,
        )

    return None, search_stage


async def _call_gemini_with_retry(client, model_name, contents, config) -> str:
    """
    Call Gemini with retry + exponential backoff + timeout.
    Returns the raw text response.
    """
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                ),
                timeout=GEMINI_TIMEOUT,
            )
            return response.text.strip()

        except asyncio.TimeoutError:
            last_error = f"Gemini call timed out after {GEMINI_TIMEOUT}s"
            logger.warning(f"Attempt {attempt}/{MAX_RETRIES}: {last_error}")

        except Exception as e:
            last_error = str(e)
            logger.warning(f"Attempt {attempt}/{MAX_RETRIES}: Gemini error — {last_error}")

        if attempt < MAX_RETRIES:
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.info(f"Retrying in {delay}s …")
            await asyncio.sleep(delay)

    raise RuntimeError(f"Gemini failed after {MAX_RETRIES} attempts: {last_error}")


async def _detect_captcha_in_screenshot(screenshot_b64: str) -> bool:
    """
    Pure-vision CAPTCHA detection using PIL only (no cv2/numpy dependency).

    Checks two independent signals:
      1. A suspiciously large uniform white/light-gray region in the centre of
         the viewport — characteristic of an reCAPTCHA / hCaptcha overlay box.
      2. Absence of typical page-content pixel variance — a challenge page is
         visually much flatter than a normal search-results page.

    Returns True if either signal fires above threshold.
    """
    try:
        img_data = base64.b64decode(screenshot_b64)
        img = Image.open(BytesIO(img_data)).convert("RGB")
        w, h = img.size

        # ── Signal 1: large light-gray/white box in the centre third ──────────
        # reCAPTCHA checkbox widget is ~300×74 px; the challenge overlay is
        # ~400×600 px.  Both appear as a near-white rectangle centred on screen.
        cx1, cy1, cx2, cy2 = w // 3, h // 3, 2 * w // 3, 2 * h // 3
        centre_crop = img.crop((cx1, cy1, cx2, cy2)).convert("L")
        centre_pixels = list(centre_crop.getdata())
        light_count = sum(1 for p in centre_pixels if p > 220)   # near-white
        light_ratio  = light_count / max(len(centre_pixels), 1)

        if light_ratio > 0.70:   # >70 % of centre crop is near-white → overlay
            logger.info(f"CAPTCHA signal 1 fired: centre light-ratio={light_ratio:.2%}")
            return True

        # ── Signal 2: abnormally low global pixel variance ─────────────────────
        # A blocked / challenge page is almost entirely white/light-gray;
        # a normal search-results page has text, images, varied colours.
        small = img.convert("L").resize((160, 100))
        pixels = list(small.getdata())
        mean   = sum(pixels) / len(pixels)
        variance = sum((p - mean) ** 2 for p in pixels) / len(pixels)

        if variance < 200 and mean > 200:   # flat + bright → near-blank page
            logger.info(f"CAPTCHA signal 2 fired: variance={variance:.1f}, mean={mean:.1f}")
            return True

        return False

    except Exception as e:
        logger.debug(f"CAPTCHA detection failed: {e}")
        return False


async def _monitor_captcha_resolution(
    # browser: BrowserController,
    browser : StealthBrowserController,
    initial_screenshot: str,
    max_wait_seconds: int = 120
) -> tuple[bool, str]:
    """
    Passively monitor for CAPTCHA resolution without interfering.
    
    Returns:
        (resolved: bool, final_screenshot: str)
    """
    logger.info("="*80)
    logger.info("CAPTCHA MONITORING STARTED")
    logger.info("="*80)
    logger.info(f"Max wait time: {max_wait_seconds}s")
    logger.info("Entering passive CAPTCHA monitoring mode (no AI interference)")
    
    start_time = asyncio.get_running_loop().time()
    check_interval = 3.0  # Check every 3 seconds
    last_change_ratio = 0.0
    stable_count = 0
    
    while True:
        elapsed = asyncio.get_running_loop().time() - start_time
        
        if elapsed > max_wait_seconds:
            logger.warning("="*80)
            logger.warning(f"CAPTCHA MONITORING TIMEOUT after {max_wait_seconds}s")
            logger.warning("User did not solve CAPTCHA in time")
            logger.warning("="*80)
            return False, initial_screenshot
        
        # Wait before taking next screenshot (passive monitoring)
        await asyncio.sleep(check_interval)
        
        try:
            # Take screenshot without grid overlay (less intrusive)
            current_screenshot = await browser.screenshot_b64()
            
            # Check if page has changed significantly (CAPTCHA likely solved)
            change_ratio = _visual_change_ratio(initial_screenshot, current_screenshot)
            
            # Log every check for debugging
            if int(elapsed) % 6 == 0:  # Log every 6 seconds (every 2 checks)
                logger.debug(f"[{int(elapsed)}s] Visual change ratio: {change_ratio:.2%}")
            
            # If significant change detected, CAPTCHA might be solved
            if change_ratio > 0.15:  # 15% change threshold
                logger.info(f"[{int(elapsed)}s] Significant page change detected ({change_ratio:.2%})")
                
                # Double-check: is CAPTCHA still visible?
                captcha_still_present = await _detect_captcha_in_screenshot(current_screenshot)
                logger.info(f"[{int(elapsed)}s] CAPTCHA detection result: {captcha_still_present}")
                
                if not captcha_still_present:
                    logger.info("="*80)
                    logger.info("✅ CAPTCHA RESOLVED - RESUMING AUTOMATION")
                    logger.info("="*80)
                    return True, current_screenshot
                else:
                    logger.info(f"[{int(elapsed)}s] Page changed but CAPTCHA still present - continuing to monitor")
                    stable_count = 0
            else:
                # Track stable periods (no change)
                if change_ratio == last_change_ratio:
                    stable_count += 1
                else:
                    stable_count = 0
                
                # Log progress every 15 seconds
                if int(elapsed) % 15 == 0:
                    logger.info(f"[{int(elapsed)}s] Still monitoring CAPTCHA... (change ratio: {change_ratio:.2%})")
            
            last_change_ratio = change_ratio
                
        except Exception as e:
            logger.error(f"[{int(elapsed)}s] Error during CAPTCHA monitoring: {e}", exc_info=True)
            await asyncio.sleep(check_interval)
            continue
    

async def run_vision_loop(
    # browser: BrowserController,
    browser: StealthBrowserController,
    task: str,
    max_steps: int = 50, # changed from 30
    pause_event: asyncio.Event = None,
    prior_tasks: list[str] = None,
) -> AsyncGenerator[dict, None]:
    """
    Core ReAct loop: screenshot → Gemini → action → repeat.
    Yields each action for real-time streaming to frontend.

    prior_tasks: ordered list of previous user queries in this session.
    When provided, the task prompt includes prior context so Gemini
    treats this as a continuation rather than a brand-new session.
    """
    client = _get_client()
    model_name = settings.GOOGLE_VISION_MODEL
    logger.info(f"Using vision model: {model_name}")

    # ── Conversation continuity: build enriched task string ──
    if prior_tasks:
        prior_summary = "; ".join(f'"{t}"' for t in prior_tasks)
        enriched_task = (
            f"[FOLLOW-UP QUERY — ongoing research session]\n"
            f"Previous queries: {prior_summary}\n"
            f"Current query: {task}\n\n"
            f"Build on the previous research context. "
            f"Do NOT repeat searches already done. "
            f"Focus on the current query."
        )
    else:
        enriched_task = task
    
    history = []          # text summaries of past steps
    recent_frames = []    # last N screenshots for visual grounding
    step = 0
    consecutive_clicks = 0
    last_click_coords = None
    last_step_frame = None
    stalled_intent = None
    stalled_repeat_count = 0
    search_fallback_stage = 0
    tools_attempt_count = 0
    tools_dismissed = False
    query_typed = False  # True only AFTER a type action is confirmed executed
    action_fingerprints: list[str] = []  # ── Bug 6: cross-step loop detection
    same_url_steps = 0          # counts consecutive steps where URL hasn't changed
    last_seen_url = ""          # tracks the URL at end of previous step

    generation_config = types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=4096,
        response_mime_type="application/json",
        system_instruction=SYSTEM_PROMPT,
    )

    

    while step < max_steps:
        # Pre-capture checks
        await browser.wait_for_visual_stability()
        await browser.inject_grid(cell_size=80)
        
        # 1. Capture current screen
        screenshot_b64 = await browser.screenshot_b64()

        # ── Blocked-domain auto-escape ──
        # Must remove_grid BEFORE go_back — otherwise the grid overlay is
        # injected into a page we're about to leave and never cleaned up.
        BLOCKED_DOMAINS = [
            "ssrn.com", "papers.ssrn.com",
            "researchgate.net", "academia.edu",
            # Subscription/paywall sites
            "ieeexplore.ieee.org", "ieee.org",
            "sciencedirect.com",
            "springer.com", "link.springer.com",
            "dl.acm.org",
            "wiley.com", "onlinelibrary.wiley.com",
            "tandfonline.com",
            "nature.com",
            "science.org",
            "jstor.org",
        ]
        current_url = browser.page.url if browser.page else ""
        if any(domain in current_url for domain in BLOCKED_DOMAINS):
            logger.warning(f"Blocked domain detected ({current_url}). Auto-navigating back.")
            await browser.remove_grid()          # ← clean up grid FIRST
            await browser.page.go_back()
            await asyncio.sleep(1.5)
            screenshot_b64 = await browser.screenshot_b64()
            history.append(f"Step {step + 1}: Auto-escaped blocked/paywalled site — navigated back.")
            if len(history) > 10:
                history = history[-10:]
            tools_dismissed = False              # ← reset on navigation
            step += 1
            continue

        screenshot_bytes = base64.b64decode(screenshot_b64)
        
        # Post-capture cleanup
        await browser.remove_grid()

        contents = []

        # 2. FIX: Reconstruct history as a proper multi-turn dialogue (User/Model roles)
        for prev_frame in recent_frames[-2:]:
            # User turn (What the model saw previously)
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=base64.b64decode(prev_frame["frame"]), mime_type="image/png"),
                        types.Part.from_text(text=f"Task: {enriched_task}\n\nStep {prev_frame['step']}. What is the next single action?")
                    ]
                )
            )
            # Model turn (What the model did previously)
            contents.append(
                types.Content(
                    role="model",
                    parts=[
                        types.Part.from_text(text=prev_frame['action_summary'])
                    ]
                )
            )

        # 3. Add the CURRENT step
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=screenshot_bytes, mime_type="image/png"),
                    types.Part.from_text(text=f"Task: {enriched_task}\n\nStep {step + 1}. What is the next single action?")
                ]
            )
        )

        # 4. Call Gemini with retry + timeout
        try:
            raw = await _call_gemini_with_retry(client, model_name, contents, generation_config)
        except RuntimeError as e:
            yield {"action": "error", "reason": str(e)}
            break

        # 5. Parse action JSON (with retry on truncation)
        action = try_repair_json(raw)
        if not action:
            logger.warning(f"Step {step + 1}: Gemini returned unparseable JSON: {raw[:300]}")
            # Don't break immediately — retry once with a fresh screenshot
            if step > 0:
                history.append(f"Step {step + 1}: JSON parse error — retrying")
                step += 1
                continue
            yield {"action": "error", "reason": f"Gemini returned invalid or truncated JSON: {raw[:200]}"}
            break

        # 6. Validate + Self-Heal Action Schema
        validation_error = validate_action(action)
        if validation_error:
            logger.warning(f"Step {step + 1}: invalid action — {validation_error}. Attempting correction...")
            action = correct_action(action)
            validation_error = validate_action(action) # Re-validate after correction
            
        if validation_error:
            logger.error(f"Step {step + 1}: could not fix action — {validation_error}")
            yield {"action": "error", "reason": f"Invalid action from Gemini: {validation_error}", "raw": action}
            break

        # ── Bug 6 Fix: Cross-step loop detection using action fingerprints.
        # The existing consecutive_clicks guard only catches repeated clicks at
        # the same coordinates.  This guard catches ANY action (navigate, type,
        # press, scroll, click) that repeats 4+ times across non-consecutive
        # steps — which is the real cause of the 30-step burn-out.
        _fp = f"{action.get('action')}:{action.get('text', '')}:{action.get('x', '')}:{action.get('y', '')}"
        action_fingerprints.append(_fp)
        if Counter(action_fingerprints)[_fp] >= 4:
            logger.warning(f"Step {step + 1}: loop detected — action '{_fp}' has repeated 4+ times. Breaking to preserve results.")
            yield {
                "action": "error",
                "reason": f"Agent stuck in a loop repeating the same action 4+ times: {_fp}. Stopping early to preserve partial results.",
                "step": step + 1,
            }
            break

        intent = _classify_intent(action)
        current_url = browser.page.url if browser.page else ""

        # ── Track how many steps we've spent on the same URL ──
        if current_url == last_seen_url:
            same_url_steps += 1
        else:
            same_url_steps = 0
            last_seen_url = current_url

        # ── Scholar snippet trap: agent stuck scrolling/clicking on Google
        # without ever clicking an actual result link ──
        if (
            "google." in current_url
            and "scholar" not in current_url
            and intent in {"google_tools_filter", "generic"}
            and action.get("action") in {"click", "scroll"}
            and same_url_steps >= 3
        ):
            logger.warning(f"Agent stuck on Google for {same_url_steps} steps without navigating — forcing scroll to results")
            action = {
                "action": "scroll",
                "delta": 400,
                "reason": "Scrolling past Scholar snippet to reach full search results.",
            }
            intent = _classify_intent(action)
        
        # CRITICAL FIX: Force type action when model says "typing" but action is not "type"
        # This prevents keyboard fallback loops when search bar is detected
        if intent == "search_input_focus" and action.get("action") != "type":
            reason_lower = _normalize_reason(action.get("reason", ""))
            # Check if the reason mentions typing a specific query
            if "typing" in reason_lower or "entering" in reason_lower:
                # Extract query from task if not in action
                if not action.get("text"):
                    # Look for quoted text or after "search for" / "find"
                    # re is already imported at the top of the module
                    query_match = re.search(r'search for[:\s]+["\']?([^"\']+)["\']?', task, re.IGNORECASE)
                    if not query_match:
                        query_match = re.search(r'find[:\s]+["\']?([^"\']+)["\']?', task, re.IGNORECASE)
                    if not query_match:
                        # Use first meaningful phrase from task
                        words = task.split()
                        query_text = ' '.join(words[:5])  # First 5 words as query
                    else:
                        query_text = query_match.group(1).strip()
                else:
                    query_text = action.get("text", "")
                
                # Force type action — no autofocus flag (no DOM selectors allowed)
                logger.info(f"Forcing type action for search query: '{query_text}'")
                action = {
                    "action": "type",
                    "text": query_text,
                    "reason": f"Typing search query: {query_text}",
                }
                intent = "generic"  # Reset intent to prevent fallback loops
                search_fallback_stage = 0  # Reset fallback counter
                # FIX: query_typed must NOT be set here — it must only be set
                # after execute_action() confirms the keystroke was delivered.
        
        # OPTIMIZED FIX: Try URL-based time filtering FIRST, before any Tools menu clicks
        # This reduces API calls by ~70% for date filtering tasks
        if intent == "google_tools_filter":
            if tools_dismissed:
                # Already dismissed the toolbar this page — never re-enter this branch
                intent = "generic"
            else:
                current_url = browser.page.url if browser.page else ""
                direct_url = _build_direct_time_query_url(current_url, task)

                # PRIORITY 1: Always try URL-based filtering first (no extra API calls)
                if direct_url and direct_url != current_url:
                    logger.info("Applying time filter via URL (most reliable method, skipping toolbar)")
                    action = {
                        "action": "navigate",
                        "text": direct_url,
                        "reason": "Applying date filter via URL parameter (more reliable than toolbar)",
                    }
                    intent = _classify_intent(action)
                    tools_attempt_count = 0  # Reset since we're not using Tools
                elif action.get("action") == "click":
                    # PRIORITY 2: Only allow ONE toolbar click attempt if URL filtering unavailable
                    tools_attempt_count += 1
                    if tools_attempt_count >= 1:  # Reduced from 2 to 1 for efficiency
                        # After 1 failed Tools attempt, immediately try query phrase augmentation
                        time_phrase = _infer_google_time_phrase(task)
                        if time_phrase:
                            logger.warning(
                                f"Tools click failed; switching to query phrase augmentation: '{time_phrase}'"
                            )
                            # Build a new search with time phrase in query
                            parsed = urlparse(current_url) if current_url else None
                            if parsed and "google." in parsed.netloc.lower():
                                query_params = parse_qs(parsed.query, keep_blank_values=True)
                                existing_q = (query_params.get("q") or [""])[0].strip()
                                if existing_q and time_phrase.lower() not in existing_q.lower():
                                    query_params["q"] = [f"{existing_q} {time_phrase}".strip()]
                                    new_url = urlunparse(parsed._replace(query=urlencode(query_params, doseq=True)))
                                    action = {
                                        "action": "navigate",
                                        "text": new_url,
                                        "reason": f"Tools failed; adding '{time_phrase}' to search query directly.",
                                    }
                                    intent = _classify_intent(action)
                                else:
                                    # Time phrase already in query — dismiss and never re-enter
                                    tools_dismissed = True
                                    action = {
                                        "action": "press",
                                        "text": "Escape",
                                        "reason": "Time filter already applied in query; dismissing toolbar.",
                                    }
                                    intent = _classify_intent(action)
                            else:
                                # Not on Google — dismiss and never re-enter
                                tools_dismissed = True
                                action = {
                                    "action": "press",
                                    "text": "Escape",
                                    "reason": "Tools unavailable; dismissing overlay.",
                                }
                                intent = _classify_intent(action)
                        else:
                            # No time phrase detected — dismiss and never re-enter
                            logger.warning("No time filter detected in task; dismissing toolbar.")
                            tools_dismissed = True
                            action = {
                                "action": "press",
                                "text": "Escape",
                                "reason": "No time filter needed; dismissing toolbar.",
                            }
                            intent = _classify_intent(action)
                        tools_attempt_count = 0

        # Reset counters when moving to different action types
        if intent != "google_tools_filter" and action.get("action") in {"navigate", "type", "press", "done"}:
            tools_attempt_count = 0

        # ── Reset tools_dismissed when the agent navigates to a new page ──
        # A fresh page load means a fresh toolbar state — the old dismiss is irrelevant.
        if action.get("action") == "navigate":
            tools_dismissed = False

        # Mark query as typed when type action is executed
        # if action.get("action") == "type":
        #     query_typed = True
        
        if intent != "search_input_focus":
            search_fallback_stage = 0
        
        # Additional reset: clear tools counter if we successfully navigated
        if action.get("action") == "navigate" and intent != "google_tools_filter":
            tools_attempt_count = 0

        # CRITICAL: Disable visual stall detection for search_input_focus if query already typed
        # This prevents keyboard fallback loops after successful typing
        if last_step_frame and intent in {"google_tools_filter", "search_input_focus"}:
            # FIX: use else: so the stall computation is entirely skipped when
            # the query was already typed.  Without else:, visual_change was still
            # computed and could still set stalled_repeat_count even after the
            # inner guard reset it.
            if intent == "search_input_focus" and query_typed:
                stalled_intent = None
                stalled_repeat_count = 0
            else:
                visual_change = _visual_change_ratio(last_step_frame, screenshot_b64)
                if visual_change < VISUAL_STALL_THRESHOLD:
                    if stalled_intent == intent:
                        stalled_repeat_count += 1
                    else:
                        stalled_intent = intent
                        stalled_repeat_count = 1
                else:
                    stalled_intent = None
                    stalled_repeat_count = 0
        else:
            stalled_intent = None
            stalled_repeat_count = 0

        if stalled_repeat_count >= STALL_REPEAT_LIMIT:
            # CRITICAL: Never trigger keyboard fallbacks if query already typed
            if intent == "search_input_focus" and query_typed:
                logger.info("Skipping keyboard fallback - query already typed, continuing normally")
                stalled_intent = None
                stalled_repeat_count = 0
            else:
                fallback_action, search_fallback_stage = _build_vision_fallback_action(
                    intent=intent,
                    task=task,
                    current_url=browser.page.url if browser.page else "",
                    search_stage=search_fallback_stage,
                )
                if fallback_action:
                    logger.warning(
                        f"Vision fallback triggered at step {step + 1} for intent '{intent}' "
                        f"(repeats={stalled_repeat_count})."
                    )
                    action = fallback_action
                    intent = _classify_intent(action)
                stalled_intent = None
                stalled_repeat_count = 0

        # --- Anti-stuck mechanism ---
        if action.get("action") == "click":
            coords = (action.get("x"), action.get("y"))
            
            # Check if clicks are within a small radius of each other
            is_same_area = False
            if last_click_coords:
                dist = math.hypot(coords[0] - last_click_coords[0], coords[1] - last_click_coords[1])
                is_same_area = dist < 25
                
            if is_same_area:
                consecutive_clicks += 1
            else:
                consecutive_clicks = 1
                last_click_coords = coords
            
            # Smarter threshold for dropdown/menu interactions
            is_dropdown_interaction = intent in {"google_tools_filter", "menu_navigation"}
            threshold = 4 if is_dropdown_interaction else 3
            
            if consecutive_clicks >= threshold:
                logger.warning(f"Anti-stuck triggered near {coords} (threshold={threshold})")
                action_summary = json.dumps({
                    "action": "error", 
                    "reason": f"STUCK: You already clicked near {coords} three times and nothing changed. The button might be unclickable, invisible under a layer, or the page is broken. DO NOT click here again. TRY SCROLLING or a different search approach."
                })
                anti_stuck_scroll = {"action": "scroll", "delta": 300, "step": step + 1, "reason": "Anti-stuck: Model is in a click loop, forcing a scroll to break it."}
                yield anti_stuck_scroll
                # FIX: actually execute the scroll in the browser — previously it
                # was only yielded to the frontend, so the page never moved and
                # the model kept seeing the same stuck state.
                await browser.execute_action({"action": "scroll", "delta": 300})
                
                history.append(f"Step {step + 1}: {action_summary}")
                if len(history) > 10:
                    history = history[-10:]
                    
                recent_frames.append({
                    "step": step + 1,
                    "frame": screenshot_b64,
                    "action_summary": action_summary,
                })
                if len(recent_frames) > 3:
                    recent_frames = recent_frames[-3:]
                last_step_frame = screenshot_b64
                step += 1
                continue

        # 7. Yield action to frontend (real-time stream)
        action["step"] = step + 1
        yield action

        # 8. Check if done
        if action.get("action") == "done":
            break

        # 8.5 ENHANCED CAPTCHA HANDLING - Automatic solving with stealth browser
        if action.get("action") == "ask_user":
            reason_lower = action.get("reason", "").lower()
            
            # Check if this is a CAPTCHA-related ask_user
            is_captcha = any(keyword in reason_lower for keyword in [
                "captcha", "recaptcha", "hcaptcha", "cloudflare", 
                "verify you are human", "prove you are not a robot",
                "i'm not a robot", "challenge"
            ])
            
            if is_captcha:
                logger.warning("="*80)
                logger.warning("🔒 CAPTCHA DETECTED")
                logger.warning("="*80)
                logger.warning(f"Reason: {action.get('reason', 'Unknown')}")
                
                # Check if browser has automatic CAPTCHA solving capability
                has_captcha_solver = hasattr(browser, 'solve_captcha')
                
                if has_captcha_solver:
                    logger.warning("🤖 Attempting automatic CAPTCHA solving with SeleniumBase...")
                    logger.warning("="*80)
                    
                    # Notify user about automatic solving attempt
                    yield {
                        "action": "wait",
                        "seconds": 1,
                        "reason": "🤖 CAPTCHA detected - attempting automatic solve with SeleniumBase...",
                        "step": step + 1,
                        "captcha_auto_solving": True
                    }
                    
                    # Try automatic CAPTCHA solving
                    try:
                        captcha_solved = await browser.solve_captcha(timeout=30)
                        
                        if captcha_solved:
                            logger.warning("="*80)
                            logger.warning("✅ CAPTCHA SOLVED AUTOMATICALLY!")
                            logger.warning("="*80)
                            
                            # Take new screenshot after solving
                            screenshot_b64 = await browser.screenshot_b64()
                            
                            # Notify user of success
                            yield {
                                "action": "wait",
                                "seconds": 1,
                                "reason": "✅ CAPTCHA solved automatically! Resuming automation...",
                                "step": step + 1,
                                "captcha_auto_solved": True
                            }
                            
                            # Update history
                            action_summary = json.dumps({
                                "action": "ask_user",
                                "reason": "CAPTCHA detected and solved automatically"
                            })
                            history.append(f"Step {step + 1}: {action_summary}")
                            if len(history) > 10:
                                history = history[-10:]
                            
                            recent_frames.append({
                                "step": step + 1,
                                "frame": screenshot_b64,
                                "action_summary": action_summary,
                            })
                            if len(recent_frames) > 3:
                                recent_frames = recent_frames[-3:]
                            
                            last_step_frame = screenshot_b64
                            step += 1
                            continue
                        else:
                            logger.warning("❌ Automatic CAPTCHA solving failed - falling back to manual mode")
                    except Exception as e:
                        logger.error(f"Automatic CAPTCHA solving error: {e}", exc_info=True)
                        logger.warning("Falling back to manual CAPTCHA resolution...")
                
                # Fallback: Manual CAPTCHA resolution (original behavior)
                logger.warning("Entering passive monitoring mode for manual resolution...")
                logger.warning("="*80)
                
                # Notify user ONCE about CAPTCHA
                yield {
                    "action": "ask_user",
                    "reason": "🔒 CAPTCHA DETECTED\n\nPlease solve it manually. I will monitor passively and resume automatically once solved.\n\nTimeout: 2 minutes",
                    "step": step + 1,
                    "captcha_mode": True,
                    "captcha_detected_at": step + 1
                }
                
                # Enter passive monitoring (no AI interference)
                try:
                    captcha_resolved, final_screenshot = await _monitor_captcha_resolution(
                        browser=browser,
                        initial_screenshot=screenshot_b64,
                        max_wait_seconds=120  # Wait up to 2 minutes
                    )
                except Exception as e:
                    logger.error(f"CAPTCHA monitoring failed with exception: {e}", exc_info=True)
                    captcha_resolved = False
                    final_screenshot = screenshot_b64
                
                if captcha_resolved:
                    logger.warning("="*80)
                    logger.warning("✅ CAPTCHA RESOLVED - RESUMING AUTOMATION")
                    logger.warning("="*80)
                    
                    # Update screenshot to post-CAPTCHA state
                    screenshot_b64 = final_screenshot
                    
                    # Notify user that automation is resuming
                    yield {
                        "action": "wait",
                        "seconds": 1,
                        "reason": "✅ CAPTCHA solved! Resuming automation...",
                        "step": step + 1,
                        "captcha_resolved": True
                    }
                    
                    # Update history with CAPTCHA resolution
                    action_summary = json.dumps({
                        "action": "ask_user",
                        "reason": "CAPTCHA detected and resolved by user"
                    })
                    history.append(f"Step {step + 1}: {action_summary}")
                    if len(history) > 10:
                        history = history[-10:]
                    
                    recent_frames.append({
                        "step": step + 1,
                        "frame": screenshot_b64,
                        "action_summary": action_summary,
                    })
                    if len(recent_frames) > 3:
                        recent_frames = recent_frames[-3:]
                    
                    last_step_frame = screenshot_b64
                    step += 1
                    continue
                else:
                    # CAPTCHA not resolved within timeout
                    logger.warning("="*80)
                    logger.warning("❌ CAPTCHA RESOLUTION FAILED")
                    logger.warning("="*80)
                    logger.warning("Possible causes:")
                    logger.warning("  1. User did not solve CAPTCHA within 2 minutes")
                    logger.warning("  2. CAPTCHA solving failed")
                    logger.warning("  3. Site is blocking automation")
                    logger.warning("="*80)
                    
                    yield {
                        "action": "error",
                        "reason": "❌ CAPTCHA could not be resolved within 2 minutes.\n\nThe site may be blocking automation.\n\nConsider using an alternative source (arXiv, Semantic Scholar, Europe PMC).",
                        "step": step + 1,
                        "captcha_failed": True
                    }
                    break
            else:
                # Non-CAPTCHA ask_user (e.g., MFA, manual decision)
                monitor_note = "Monitoring screen for manual resolution."
                monitor_wait = 2.0

                if pause_event:
                    try:
                        await asyncio.wait_for(pause_event.wait(), timeout=monitor_wait)
                        if pause_event.is_set():
                            pause_event.clear()
                            monitor_note = "User confirmed resolution; continuing."
                    except asyncio.TimeoutError:
                        monitor_note = "Awaiting manual resolution; continuing visual monitoring."
                else:
                    await asyncio.sleep(monitor_wait)

                action_summary = json.dumps({"action": "ask_user", "reason": action.get("reason", "")})
                history.append(f"Step {step + 1}: {action_summary} -> ({monitor_note})")
                if len(history) > 10:
                    history = history[-10:]

                recent_frames.append({
                    "step": step + 1,
                    "frame": screenshot_b64,
                    "action_summary": action_summary + f" ({monitor_note})",
                })
                if len(recent_frames) > 3:
                    recent_frames = recent_frames[-3:]

                last_step_frame = screenshot_b64
                step += 1
                continue

        # 9. Execute action
        result = await browser.execute_action(action)
        if result.startswith("ERROR"):
            yield {"action": "error", "reason": f"Browser action failed: {result}"}
            break

        # FIX: set query_typed only after the browser confirms the keystroke
        # was delivered.  Setting it before execute_action (as it was in the
        # forced-type block) caused the stall guard to suppress a re-try even
        # when the text was never actually typed.
        if action.get("action") == "type" and result == "OK":
            query_typed = True
            # Extra wait after typing to allow UI to settle (e.g. cursor blink, prediction list)
            await asyncio.sleep(0.5)

        # Reset query_typed on navigation — a new page load clears the search
        # bar, so the flag must not carry over to a fresh context.
        if action.get("action") == "navigate":
            query_typed = False

        if result == "DONE":
            break

        # 10. Update history
        action_summary = json.dumps({k: action[k] for k in ("action", "reason") if k in action})
        history.append(f"Step {step + 1}: {action_summary}")
        if len(history) > 10:
            history = history[-10:]

        # Keep last 3 frames (we use 2 in context, keep 1 extra for rolling)
        recent_frames.append({
            "step": step + 1,
            "frame": screenshot_b64,
            "action_summary": action_summary,
        })
        if len(recent_frames) > 3:
            recent_frames = recent_frames[-3:]

        last_step_frame = screenshot_b64
        step += 1

    logger.info(f"Vision loop completed after {step} steps")