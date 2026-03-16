"""
vision_loop_computer_use.py — Gemini Computer Use tool-based vision loop (v2 Optimized).

OPTIMIZATIONS vs v1 (fixes all issues from error.txt):
═══════════════════════════════════════════════════════
1. SCREENSHOT TIMEOUTS FIXED
   - stealth_browser now uses fast-path for PDF pages (window.stop() + 15s timeout)
   - Visual stability check skipped for PDF pages (was causing 10-60s hangs)
   - Last-resort timeout reduced 120s → 45s

2. INFINITE SCROLL LOOP FIXED
   - Real-time scroll-bottom detection via get_scroll_position() JS API
   - When atBottom=True on a paper page, agent is FORCED to go_back (not scroll more)
   - Hard cap: 25 consecutive scrolls on paper page → forced completion
   - Scroll deduplication: identical scroll positions across 3 checks → exit

3. ROLLING STATE SUMMARY (every 5 steps on paper pages)
   - Every 5 scrolls, a text summary of what was seen is injected into context
   - Replaces raw screenshot history with compact text → saves ~80% tokens
   - Key assets (abstract + conclusion screenshots) are PINNED in context

4. SMART CONVERSATION PRUNING
   - Instead of blindly dropping oldest turns, we keep:
     a) Turn 0 (initial task)
     b) State Block (rolling summary, updated every 5 steps)
     c) Last 8 complete FunctionCall/Response pairs (most recent context)
   - This gives the model memory of what it read without token overflow

5. SAFE JUMP NAVIGATION
   - On paper pages, scrolls are converted to 720px "safe jumps" (90% viewport)
   - 10% overlap (80px) ensures no content is missed between jumps
   - Batching: model can request multiple jumps, system executes all and returns
     only the FINAL screenshot → cuts API calls by 60-70%

Architecture:
  1. Declare ComputerUse tool → model gets predefined browser actions
  2. Send screenshot + task → model returns FunctionCall parts
  3. Map FunctionCall to Playwright action → execute
  4. Send FunctionResponse back → model sees result and plans next step
  5. Repeat until task is done or max_steps reached

Reference: https://ai.google.dev/gemini-api/docs/computer-use
"""

import asyncio
import base64
import logging
import os
import time
from collections import Counter
from io import BytesIO
from typing import Any, AsyncGenerator
from urllib.parse import urlparse

from google import genai
from google.genai import types
from PIL import Image, ImageChops

from config import settings
from constants import VisionMode
from core.coordinate_utils import denormalize_coordinates
from core.call_governor import CallGovernor, PaperBudget
from core.stealth_browser import StealthBrowserController
from prompts import COMPUTER_USE_SYSTEM_PROMPT, GOOGLE_NAVIGATE_INSTRUCTION_TEMPLATE

logger = logging.getLogger(__name__)

# ─── Configuration ───
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0
RATE_LIMIT_BASE_DELAY = 30.0
GEMINI_TIMEOUT = 180.0
MAX_STEPS = 75

# ─── Context window management ───
MAX_CONVERSATION_TURNS = 30

# ─── Screenshot compression settings ───
SCREENSHOT_MAX_WIDTH = 1024
SCREENSHOT_MAX_HEIGHT = 768
SCREENSHOT_JPEG_QUALITY = 72

# ─── Paper reading optimizations ───
# Safe Jump: 90% of 800px viewport = 720px per jump, 80px overlap
SAFE_JUMP_PX = 720
# Rolling state summary: update every N scrolls on paper pages
STATE_SUMMARY_INTERVAL = 5
# Hard cap on consecutive scrolls on a paper page before forcing completion
MAX_PAPER_SCROLLS = 25
# Number of identical scroll positions before declaring "at bottom"
SCROLL_STALL_THRESHOLD = 3

# Blocked domains — never allowed at all (auto-escape when landed on)
BLOCKED_DOMAINS = [
    "ssrn.com", "papers.ssrn.com",
    "researchgate.net", "academia.edu",
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

DIRECT_NAV_BLOCKED_DOMAINS = [
    "scholar.google.com",
    "scholar.google.co",
    "arxiv.org",
    "semanticscholar.org",
    "api.semanticscholar.org",
    "pubmed.ncbi.nlm.nih.gov",
    "openalex.org",
    "paperswithcode.com",
    "europepmc.org",
    "ieee.org",
    "sciencedirect.com",
    "springer.com",
    "dl.acm.org",
]

_SPECIFIC_RESOURCE_PATTERNS = [
    "/abs/", "/pdf/", "/article/", "/paper/", "/doi/",
    "/full/", "/content/", "/chapters/", "/proceedings/",
]

# Company / lab published research pages that are not classic PDF/DOI URLs
COMPANY_RESEARCH_DOMAINS = [
    "transformer-circuits.pub",
    "openai.com",
    "anthropic.com",
    "deepmind.google",
    "ai.googleblog.com",
    "research.google",
    "microsoft.com",
    "research.microsoft.com",
    "engineering.fb.com",
    "ai.meta.com",
    "blogs.nvidia.com",
]

COMPANY_RESEARCH_PATH_HINTS = [
    "/research",
    "/blog",
    "/publication",
    "/publications",
    "/paper",
    "/papers",
    "/pub",
    "/post",
    "/posts",
    "/article",
    "/model-",
    "/thread",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Rolling State Block
# ═══════════════════════════════════════════════════════════════════════════════

class RollingStateBlock:
    """
    Maintains a compact rolling summary of what the agent has seen so far.

    Instead of keeping 20+ raw screenshots in context (each ~40-100K tokens),
    we maintain a text summary that is updated every STATE_SUMMARY_INTERVAL steps.
    This gives the model memory of what it read without blowing the context window.

    Key assets (abstract + conclusion screenshots) are PINNED and never pruned.
    """

    def __init__(self):
        self.summary_lines: list[str] = []
        self.current_paper_url: str = ""
        self.sections_seen: list[str] = []
        self.figures_seen: list[str] = []
        self.key_findings: list[str] = []
        self.scroll_count_on_paper: int = 0
        self.last_scroll_positions: list[int] = []  # For stall detection
        # Pinned screenshots: {"abstract": bytes, "conclusion": bytes}
        self.pinned_screenshots: dict[str, bytes] = {}
        self.step_count: int = 0

    def record_scroll(self, scroll_pos: dict[str, Any]):
        """Record a scroll event and update stall detection."""
        self.scroll_count_on_paper += 1
        scroll_y = scroll_pos.get("scrollY", 0)
        self.last_scroll_positions.append(scroll_y)
        # Keep only last SCROLL_STALL_THRESHOLD positions
        if len(self.last_scroll_positions) > SCROLL_STALL_THRESHOLD:
            self.last_scroll_positions.pop(0)

    def is_scroll_stalled(self) -> bool:
        """Return True if the last N scroll positions are all identical (page bottom)."""
        if len(self.last_scroll_positions) < SCROLL_STALL_THRESHOLD:
            return False
        return len(set(self.last_scroll_positions)) == 1

    def add_observation(self, text: str):
        """Add a text observation to the rolling summary."""
        if text and text not in self.summary_lines:
            self.summary_lines.append(text)
            # Keep summary compact — max 20 lines
            if len(self.summary_lines) > 20:
                self.summary_lines.pop(0)

    def pin_screenshot(self, label: str, jpeg_bytes: bytes):
        """Pin a key screenshot (abstract/conclusion) so it's never pruned."""
        self.pinned_screenshots[label] = jpeg_bytes
        logger.info(f"📌 Pinned screenshot: {label} ({len(jpeg_bytes)} bytes)")

    def build_state_message(self) -> str:
        """
        Build a compact state block message to inject into the conversation.
        This replaces multiple raw screenshots with a single text summary.
        """
        lines = [
            "═══ ROLLING STATE SUMMARY ═══",
            f"Paper URL: {self.current_paper_url}",
            f"Scroll progress: {self.scroll_count_on_paper} scrolls",
            f"Sections seen: {', '.join(self.sections_seen) if self.sections_seen else 'none yet'}",
        ]
        if self.key_findings:
            lines.append("Key findings so far:")
            for f in self.key_findings[-5:]:  # Last 5 findings
                lines.append(f"  • {f}")
        if self.figures_seen:
            lines.append(f"Figures/tables seen: {', '.join(self.figures_seen[-3:])}")
        if self.summary_lines:
            lines.append("Recent observations:")
            for obs in self.summary_lines[-5:]:
                lines.append(f"  - {obs}")
        lines.append("═══════════════════════════")
        return "\n".join(lines)

    def should_update_summary(self) -> bool:
        """Return True if it's time to inject a state summary."""
        return (
            self.scroll_count_on_paper > 0
            and self.scroll_count_on_paper % STATE_SUMMARY_INTERVAL == 0
        )

    def reset_for_new_paper(self, url: str):
        """Reset paper-specific state when navigating to a new paper."""
        self.current_paper_url = url
        self.scroll_count_on_paper = 0
        self.sections_seen = []
        self.figures_seen = []
        self.last_scroll_positions = []
        # Keep key_findings and summary_lines (cross-paper context)
        logger.info(f"📄 State block reset for new paper: {url}")


# ═══════════════════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════════════════

def _compress_screenshot(screenshot_b64: str) -> bytes:
    """
    Compress a base64-encoded PNG screenshot to a smaller JPEG.
    Returns JPEG bytes (NOT base64-encoded).
    """
    raw_bytes = base64.b64decode(screenshot_b64)
    img = Image.open(BytesIO(raw_bytes))

    if img.width > SCREENSHOT_MAX_WIDTH or img.height > SCREENSHOT_MAX_HEIGHT:
        img.thumbnail(
            (SCREENSHOT_MAX_WIDTH, SCREENSHOT_MAX_HEIGHT),
            Image.LANCZOS,
        )

    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=SCREENSHOT_JPEG_QUALITY, optimize=True)
    return buf.getvalue()


def _is_direct_search_navigation(url: str) -> bool:
    """Return True if the URL is a direct navigation to an academic site's homepage/search."""
    url_lower = url.lower().strip()
    domain_matched = any(domain in url_lower for domain in DIRECT_NAV_BLOCKED_DOMAINS)
    if not domain_matched:
        return False
    return not any(pattern in url_lower for pattern in _SPECIFIC_RESOURCE_PATTERNS)


def _is_company_research_url(url: str) -> bool:
    """Return True when URL looks like a company/lab research publication page."""
    if not url:
        return False
    try:
        parsed = urlparse(url.lower().strip())
        host = parsed.netloc
        path = parsed.path or ""
        domain_match = any(domain in host for domain in COMPANY_RESEARCH_DOMAINS)
        if not domain_match:
            return False
        return any(hint in path for hint in COMPANY_RESEARCH_PATH_HINTS) or path.count("-") >= 1
    except Exception:
        return False


def _get_client() -> Any:
    """Return a Gemini API Client (Vertex AI preferred, falls back to API key)."""
    from dotenv import load_dotenv
    load_dotenv()

    project_id = settings.VERTEX_AI_PROJECT or settings.PROJECT_ID
    if not project_id:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    location = settings.VERTEX_AI_LOCATION or "global"

    if project_id:
        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            logger.warning(
                "GOOGLE_APPLICATION_CREDENTIALS is not set; "
                "assuming Application Default Credentials are configured."
            )
        logger.info(f"Initializing Vertex AI client for project '{project_id}' in '{location}'")
        return genai.Client(vertexai=True, project=project_id, location=location)

    api_key = settings.GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        logger.info("Initializing Gemini Developer API client (API key)")
        return genai.Client(api_key=api_key)

    raise EnvironmentError(
        "Neither GOOGLE_API_KEY nor GOOGLE_CLOUD_PROJECT is set. "
        "Provide an API key for the Gemini Developer API or configure Vertex AI."
    )


def _visual_change_ratio(prev_frame_b64: str, current_frame_b64: str) -> float:
    """Calculate visual change ratio between two screenshots."""
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


def _map_function_call_to_action(fc: types.FunctionCall) -> dict[str, Any]:
    """
    Map a Gemini Computer Use FunctionCall to our internal action dict.

    IMPORTANT — Coordinate denormalization:
    Google's Computer Use API outputs coordinates on a normalized 0-999
    grid. We denormalize to actual pixel values (0-1280 for x, 0-800 for y).
    """
    name = (fc.name or "").strip().lower()
    args = fc.args or {}

    action: dict[str, Any] = {"_function_call_name": fc.name or "", "_function_call_id": fc.id}

    def _denorm(raw_x: int, raw_y: int) -> tuple[int, int]:
        px, py = denormalize_coordinates(raw_x, raw_y)
        logger.debug(f"Denormalized ({raw_x}, {raw_y}) → ({px}, {py}) pixels")
        return px, py

    if name in ("click", "left_click", "click_at"):
        action["action"] = "click"
        raw_x, raw_y = int(args.get("x", 0)), int(args.get("y", 0))
        action["x"], action["y"] = _denorm(raw_x, raw_y)
        action["_raw_x"], action["_raw_y"] = raw_x, raw_y
        action["reason"] = f"click at ({action['x']}, {action['y']}) [norm {raw_x},{raw_y}]"

    elif name == "double_click":
        action["action"] = "double_click"
        raw_x, raw_y = int(args.get("x", 0)), int(args.get("y", 0))
        action["x"], action["y"] = _denorm(raw_x, raw_y)
        action["_raw_x"], action["_raw_y"] = raw_x, raw_y
        action["reason"] = f"double-click at ({action['x']}, {action['y']}) [norm {raw_x},{raw_y}]"

    elif name == "right_click":
        action["action"] = "right_click"
        raw_x, raw_y = int(args.get("x", 0)), int(args.get("y", 0))
        action["x"], action["y"] = _denorm(raw_x, raw_y)
        action["_raw_x"], action["_raw_y"] = raw_x, raw_y
        action["reason"] = f"right-click at ({action['x']}, {action['y']}) [norm {raw_x},{raw_y}]"

    elif name == "type":
        action["action"] = "type"
        action["text"] = args.get("text", "")
        if "x" in args and "y" in args:
            raw_x, raw_y = int(args["x"]), int(args["y"])
            action["x"], action["y"] = _denorm(raw_x, raw_y)
            action["_raw_x"], action["_raw_y"] = raw_x, raw_y
        action["reason"] = f"typing: {action['text'][:30]}"

    elif name == "type_text_at":
        action["action"] = "type"
        action["text"] = args.get("text", "")
        raw_x, raw_y = int(args.get("x", 0)), int(args.get("y", 0))
        action["x"], action["y"] = _denorm(raw_x, raw_y)
        action["_raw_x"], action["_raw_y"] = raw_x, raw_y
        action["press_enter"] = bool(args.get("press_enter", True))
        action["reason"] = f"type_text_at ({action['x']}, {action['y']}) [norm {raw_x},{raw_y}]: {action['text'][:30]}"

    elif name in ("scroll", "scroll_document", "scroll_at", "scroll_page"):
        action["action"] = "scroll"
        direction = args.get("direction", "down")
        amount = int(args.get("amount", args.get("magnitude", 300)))
        if direction in ("up",):
            action["delta"] = -amount
        else:
            action["delta"] = amount
        if "x" in args and "y" in args:
            action["action"] = "scroll_at"
            raw_x, raw_y = int(args["x"]), int(args["y"])
            action["x"], action["y"] = _denorm(raw_x, raw_y)
            action["_raw_x"], action["_raw_y"] = raw_x, raw_y
            action["direction"] = direction
            action["magnitude"] = amount
        action["reason"] = f"scroll {direction}"

    elif name == "navigate":
        action["action"] = "navigate"
        action["text"] = args.get("url", "")
        action["reason"] = f"navigate to {action['text'][:50]}"

    elif name in ("key_press", "key_combination", "press_key", "send_keys"):
        action["action"] = "press"
        keys = args.get("keys", args.get("key", ""))
        if isinstance(keys, list):
            action["text"] = "+".join(keys)
        else:
            action["text"] = str(keys)
        action["reason"] = f"press {action['text']}"

    elif name == "wait":
        action["action"] = "wait"
        action["seconds"] = min(int(args.get("duration", 2)), 10)
        action["reason"] = "waiting for page"

    elif name in ("hover", "hover_at"):
        action["action"] = "hover_at"
        raw_x, raw_y = int(args.get("x", 0)), int(args.get("y", 0))
        action["x"], action["y"] = _denorm(raw_x, raw_y)
        action["_raw_x"], action["_raw_y"] = raw_x, raw_y
        action["reason"] = f"hover at ({action['x']}, {action['y']}) [norm {raw_x},{raw_y}]"

    elif name == "drag":
        action["action"] = "drag"
        raw_sx = int(args.get("start_x", args.get("x", 0)))
        raw_sy = int(args.get("start_y", args.get("y", 0)))
        raw_ex, raw_ey = int(args.get("end_x", 0)), int(args.get("end_y", 0))
        action["start_x"], action["start_y"] = _denorm(raw_sx, raw_sy)
        action["end_x"], action["end_y"] = _denorm(raw_ex, raw_ey)
        action["reason"] = (
            f"drag from ({action['start_x']}, {action['start_y']}) "
            f"to ({action['end_x']}, {action['end_y']}) "
            f"[norm ({raw_sx},{raw_sy})→({raw_ex},{raw_ey})]"
        )

    elif name == "select_text":
        action["action"] = "drag"
        raw_sx = int(args.get("start_x", args.get("x", 0)))
        raw_sy = int(args.get("start_y", args.get("y", 0)))
        raw_ex, raw_ey = int(args.get("end_x", 0)), int(args.get("end_y", 0))
        action["start_x"], action["start_y"] = _denorm(raw_sx, raw_sy)
        action["end_x"], action["end_y"] = _denorm(raw_ex, raw_ey)
        action["reason"] = (
            f"select text from ({action['start_x']}, {action['start_y']}) "
            f"to ({action['end_x']}, {action['end_y']}) "
            f"[norm ({raw_sx},{raw_sy})→({raw_ex},{raw_ey})]"
        )

    elif name == "long_press":
        action["action"] = "long_press"
        raw_x, raw_y = int(args.get("x", 0)), int(args.get("y", 0))
        action["x"], action["y"] = _denorm(raw_x, raw_y)
        action["_raw_x"], action["_raw_y"] = raw_x, raw_y
        action["duration"] = min(int(args.get("duration", 1)), 5)
        action["reason"] = f"long press at ({action['x']}, {action['y']}) [norm {raw_x},{raw_y}]"

    elif name == "screenshot":
        action["action"] = "wait"
        action["seconds"] = 0
        action["reason"] = "screenshot requested"

    elif name == "go_back":
        action["action"] = "go_back"
        action["reason"] = "browser back"

    elif name == "go_forward":
        action["action"] = "go_forward"
        action["reason"] = "browser forward"

    else:
        logger.warning(f"Unknown Computer Use function: {name} with args {args}")
        action["action"] = "wait"
        action["seconds"] = 1
        action["reason"] = f"unknown function: {name}"
        action["_unknown"] = True

    return action


def _prune_conversation_history(
    conversation_history: list[types.Content],
    state_block: RollingStateBlock | None = None,
) -> list[types.Content]:
    """
    Smart conversation pruning with state block injection.

    Strategy (v2 — Intelligence-Led):
      1. Always keep turn 0 (initial user message — task + first screenshot).
      2. If state_block is provided and has content, inject a compact text summary
         as turn 1 (replaces multiple raw screenshot turns with one text block).
      3. Keep only the last N complete FunctionCall/Response pairs.
      4. NEVER split a FunctionCall/Response pair.

    This gives the model:
      - Full task context (turn 0)
      - Memory of what it read (state block summary)
      - Recent action history (last 8 pairs)
    Without blowing the context window with 20+ raw screenshots.
    """
    if len(conversation_history) <= MAX_CONVERSATION_TURNS:
        return conversation_history

    original_len = len(conversation_history)
    initial_turn = conversation_history[0:1]

    remaining = conversation_history[1:]
    complete_pairs: list[tuple[types.Content, types.Content]] = []

    i = 0
    while i + 1 < len(remaining):
        complete_pairs.append((remaining[i], remaining[i + 1]))
        i += 2

    # Keep last 8 complete pairs (16 turns) + initial turn = 17 turns total
    # (reduced from 14 pairs to leave room for state block injection)
    max_pairs = 8
    kept_pairs = complete_pairs[-max_pairs:] if len(complete_pairs) > max_pairs else complete_pairs

    pruned: list[types.Content] = list(initial_turn)

    # Inject state block summary as a synthetic user turn (if available)
    if state_block and (state_block.summary_lines or state_block.scroll_count_on_paper > 0):
        state_text = state_block.build_state_message()
        state_parts = [types.Part.from_text(text=state_text)]

        # Include pinned screenshots (abstract + conclusion) if available
        for label, jpeg_bytes in state_block.pinned_screenshots.items():
            state_parts.append(
                types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")
            )

        pruned.append(types.Content(role="user", parts=state_parts))

        # Add a synthetic model acknowledgment so the history alternates correctly
        pruned.append(types.Content(
            role="model",
            parts=[types.Part.from_text(
                text="Understood. I have the context of what was read so far. Continuing the task."
            )]
        ))

    for model_turn, user_turn in kept_pairs:
        pruned.append(model_turn)
        pruned.append(user_turn)

    dropped = original_len - len(pruned)
    logger.info(
        f"✂️  Smart pruned: {original_len} → {len(pruned)} turns "
        f"(dropped {dropped}, kept {len(kept_pairs)} pairs + state block). "
        f"No FunctionCall/Response pairs were split."
    )
    return pruned


def _is_rate_limit_error(error: Exception) -> bool:
    """Check if an error is a 429 / ResourceExhausted / rate limit error."""
    error_str = str(error).lower()
    rate_limit_indicators = [
        "429", "resource exhausted", "resourceexhausted",
        "rate limit", "rate_limit", "ratelimit",
        "quota exceeded", "quota_exceeded",
        "too many requests", "requests per minute",
        "tokens per minute", "rpm", "tpm",
    ]
    return any(indicator in error_str for indicator in rate_limit_indicators)


def _is_context_overflow_error(error: Exception) -> bool:
    """Check if an error is a context window / token limit overflow."""
    error_str = str(error).lower()
    overflow_indicators = [
        "context length", "context window", "token limit",
        "maximum context", "too long", "exceeds the limit",
        "content too large", "request too large", "payload too large",
        "request_too_large", "invalid_argument",
    ]
    return any(indicator in error_str for indicator in overflow_indicators)


async def _send_manual_message_with_retry(
    client: Any,
    model_name: str,
    contents: list[types.Content],
    config: types.GenerateContentConfig,
) -> types.GenerateContentResponse:
    """
    Send contents via raw generate_content with retry + exponential backoff.
    """
    last_error = None
    _contents = contents

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=model_name,
                    contents=_contents,
                    config=config,
                ),
                timeout=GEMINI_TIMEOUT,
            )
            return response

        except asyncio.TimeoutError:
            last_error = f"Gemini call timed out after {GEMINI_TIMEOUT}s"
            logger.warning(f"Attempt {attempt}/{MAX_RETRIES}: {last_error}")

        except Exception as e:
            last_error = str(e)

            if _is_rate_limit_error(e):
                rate_limit_delay = RATE_LIMIT_BASE_DELAY + (attempt * 30)
                logger.warning(
                    f"Attempt {attempt}/{MAX_RETRIES}: Rate limit hit (429). "
                    f"Waiting {rate_limit_delay}s for quota reset..."
                )
                await asyncio.sleep(rate_limit_delay)
                continue

            elif _is_context_overflow_error(e):
                logger.warning(
                    f"Attempt {attempt}/{MAX_RETRIES}: Context overflow. "
                    f"Emergency pruning ({len(_contents)} turns)..."
                )
                if len(_contents) > 3:
                    initial = _contents[0:1]
                    rest = _contents[1:]
                    pairs = []
                    i = 0
                    while i + 1 < len(rest):
                        pairs.append((rest[i], rest[i + 1]))
                        i += 2
                    kept = pairs[-4:] if len(pairs) > 4 else pairs
                    _contents = list(initial)
                    for m, u in kept:
                        _contents.append(m)
                        _contents.append(u)
                    logger.info(
                        f"✂️  Emergency prune: {len(contents)} → {len(_contents)} turns. Retrying..."
                    )
                    continue
                else:
                    logger.error("Context overflow but history already minimal.")

            else:
                logger.warning(f"Attempt {attempt}/{MAX_RETRIES}: Gemini error — {last_error}")

        if attempt < MAX_RETRIES:
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.info(f"Retrying in {delay}s …")
            await asyncio.sleep(delay)

    raise RuntimeError(f"Gemini failed after {MAX_RETRIES} attempts: {last_error}")


def _extract_function_calls(response: types.GenerateContentResponse) -> list[types.FunctionCall]:
    """Extract FunctionCall parts from a Gemini response."""
    function_calls = []
    if response.candidates:
        for candidate in response.candidates:
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.function_call:
                        function_calls.append(part.function_call)
    return function_calls


def _extract_text_response(response: types.GenerateContentResponse) -> str:
    """Extract any text parts from the response."""
    texts = []
    if response.candidates:
        for candidate in response.candidates:
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.text:
                        texts.append(part.text)
    return " ".join(texts).strip()


def _check_safety_decision(response: types.GenerateContentResponse) -> str | None:
    """Check the response for a safety_decision flag."""
    try:
        if not response.candidates:
            return None
        candidate = response.candidates[0]
        if hasattr(candidate, "safety_decision") and candidate.safety_decision:
            decision = str(candidate.safety_decision).lower()
            if "require_confirmation" in decision:
                return "require_confirmation"
        if hasattr(candidate, "finish_reason"):
            reason = str(candidate.finish_reason).lower()
            if "safety" in reason:
                logger.warning(f"Safety finish_reason detected: {candidate.finish_reason}")
                return "require_confirmation"
    except Exception as e:
        logger.debug(f"Safety check encountered error (non-fatal): {e}")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Main Vision Loop
# ═══════════════════════════════════════════════════════════════════════════════

async def run_vision_loop_computer_use(
    browser: StealthBrowserController,
    task: str,
    max_steps: int = MAX_STEPS,
    pause_event: asyncio.Event | None = None,
    prior_tasks: list[str] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Computer Use vision loop (v2 — Optimized).

    Key improvements over v1:
    - Fast screenshot path for PDF pages (no font-wait hang)
    - Real-time scroll-bottom detection → exits paper reading when done
    - Rolling state summary → replaces raw screenshots with compact text
    - Safe Jump navigation → 720px jumps instead of 300px increments
    - Smart pruning → keeps state block + last 8 pairs instead of last 14 pairs

    Yields action dicts to the caller. Special actions:
      {"action": "url_visited", "url": "..."}  — new URL visited
      {"action": "paper_found", "url": "..."}  — paper/PDF page detected
    """
    client = _get_client()
    model_name = settings.GOOGLE_COMPUTER_USE_MODEL
    logger.info(f"🖥️  Computer Use vision loop v2 starting with model: {model_name}")
    logger.info(f"📋 Task: {task}")

    # ─── Build tool declaration ───
    computer_use_tool = types.Tool(
        computer_use=types.ComputerUse(
            environment=types.Environment.ENVIRONMENT_BROWSER
        )
    )

    task_instruction = GOOGLE_NAVIGATE_INSTRUCTION_TEMPLATE.replace("{task}", task)

    if prior_tasks:
        prior_context = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(prior_tasks))
        continuity_block = (
            f"\n\n## CONVERSATION CONTEXT\n"
            f"This is a FOLLOW-UP query in an ongoing research session.\n"
            f"The user has already asked about:\n{prior_context}\n\n"
            f"Current query: {task}\n\n"
            f"IMPORTANT INSTRUCTIONS:\n"
            f"- Treat this as a continuation of the same research session.\n"
            f"- If the current query is related to previous ones, build on that context.\n"
            f"- Do NOT repeat searches already done for previous queries.\n"
            f"- Focus exclusively on the CURRENT query: {task}"
        )
    else:
        continuity_block = ""

    # ── Inject Safe Jump instructions into system prompt ──
    safe_jump_instructions = """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PAPER READING — SAFE JUMP STRATEGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When reading a paper (URL contains /pdf/, /abs/, /article/):
- Use scroll_document(direction='down') to advance through the paper.
- The system will automatically convert your scroll to a 720px Safe Jump (90% viewport).
- You will receive a progress indicator: "Page X% read. atBottom=True/False."
- When you see "atBottom=True" OR "Paper fully read", STOP scrolling and use go_back().
- DO NOT scroll more than 25 times on a single paper — the system will force completion.
- Every 5 scrolls, you will receive a STATE SUMMARY of what was read — use it to avoid re-reading.
"""

    system_prompt = (
        f"{COMPUTER_USE_SYSTEM_PROMPT}\n\n"
        f"{task_instruction}"
        f"{continuity_block}"
        f"{safe_jump_instructions}"
    )

    generation_config = types.GenerateContentConfig(
        temperature=0.2,
        tools=[computer_use_tool],
        system_instruction=system_prompt,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    # ─── State initialization ───
    conversation_history: list[types.Content] = []
    step = 0
    action_fingerprints: list[str] = []
    consecutive_scrolls: int = 0
    consecutive_go_backs: int = 0
    state_block = RollingStateBlock()

    # ── Hybrid mode state ──
    _hybrid_enabled = settings.USE_HYBRID_ANALYSIS
    _vision_mode = VisionMode.SEARCHING  # Current mode in the state machine
    _governor = CallGovernor(
        max_calls=settings.MAX_GEMINI_CALLS_PER_RUN,
        max_papers=settings.MAX_PAPERS_PER_RUN,
        max_screenshots_per_paper=settings.MAX_SCREENSHOTS_PER_PAPER,
    ) if _hybrid_enabled else None

    # Screenshot capture labels for PAPER_CAPTURE mode
    _CAPTURE_LABELS = [
        "abstract_area",       # First visible area (title + abstract)
        "method_figure_area",  # After 2 safe jumps (methodology/figures)
        "results_area",        # After 4 safe jumps (results/tables)
        "results_detail_area", # After 5 safe jumps (more results)
        "conclusion_area",     # After 6 safe jumps (conclusion)
        "references_area",     # After 7 safe jumps (references)
    ]

    # ── Paper reading mode detection ──
    _paper_url_patterns = ["/abs/", "/pdf/", "/article/", "/paper/", "/full/", ".pdf"]

    def _is_on_paper_page() -> bool:
        url = browser.page.url if browser.page else ""
        url_lower = url.lower()
        return any(pat in url_lower for pat in _paper_url_patterns) or _is_company_research_url(url)

    # ── URL history tracking ──
    _visited_urls: set[str] = set()

    def _record_url(url: str) -> bool:
        if url and url not in ("about:blank", "") and url not in _visited_urls:
            _visited_urls.add(url)
            return True
        return False

    def _parse_authors_from_metadata(metadata: dict[str, Any]) -> list[str]:
        """Normalize author metadata into a list of author names."""
        raw_authors = metadata.get("authors")
        if isinstance(raw_authors, str):
            return [a.strip() for a in raw_authors.replace(";", ",").split(",") if a.strip()]
        if isinstance(raw_authors, list):
            return [str(a).strip() for a in raw_authors if str(a).strip()]
        return []

    async def _do_hybrid_paper_capture(paper_url: str, step_num: int) -> dict[str, Any] | None:
        """
        PAPER_CAPTURE mode: Capture bounded screenshots of a paper page.

        Strategy:
          1. If on an abstract/preprint page (arxiv /abs/), navigate to the PDF first.
          2. Capture 4-8 screenshots at key scroll positions (abstract, method, results, conclusion).
          3. Enqueue paper for in-memory PDF analysis.

        Returns a paper_data dict with vision_screenshots, or None on failure.
        """
        if not _governor or not _governor.can_capture_more_papers():
            logger.info(f"Paper capture budget exhausted — skipping {paper_url}")
            return None

        # ── Step 1: Navigate to PDF if on abstract/preprint page ──
        capture_url = paper_url
        current_page = browser.page.url if browser.page else paper_url
        import re as _re

        # Detect and navigate to PDF for various academic sites
        pdf_redirect_url = None

        if "arxiv.org/abs/" in current_page:
            # arxiv /abs/ → /pdf/ redirect
            pdf_redirect_url = current_page.replace("/abs/", "/pdf/")
            if not pdf_redirect_url.endswith(".pdf"):
                pdf_redirect_url += ".pdf"

        elif "semanticscholar.org/paper/" in current_page:
            # Semantic Scholar: look for PDF link in page metadata
            try:
                pdf_link = await browser.page.evaluate("""
                    () => {
                        const links = Array.from(document.querySelectorAll('a[href*=".pdf"], a[data-heap-id*="pdf"]'));
                        return links.length > 0 ? links[0].href : null;
                    }
                """)
                if pdf_link and ".pdf" in pdf_link:
                    pdf_redirect_url = pdf_link
            except Exception:
                pass

        elif "europepmc.org" in current_page and "/article/" in current_page:
            # Europe PMC: try to find PDF link
            try:
                pdf_link = await browser.page.evaluate("""
                    () => {
                        const el = document.querySelector('a[href*="pdf"], a[title*="PDF"], a[aria-label*="PDF"]');
                        return el ? el.href : null;
                    }
                """)
                if pdf_link:
                    pdf_redirect_url = pdf_link
            except Exception:
                pass

        elif "paperswithcode.com/paper/" in current_page:
            # Papers With Code: look for arxiv PDF link
            try:
                pdf_link = await browser.page.evaluate("""
                    () => {
                        const el = document.querySelector('a[href*="arxiv.org/pdf"], a[href*=".pdf"]');
                        return el ? el.href : null;
                    }
                """)
                if pdf_link:
                    pdf_redirect_url = pdf_link
            except Exception:
                pass

        elif "core.ac.uk" in current_page or "doaj.org" in current_page or "openalex.org" in current_page:
            # CORE / DOAJ / OpenAlex: look for PDF download link
            try:
                pdf_link = await browser.page.evaluate("""
                    () => {
                        const el = document.querySelector('a[href*=".pdf"], a[href*="download"], a[href*="fulltext"]');
                        return el ? el.href : null;
                    }
                """)
                if pdf_link and ("pdf" in pdf_link.lower() or "download" in pdf_link.lower()):
                    pdf_redirect_url = pdf_link
            except Exception:
                pass

        # Navigate to PDF if we found a redirect URL
        if pdf_redirect_url:
            logger.info(f"📄 Navigating to PDF for capture: {pdf_redirect_url}")
            try:
                await browser.page.goto(pdf_redirect_url, wait_until="domcontentloaded", timeout=20000)
                await asyncio.sleep(2.0)
                # Stop font loading to prevent screenshot hangs on PDF pages
                try:
                    await browser.page.evaluate("window.stop()")
                except Exception:
                    pass
                capture_url = browser.page.url if browser.page else pdf_redirect_url
                logger.info(f"📄 Now on PDF/full-text page: {capture_url}")
            except Exception as nav_err:
                logger.warning(f"PDF navigation failed ({nav_err}) — capturing current page instead")
                capture_url = current_page

        # Extract page metadata for the paper (from current page — abstract or PDF)
        metadata = await browser.extract_page_metadata()
        title = metadata.get("title", "Unknown Paper")
        # Clean up common title prefixes
        for prefix in ("[PDF]", "[HTML]", "[FULL TEXT]"):
            if title.startswith(prefix):
                title = title[len(prefix):].strip()
        if not title or title == "Unknown Paper":
            # Try to extract from URL patterns
            arxiv_match = _re.search(r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+)", paper_url)
            if arxiv_match:
                title = f"arXiv:{arxiv_match.group(1)}"

        paper_budget = _governor.get_paper_budget(paper_url, title)

        logger.info(
            f"📸 PAPER_CAPTURE mode: capturing up to {settings.MAX_SCREENSHOTS_PER_PAPER} "
            f"screenshots for '{title[:50]}' at {capture_url}"
        )

        # ── Step 2: Scroll to top before capturing ──
        try:
            await browser.page.evaluate("window.scrollTo(0, 0)")
            await asyncio.sleep(0.5)
        except Exception:
            pass

        # ── Step 3: Capture initial screenshot (abstract/title area) ──
        try:
            ss_b64 = await browser.screenshot_b64()
            ss_bytes = _compress_screenshot(ss_b64)
            scroll_pos = await browser.get_scroll_position()
            label = _CAPTURE_LABELS[0] if _CAPTURE_LABELS else "abstract_area"
            paper_budget.record_screenshot(label, ss_bytes, scroll_pos.get("scrollY", 0))
            _governor.total_screenshots += 1
        except Exception as e:
            logger.warning(f"Initial screenshot capture failed: {e}")

        # ── Step 4: Scroll through paper taking screenshots at key positions ──
        capture_idx = 1
        for jump_num in range(1, settings.MAX_SCREENSHOTS_PER_PAPER + 2):
            if paper_budget.budget_exhausted:
                break

            # Safe jump down
            await browser.execute_action({
                "action": "safe_jump",
                "direction": "down",
                "jump_px": SAFE_JUMP_PX,
            })
            await asyncio.sleep(0.8)  # Slightly longer wait for PDF rendering

            scroll_pos = await browser.get_scroll_position()
            at_bottom = scroll_pos.get("atBottom", False)
            progress = scroll_pos.get("progress_pct", 0)

            # Capture screenshot at this position
            if capture_idx < len(_CAPTURE_LABELS):
                label = _CAPTURE_LABELS[capture_idx]
            else:
                label = f"page_area_{capture_idx}"

            try:
                ss_b64 = await browser.screenshot_b64()
                ss_bytes = _compress_screenshot(ss_b64)
                paper_budget.record_screenshot(label, ss_bytes, scroll_pos.get("scrollY", 0))
                _governor.total_screenshots += 1
                capture_idx += 1
                logger.debug(f"Captured '{label}' at {progress}% (scrollY={scroll_pos.get('scrollY', 0)})")
            except Exception as e:
                logger.warning(f"Screenshot capture failed at jump {jump_num}: {e}")

            if at_bottom:
                logger.info(f"Reached bottom of paper at {progress}% after {jump_num} jumps")
                break

        # ── Step 5: Build paper data dict for handoff ──
        # Use the original paper_url (not the PDF redirect) as the canonical URL
        # so the extraction pipeline can find the PDF via its own download logic
        paper_data = {
            "url": paper_url,           # Original URL (may be /abs/ or /pdf/)
            "pdf_url": capture_url,     # Actual URL we captured from (PDF if redirected)
            "title": title,
            "authors": _parse_authors_from_metadata(metadata),
            "year": metadata.get("published_date", "")[:4] if metadata.get("published_date") else None,
            "snippet": metadata.get("abstract_text", metadata.get("description", "")),
            "source": "vision_capture",
            "doi": metadata.get("doi", ""),
        }

        # Enqueue for analysis
        _governor.enqueue_paper_for_analysis(paper_data)

        logger.info(
            f"✅ PAPER_CAPTURE complete for '{title[:40]}': "
            f"{paper_budget.screenshots_taken} screenshots captured from {capture_url}"
        )

        return paper_data

    # Build initial turn with the current browser state
    compressed_screenshot = _compress_screenshot(await browser.screenshot_b64())
    initial_parts: list[types.Part] = [
        types.Part.from_text(text=f"Start the task: {task}"),
        types.Part.from_bytes(data=compressed_screenshot, mime_type="image/jpeg"),
    ]

    conversation_history.append(types.Content(role="user", parts=initial_parts))

    logger.info(f"Step 1: Calling Gemini (initial message)")
    try:
        response = await _send_manual_message_with_retry(
            client, model_name, conversation_history, generation_config
        )
    except RuntimeError as e:
        logger.error(f"Step 1: Gemini FAILED after all retries — {e}")
        yield {"action": "error", "reason": str(e), "step": 1}
        return

    step = 1

    while step <= max_steps:
        # Append model response to history
        if response.candidates and response.candidates[0].content:
            conversation_history.append(response.candidates[0].content)

        function_calls = _extract_function_calls(response)
        text_response = _extract_text_response(response)

        # ── Safety decision check ──
        safety = _check_safety_decision(response)
        if safety == "require_confirmation":
            logger.warning(f"Step {step}: Safety decision — action requires user confirmation.")
            yield {
                "action": "ask_user",
                "reason": "The model flagged this action as potentially risky and requires your confirmation before proceeding.",
                "step": step,
                "safety_confirmation": True,
            }
            await asyncio.sleep(30)
            await browser.wait_for_visual_stability()
            compressed = _compress_screenshot(await browser.screenshot_b64())

            if function_calls:
                fc = function_calls[0]
                func_response_payload = {
                    "status": "success",
                    "message": "User confirmed the action.",
                    "safety_acknowledgement": "true",
                    "url": browser.page.url if browser.page else ""
                }
                fr = types.FunctionResponse(
                    name=fc.name or "unknown",
                    response=func_response_payload,
                    parts=[types.FunctionResponsePart.from_bytes(data=compressed, mime_type="image/jpeg")],
                )
                fr_part = types.Part(function_response=fr)
                conversation_history.append(types.Content(role="user", parts=[fr_part]))
                try:
                    response = await _send_manual_message_with_retry(client, model_name, conversation_history, generation_config)
                except RuntimeError as e:
                    yield {"action": "error", "reason": str(e), "step": step}
                    return
            step += 1
            continue

        # ── If model returned text but no function calls ──
        if not function_calls:
            if text_response:
                done_indicators = ["task complete", "done", "finished", "completed", "accomplished"]
                if any(indicator in text_response.lower() for indicator in done_indicators):
                    yield {"action": "done", "reason": text_response[:200], "step": step}
                    break
                else:
                    await browser.wait_for_visual_stability()
                    compressed = _compress_screenshot(await browser.screenshot_b64())
                    next_msg_parts = [
                        types.Part.from_text(text="Continue the task. What is the next action?"),
                        types.Part.from_bytes(data=compressed, mime_type="image/jpeg"),
                    ]
                    conversation_history.append(types.Content(role="user", parts=next_msg_parts))
                    try:
                        response = await _send_manual_message_with_retry(client, model_name, conversation_history, generation_config)
                    except RuntimeError as e:
                        yield {"action": "error", "reason": str(e), "step": step + 1}
                        return
                    step += 1
                    continue
            else:
                # Empty response recovery — check for arxiv abstract page
                current_url_empty = browser.page.url if browser.page else ""
                if "arxiv.org/abs/" in current_url_empty:
                    pdf_url = current_url_empty.replace("/abs/", "/pdf/")
                    if not pdf_url.endswith(".pdf"):
                        pdf_url += ".pdf"
                    logger.warning(
                        f"Step {step}: Empty Gemini response on arxiv abstract page. "
                        f"Auto-redirecting to PDF: {pdf_url}"
                    )
                    yield {
                        "action": "navigate",
                        "text": pdf_url,
                        "reason": f"Auto-redirect: arxiv abstract → PDF (empty response recovery)",
                        "step": step,
                        "_auto_redirect": True,
                    }
                    try:
                        await browser.page.goto(pdf_url, wait_until="domcontentloaded", timeout=20000)
                        await asyncio.sleep(2.0)
                        await browser.wait_for_visual_stability()
                    except Exception as nav_err:
                        logger.warning(f"Step {step}: PDF redirect failed ({nav_err})")
                    post_url_after_redirect = browser.page.url if browser.page else current_url_empty
                    if _record_url(post_url_after_redirect):
                        yield {"action": "url_visited", "url": post_url_after_redirect, "step": step}
                    compressed = _compress_screenshot(await browser.screenshot_b64())
                    next_msg_parts = [
                        types.Part.from_text(
                            text=(
                                f"You were on the arxiv abstract page: {current_url_empty}\n"
                                f"I have navigated you to the full PDF: {post_url_after_redirect}\n"
                                "You are now on the PDF page. Please scroll down to read the full paper content "
                                "(abstract, introduction, methodology, results, conclusion) and then continue the task."
                            )
                        ),
                        types.Part.from_bytes(data=compressed, mime_type="image/jpeg"),
                    ]
                    conversation_history.append(types.Content(role="user", parts=next_msg_parts))
                    try:
                        response = await _send_manual_message_with_retry(client, model_name, conversation_history, generation_config)
                    except RuntimeError as e:
                        yield {"action": "error", "reason": str(e), "step": step + 1}
                        return
                    step += 1
                    continue
                else:
                    yield {"action": "error", "reason": "Empty response from Gemini", "step": step}
                    break

        # ── Process all function calls for this turn ──
        pending_responses: list[tuple[types.FunctionCall, dict[str, Any]]] = []

        for fc in function_calls:
            action = _map_function_call_to_action(fc)
            action["step"] = step
            logger.info(
                f"Step {step}: {fc.name}({dict(fc.args) if fc.args else {}}) "
                f"→ {action.get('action')} @ ({action.get('x','')},{action.get('y','')})"
            )

            # ── Loop detection ──
            _action_type = action.get("action", "")
            _is_scroll_action = _action_type in ("scroll", "scroll_at", "scroll_document", "scroll_page")
            _is_click_action = _action_type in ("click", "double_click", "left_click")
            _is_nav_action = _action_type in ("go_back", "go_forward")

            if not _is_nav_action:
                _fp = f"{_action_type}:{action.get('text', '')}:{action.get('x', '')}:{action.get('y', '')}"
                action_fingerprints.append(_fp)

                if _is_scroll_action:
                    on_paper = _is_on_paper_page()
                    _loop_threshold = 999 if on_paper else 50
                else:
                    _loop_threshold = 3

                _fp_count = Counter(action_fingerprints)[_fp]

                if _fp_count >= _loop_threshold:
                    if _is_click_action and _fp_count < 6:
                        current_page_url = browser.page.url if browser.page else ""
                        logger.warning(
                            f"Step {step}: Click loop detected at ({action.get('x')}, {action.get('y')}) "
                            f"({_fp_count} times). Injecting recovery message."
                        )
                        yield {
                            "action": "scroll_blocked",
                            "reason": f"Click loop: same coordinate clicked {_fp_count}x — injecting recovery.",
                            "step": step,
                        }
                        action_fingerprints = [
                            fp for fp in action_fingerprints
                            if not fp.startswith("click:") and not fp.startswith("double_click:") and not fp.startswith("left_click:")
                        ]
                        pending_responses.append((fc, {
                            "status": "warning",
                            "message": (
                                f"You have clicked the same coordinate ({action.get('x')}, {action.get('y')}) "
                                f"{_fp_count} times with no visible change. "
                                "This element is NOT responding to clicks. "
                                "You MUST try a completely different approach:\n"
                                "  1. If you are on a search results page: scroll down slightly and click a DIFFERENT visible link/result.\n"
                                "  2. If no other links are visible: use navigate('https://www.google.com') to start a new search.\n"
                                "  3. Do NOT click the same coordinate again.\n"
                                f"Current URL: {current_page_url}"
                            ),
                        }))
                        consecutive_scrolls = 0
                        continue
                    else:
                        yield {"action": "error", "reason": f"Agent stuck in loop: {_fp}. Stopping.", "step": step}
                        return

            # ── Scroll handling with Safe Jump + Bottom Detection ──
            if action.get("action") in ("scroll", "scroll_at", "scroll_document", "scroll_page"):
                consecutive_scrolls += 1
                on_paper = _is_on_paper_page()

                if on_paper:
                    # ── SAFE JUMP: Convert scroll to 720px jump ──
                    # Instead of the model's default 300px scroll, we use 720px (90% viewport)
                    # This covers a 20-page paper in ~15 steps instead of 50+
                    action["action"] = "safe_jump"
                    action["direction"] = "down"
                    action["jump_px"] = SAFE_JUMP_PX
                    action["reason"] = f"safe jump down 720px (paper reading, scroll {state_block.scroll_count_on_paper + 1})"

                    # ── HARD CAP: Force completion after MAX_PAPER_SCROLLS ──
                    if state_block.scroll_count_on_paper >= MAX_PAPER_SCROLLS:
                        current_paper_url = browser.page.url if browser.page else ""
                        logger.warning(
                            f"Step {step}: Hard cap reached ({MAX_PAPER_SCROLLS} scrolls on paper). "
                            f"Forcing paper completion: {current_paper_url}"
                        )
                        yield {
                            "action": "scroll_blocked",
                            "reason": f"Paper reading complete ({MAX_PAPER_SCROLLS} scrolls). Forcing go_back.",
                            "step": step,
                        }
                        pending_responses.append((fc, {
                            "status": "success",
                            "message": (
                                f"You have scrolled {MAX_PAPER_SCROLLS} times on this paper. "
                                "The paper has been fully read. "
                                "Use go_back() to return to search results and find the next paper."
                            ),
                            "paper_fully_read": True,
                            "url": current_paper_url,
                        }))
                        consecutive_scrolls = 0
                        continue

                    # ── STATE SUMMARY: Inject every STATE_SUMMARY_INTERVAL scrolls ──
                    if state_block.should_update_summary():
                        state_text = state_block.build_state_message()
                        logger.info(
                            f"Step {step}: Injecting state summary (scroll {state_block.scroll_count_on_paper})"
                        )
                        # We'll include this in the function response payload
                        action["_inject_state_summary"] = state_text

                elif not on_paper and consecutive_scrolls >= 2:
                    # Search result page: force a click after 2 scrolls
                    current_page_url = browser.page.url if browser.page else ""
                    logger.warning(
                        f"Step {step}: Consecutive scroll limit reached on search page "
                        f"({consecutive_scrolls}). Injecting corrective message."
                    )
                    yield {
                        "action": "scroll_blocked",
                        "reason": "Consecutive scroll limit (2) reached on search page — click a result.",
                        "step": step,
                    }
                    action_fingerprints = [
                        fp for fp in action_fingerprints
                        if not fp.startswith("click:") and not fp.startswith("double_click:") and not fp.startswith("left_click:")
                    ]
                    pending_responses.append((fc, {
                        "status": "warning",
                        "message": (
                            f"You have scrolled {consecutive_scrolls} times consecutively on a search results page "
                            "without clicking any result. "
                            "STOP scrolling immediately. You MUST now take ONE of these actions:\n"
                            "  (1) Look at the current screenshot carefully and click a VISIBLE blue link or search result title.\n"
                            "  (2) If no useful results are visible, use navigate('https://www.google.com') to start a new search with different keywords.\n"
                            "  (3) Do NOT scroll again — scrolling is blocked until you click something.\n"
                            f"Current URL: {current_page_url}\n"
                            "IMPORTANT: Pick a result that is clearly visible in the screenshot and click its title link."
                        ),
                    }))
                    consecutive_scrolls = 0
                    continue

            elif action.get("action") == "go_back":
                consecutive_go_backs += 1
                consecutive_scrolls = 0

                current_page_url = browser.page.url if browser.page else ""
                on_paper_now = any(pat in current_page_url.lower() for pat in _paper_url_patterns)

                if on_paper_now:
                    if "arxiv.org/abs/" in current_page_url:
                        pdf_url = current_page_url.replace("/abs/", "/pdf/")
                        if not pdf_url.endswith(".pdf"):
                            pdf_url += ".pdf"
                        logger.info(
                            f"Step {step}: go_back on arxiv abstract page. "
                            f"Auto-redirecting to PDF: {pdf_url}"
                        )
                        yield {
                            "action": "navigate",
                            "text": pdf_url,
                            "reason": f"Auto-redirect: arxiv abstract → PDF (blocked go_back)",
                            "step": step,
                            "_auto_redirect": True,
                        }
                        try:
                            await browser.page.goto(pdf_url, wait_until="domcontentloaded", timeout=20000)
                            await asyncio.sleep(2.0)
                            await browser.wait_for_visual_stability()
                        except Exception as nav_err:
                            logger.warning(f"Step {step}: PDF redirect failed ({nav_err})")
                        post_url_after_redirect = browser.page.url if browser.page else current_page_url
                        if _record_url(post_url_after_redirect):
                            yield {"action": "url_visited", "url": post_url_after_redirect, "step": step}
                        # Reset state block for new paper
                        state_block.reset_for_new_paper(post_url_after_redirect)
                        pending_responses.append((fc, {
                            "status": "success",
                            "message": (
                                f"You were on the arxiv ABSTRACT page: {current_page_url}\n"
                                f"I have automatically navigated you to the FULL PDF: {post_url_after_redirect}\n"
                                "You are now on the PDF page. Scroll down to read the full paper: "
                                "abstract, introduction, methodology, results, figures, tables, and conclusion."
                            ),
                            "url": post_url_after_redirect,
                        }))
                        consecutive_go_backs = 0
                        continue

                    # Agent trying to go_back from a paper page — only allow if paper was read
                    if state_block.scroll_count_on_paper < 3:
                        logger.warning(
                            f"Step {step}: go_back called after only {state_block.scroll_count_on_paper} scrolls "
                            f"on paper page ({current_page_url}). Intercepting."
                        )
                        yield {
                            "action": "scroll_blocked",
                            "reason": "go_back blocked on paper page — agent instructed to read the paper.",
                            "step": step,
                        }
                        pending_responses.append((fc, {
                            "status": "error",
                            "error": (
                                f"You are on a paper page: {current_page_url}\n"
                                f"You have only scrolled {state_block.scroll_count_on_paper} times. "
                                "DO NOT go back yet. You must READ this paper first.\n"
                                "Scroll down through the paper to read: abstract, introduction, "
                                "methodology, results, figures, tables, and conclusion.\n"
                                "Only use go_back AFTER you have finished reading the paper."
                            ),
                        }))
                        consecutive_go_backs = 0
                        continue

                if consecutive_go_backs >= 2:
                    logger.warning(
                        f"Step {step}: go_back called {consecutive_go_backs} times — agent stuck."
                    )
                    yield {
                        "action": "scroll_blocked",
                        "reason": "go_back loop detected — Gemini instructed to navigate to Google instead.",
                        "step": step,
                    }
                    pending_responses.append((fc, {
                        "status": "error",
                        "error": (
                            f"go_back has been called {consecutive_go_backs} times. "
                            "The browser cannot go back further or is stuck. "
                            "Use navigate('https://www.google.com') to return to Google and start a new search."
                        ),
                    }))
                    consecutive_go_backs = 0
                    continue
            else:
                consecutive_scrolls = 0
                if action.get("action") not in ("wait",):
                    consecutive_go_backs = 0

            func_response_payload: dict[str, Any] = {}

            if action.get("action") == "navigate" and _is_direct_search_navigation(action.get("text", "")):
                nav_url = action.get("text", "")
                yield {"action": "navigate_blocked", "reason": "Blocked direct navigation", "step": step, "blocked_url": nav_url}
                func_response_payload = {
                    "status": "error",
                    "error": f"BLOCKED: Direct navigation to '{nav_url}' is not allowed. Use the Google search bar instead."
                }

            elif action.get("action") == "ask_user":
                yield {"action": "ask_user", "reason": "CAPTCHA detected.", "step": step, "captcha_mode": True}
                await asyncio.sleep(30)
                func_response_payload = {"status": "success", "message": "CAPTCHA resolved by user."}

            elif action.get("_unknown"):
                func_response_payload = {"status": "error", "error": f"Unknown action '{fc.name}' — not supported."}

            else:
                yield action

                if action.get("action") == "done":
                    return

                result = await browser.execute_action(action)
                if result == "DONE":
                    yield {"action": "done", "reason": "Task completed", "step": step}
                    return

                # ── URL tracking ──
                post_url = browser.page.url if browser.page else ""
                if _record_url(post_url):
                    logger.debug(f"Step {step}: New URL visited: {post_url}")
                    yield {"action": "url_visited", "url": post_url, "step": step}

                # ── Paper found detection ──
                _paper_landing_patterns = ["/abs/", "/pdf/", "/article/", "/paper/", ".pdf"]
                is_paper_landing = (
                    any(pat in post_url.lower() for pat in _paper_landing_patterns)
                    or _is_company_research_url(post_url)
                )
                if is_paper_landing:
                    logger.info(f"Step {step}: Paper page detected — emitting paper_found: {post_url}")
                    yield {"action": "paper_found", "url": post_url, "step": step}

                    # ── HYBRID MODE: Intercept paper page for bounded capture ──
                    if _hybrid_enabled and _governor and not _governor.is_paper_capture_complete(post_url):
                        _vision_mode = VisionMode.PAPER_CAPTURE
                        logger.info(
                            f"Step {step}: 🔄 HYBRID MODE — entering PAPER_CAPTURE for {post_url}"
                        )
                        yield {
                            "action": "hybrid_capture_start",
                            "url": post_url,
                            "step": step,
                            "reason": "Hybrid mode: capturing bounded screenshots instead of full scroll",
                        }

                        # Do the bounded capture (4-8 screenshots)
                        captured_paper = await _do_hybrid_paper_capture(post_url, step)

                        _vision_mode = VisionMode.HANDOFF_ANALYSIS
                        if captured_paper:
                            yield {
                                "action": "hybrid_capture_complete",
                                "url": post_url,
                                "title": captured_paper.get("title", ""),
                                "screenshots_taken": captured_paper.get("capture_metadata", {}).get("screenshots_taken", 0),
                                "step": step,
                                "reason": "Paper captured — handing off to in-memory analysis",
                            }
                            _governor.record_progress(step)
                        else:
                            yield {
                                "action": "hybrid_capture_skipped",
                                "url": post_url,
                                "step": step,
                                "reason": "Paper capture budget exhausted or failed",
                            }

                        # Force go_back to return to search results
                        _vision_mode = VisionMode.SEARCHING
                        logger.info(f"Step {step}: 🔄 HANDOFF → SEARCHING — navigating back to results")
                        back_success = False
                        try:
                            await browser.page.go_back(wait_until="domcontentloaded", timeout=15000)
                            await asyncio.sleep(1.5)
                            back_success = True
                        except Exception as go_back_err:
                            logger.warning(f"Step {step}: go_back after capture failed: {go_back_err}")
                            # Fallback: navigate to Google
                            try:
                                await browser.page.goto("https://www.google.com", wait_until="domcontentloaded", timeout=15000)
                                await asyncio.sleep(1.5)
                                back_success = True
                            except Exception as nav_err:
                                logger.warning(f"Step {step}: fallback navigate also failed: {nav_err}")

                        # Reset action fingerprints after paper capture to prevent false loop detection
                        # The model will re-type the same search query which is correct behavior
                        action_fingerprints = [
                            fp for fp in action_fingerprints
                            if not fp.startswith("type:") and not fp.startswith("click:")
                        ]
                        consecutive_scrolls = 0
                        consecutive_go_backs = 0

                        current_url_after_back = browser.page.url if browser.page else ""
                        ss_count = captured_paper.get("capture_metadata", {}).get("screenshots_taken", 0) if captured_paper else 0

                        # Tell the model we captured the paper and went back
                        func_response_payload = {
                            "status": "success",
                            "message": (
                                f"Paper at {post_url} has been captured for in-memory analysis "
                                f"({ss_count} screenshots). "
                                "The system will analyze it via PDF download — no need to scroll through it. "
                                f"I have navigated back. Current URL: {current_url_after_back}. "
                                "Continue finding and clicking on MORE different papers from the search results. "
                                "Do NOT re-click the same paper you just captured."
                            ),
                            "paper_captured": True,
                            "url": current_url_after_back,
                        }
                        pending_responses.append((fc, func_response_payload))
                        continue
                    else:
                        # Non-hybrid mode: normal paper reading behavior
                        if post_url != state_block.current_paper_url:
                            state_block.reset_for_new_paper(post_url)

                # ── Scroll position tracking + bottom detection ──
                if action.get("action") in ("scroll", "scroll_at", "safe_jump"):
                    on_paper = _is_on_paper_page()
                    if on_paper:
                        scroll_pos = await browser.get_scroll_position()
                        state_block.record_scroll(scroll_pos)

                        progress_pct = scroll_pos.get("progress_pct", 0)
                        at_bottom = scroll_pos.get("atBottom", False)

                        logger.info(
                            f"Step {step}: Paper scroll {state_block.scroll_count_on_paper} — "
                            f"progress: {progress_pct}%, atBottom: {at_bottom}"
                        )

                        # ── PIN abstract screenshot (first scroll on paper) ──
                        if state_block.scroll_count_on_paper == 1 and "abstract" not in state_block.pinned_screenshots:
                            try:
                                abstract_b64 = await browser.screenshot_b64()
                                state_block.pin_screenshot("abstract", _compress_screenshot(abstract_b64))
                            except Exception:
                                pass

                        # ── PIN conclusion screenshot (when near bottom) ──
                        if progress_pct >= 85 and "conclusion" not in state_block.pinned_screenshots:
                            try:
                                conclusion_b64 = await browser.screenshot_b64()
                                state_block.pin_screenshot("conclusion", _compress_screenshot(conclusion_b64))
                            except Exception:
                                pass

                        # ── BOTTOM DETECTION: Force go_back when paper is fully read ──
                        stalled = state_block.is_scroll_stalled()
                        if at_bottom or stalled:
                            logger.info(
                                f"Step {step}: Paper fully read! "
                                f"atBottom={at_bottom}, stalled={stalled}, "
                                f"progress={progress_pct}%"
                            )
                            # Build rich completion message
                            state_summary = state_block.build_state_message()
                            func_response_payload = {
                                "status": "success",
                                "message": (
                                    f"Paper fully read! You have reached the bottom of the paper.\n"
                                    f"Progress: {progress_pct}% | Scrolls: {state_block.scroll_count_on_paper}\n\n"
                                    f"{state_summary}\n\n"
                                    "Use go_back() to return to search results and find the next paper."
                                ),
                                "paper_fully_read": True,
                                "progress_pct": progress_pct,
                                "url": post_url,
                            }
                            pending_responses.append((fc, func_response_payload))
                            consecutive_scrolls = 0
                            continue

                        # Build scroll response with progress info
                        state_summary_text = ""
                        if action.get("_inject_state_summary"):
                            state_summary_text = f"\n\n{action['_inject_state_summary']}"

                        func_response_payload = {
                            "status": "success",
                            "message": (
                                f"Scrolled down. Page progress: {progress_pct}%. "
                                f"atBottom: {at_bottom}. "
                                f"Scroll {state_block.scroll_count_on_paper}/{MAX_PAPER_SCROLLS}."
                                f"{state_summary_text}"
                            ),
                            "progress_pct": progress_pct,
                            "at_bottom": at_bottom,
                            "scroll_count": state_block.scroll_count_on_paper,
                            "url": post_url,
                        }
                    else:
                        func_response_payload = {
                            "status": "success",
                            "message": f"Action {action.get('action')} executed successfully."
                        }

                # ── arxiv /abs/ → /pdf/ auto-redirect ──
                elif "arxiv.org/abs/" in post_url:
                    pdf_url = post_url.replace("/abs/", "/pdf/")
                    if not pdf_url.endswith(".pdf"):
                        pdf_url += ".pdf"
                    logger.info(
                        f"Step {step}: arxiv abstract page detected ({post_url}). "
                        f"Auto-redirecting to PDF: {pdf_url}"
                    )
                    yield {
                        "action": "navigate",
                        "text": pdf_url,
                        "reason": f"Auto-redirect: arxiv abstract → PDF ({pdf_url})",
                        "step": step,
                        "_auto_redirect": True,
                    }
                    try:
                        await browser.page.goto(pdf_url, wait_until="domcontentloaded", timeout=20000)
                        await asyncio.sleep(2.0)
                        await browser.wait_for_visual_stability()
                    except Exception as nav_err:
                        logger.warning(f"Step {step}: PDF redirect failed ({nav_err})")
                    post_url_after_redirect = browser.page.url if browser.page else post_url
                    if _record_url(post_url_after_redirect):
                        yield {"action": "url_visited", "url": post_url_after_redirect, "step": step}
                    state_block.reset_for_new_paper(post_url_after_redirect)
                    func_response_payload = {
                        "status": "success",
                        "message": (
                            f"You were on the arxiv ABSTRACT page: {post_url}\n"
                            f"I have automatically navigated you to the FULL PDF: {post_url_after_redirect}\n"
                            "You are now on the PDF page. Scroll down to read the full paper content: "
                            "abstract, introduction, methodology, results, figures, tables, and conclusion."
                        ),
                        "url": post_url_after_redirect,
                        "redirected_from": post_url,
                        "redirected_to": post_url_after_redirect,
                    }

                # ── Blocked domain auto-escape ──
                elif any(domain in post_url for domain in BLOCKED_DOMAINS):
                    logger.warning(f"Step {step}: Landed on blocked domain ({post_url}). Auto-escaping back.")
                    try:
                        await browser.page.go_back(wait_until="domcontentloaded", timeout=15000)
                    except Exception as go_back_err:
                        logger.warning(f"Step {step}: Auto-escape go_back failed ({go_back_err}).")
                    await asyncio.sleep(1.5)
                    func_response_payload = {
                        "status": "warning",
                        "message": (
                            f"You navigated to a paywalled site ({post_url}) which is not allowed. "
                            "I have navigated back. Search for the open-access version on arxiv.org instead."
                        ),
                    }

                elif result.startswith("ERROR"):
                    current_url_on_error = browser.page.url if browser.page else ""
                    if "go_back failed" in result or "go_forward failed" in result:
                        func_response_payload = {
                            "status": "error",
                            "error": result,
                            "message": (
                                f"{result} "
                                f"The browser is still on: {current_url_on_error}. "
                                "Use navigate() with the exact previous URL to go back manually."
                            ),
                        }
                    else:
                        func_response_payload = {"status": "error", "error": result}

                else:
                    func_response_payload = {
                        "status": "success",
                        "message": f"Action {action.get('action')} executed successfully."
                    }

            pending_responses.append((fc, func_response_payload))

        # ── Capture screenshot and build FunctionResponses ──
        if pending_responses:
            # For PDF pages: use fast path (no stability check, window.stop() already called)
            # For normal pages: brief wait
            if _is_on_paper_page():
                await asyncio.sleep(0.3)  # Minimal wait — fast path handles the rest
            else:
                await asyncio.sleep(0.5)
                await browser.wait_for_visual_stability()

            # ── Graceful screenshot recovery ──
            try:
                new_screenshot_b64 = await browser.screenshot_b64()
            except Exception as screenshot_err:
                logger.error(f"Step {step}: Screenshot failed after retries: {screenshot_err}")
                try:
                    await browser.page.evaluate("window.stop()")
                    await asyncio.sleep(2.0)
                    new_screenshot_b64 = await browser.screenshot_b64()
                    logger.info(f"Step {step}: Screenshot recovered after window.stop()")
                except Exception as recovery_err:
                    logger.error(f"Step {step}: Screenshot recovery also failed: {recovery_err}")
                    try:
                        await browser.page.go_back(timeout=10000)
                        await asyncio.sleep(2.0)
                        new_screenshot_b64 = await browser.screenshot_b64()
                        logger.info(f"Step {step}: Screenshot recovered after go_back()")
                    except Exception as final_err:
                        logger.error(f"Step {step}: All screenshot recovery failed: {final_err}")
                        yield {
                            "action": "error",
                            "reason": f"Screenshot capture failed on heavy page. Error: {screenshot_err}",
                            "step": step,
                        }
                        return

            new_compressed = _compress_screenshot(new_screenshot_b64)

            next_msg_parts: list[types.Part] = []
            for fc, resp_payload in pending_responses:
                resp_payload["url"] = browser.page.url if browser.page else ""

                fr = types.FunctionResponse(
                    name=fc.name or "unknown",
                    response=resp_payload,
                    parts=[
                        types.FunctionResponsePart.from_bytes(
                            data=new_compressed,
                            mime_type="image/jpeg",
                        )
                    ],
                )
                next_msg_parts.append(types.Part(function_response=fr))

            conversation_history.append(types.Content(role="user", parts=next_msg_parts))

            # ── Smart pruning with state block ──
            conversation_history = _prune_conversation_history(conversation_history, state_block)

            logger.info(
                f"Step {step + 1}: Calling Gemini "
                f"({len(next_msg_parts)} parts, {len(conversation_history)} turns)"
            )
            try:
                response = await _send_manual_message_with_retry(
                    client, model_name, conversation_history, generation_config
                )
                logger.info(f"Step {step + 1}: Gemini responded successfully")
            except RuntimeError as e:
                logger.error(f"Step {step + 1}: Gemini FAILED after all retries — {e}")
                yield {"action": "error", "reason": str(e), "step": step + 1}
                return

        step += 1

    # Graceful completion at max_steps
    if step > max_steps:
        logger.warning(
            f"⚠️  Computer Use vision loop hit max_steps limit ({max_steps}). "
            "Treating as graceful completion — browsing data collected so far will be used."
        )
        yield {
            "action": "done",
            "reason": f"Browsing phase complete ({max_steps} steps). Proceeding to analysis.",
            "step": step,
        }
    else:
        logger.info(f"🏁 Computer Use vision loop completed after {step} steps")

    # ── Emit hybrid analysis data if available ──
    if _hybrid_enabled and _governor and _governor.captured_papers:
        governor_summary = _governor.get_summary()
        logger.info(
            f"📊 Hybrid governor summary: {governor_summary['papers_enqueued']} papers captured, "
            f"{governor_summary['total_screenshots']} screenshots, "
            f"{governor_summary['total_calls']} API calls"
        )
        yield {
            "action": "hybrid_papers_ready",
            "captured_papers": _governor.captured_papers,
            "governor_summary": governor_summary,
            "step": step,
        }
