"""
call_governor.py — Centralized rate-limit, cost, and call budget governor.

Enforces:
  - Per-run Gemini call cap (total API calls across the entire session)
  - Per-paper capture cap (max screenshots per paper in PAPER_CAPTURE mode)
  - Centralized 429 backoff + cooldown (exponential backoff with jitter)
  - No-progress / stagnation breaker (detects when the agent is stuck)

All limits are configurable via environment variables with sensible defaults.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ─── Defaults (overridable via env vars) ───
DEFAULT_MAX_GEMINI_CALLS_PER_RUN = int(os.getenv("MAX_GEMINI_CALLS_PER_RUN", "200"))
DEFAULT_MAX_SCREENSHOTS_PER_PAPER = int(os.getenv("MAX_SCREENSHOTS_PER_PAPER", "6"))
DEFAULT_MAX_PAPERS_PER_RUN = int(os.getenv("MAX_PAPERS_PER_RUN", "15"))
DEFAULT_RATE_LIMIT_COOLDOWN_BASE = float(os.getenv("RATE_LIMIT_COOLDOWN_BASE", "30.0"))
DEFAULT_RATE_LIMIT_COOLDOWN_MAX = float(os.getenv("RATE_LIMIT_COOLDOWN_MAX", "180.0"))
DEFAULT_STAGNATION_THRESHOLD = int(os.getenv("STAGNATION_THRESHOLD", "5"))


@dataclass
class PaperBudget:
    """Tracks budget usage for a single paper during PAPER_CAPTURE mode."""
    url: str = ""
    title: str = ""
    screenshots_taken: int = 0
    screenshot_labels: list = field(default_factory=list)
    screenshot_data: list = field(default_factory=list)  # list of (label, jpeg_bytes)
    scroll_positions: list = field(default_factory=list)
    capture_complete: bool = False
    handoff_ready: bool = False
    start_time: float = field(default_factory=time.time)

    @property
    def budget_exhausted(self) -> bool:
        return self.screenshots_taken >= DEFAULT_MAX_SCREENSHOTS_PER_PAPER

    def record_screenshot(self, label: str, jpeg_bytes: bytes, scroll_y: int = 0):
        """Record a captured screenshot with its label and position."""
        if self.budget_exhausted:
            logger.warning(f"Screenshot budget exhausted for '{self.title}' — ignoring capture.")
            return False
        self.screenshots_taken += 1
        self.screenshot_labels.append(label)
        self.screenshot_data.append((label, jpeg_bytes))
        self.scroll_positions.append(scroll_y)
        logger.info(
            f"📸 Paper '{self.title[:40]}' screenshot {self.screenshots_taken}/"
            f"{DEFAULT_MAX_SCREENSHOTS_PER_PAPER}: {label} (scrollY={scroll_y})"
        )
        if self.budget_exhausted:
            self.capture_complete = True
            self.handoff_ready = True
            logger.info(f"✅ Paper '{self.title[:40]}' capture quota met — ready for handoff.")
        return True


class CallGovernor:
    """
    Centralized governor for API calls, rate limits, and session budgets.

    Usage:
        governor = CallGovernor()
        if governor.can_make_call():
            governor.record_call()
            # ... make API call ...
        else:
            # budget exhausted, stop
    """

    def __init__(
        self,
        max_calls: int = DEFAULT_MAX_GEMINI_CALLS_PER_RUN,
        max_papers: int = DEFAULT_MAX_PAPERS_PER_RUN,
        max_screenshots_per_paper: int = DEFAULT_MAX_SCREENSHOTS_PER_PAPER,
    ):
        self.max_calls = max_calls
        self.max_papers = max_papers
        self.max_screenshots_per_paper = max_screenshots_per_paper

        # Counters
        self.total_calls: int = 0
        self.total_papers_captured: int = 0
        self.total_screenshots: int = 0

        # Rate limit state
        self._rate_limit_hits: int = 0
        self._last_rate_limit_time: float = 0.0
        self._cooldown_until: float = 0.0

        # Stagnation detection
        self._last_progress_step: int = 0
        self._stagnation_counter: int = 0

        # Paper budgets
        self._paper_budgets: dict[str, PaperBudget] = {}

        # Captured papers queue (for handoff to extraction)
        self.captured_papers: list[dict] = []

        self._start_time = time.time()

    # ─── Call budget ───

    def can_make_call(self) -> bool:
        """Check if we're within the per-run call budget."""
        if self.total_calls >= self.max_calls:
            logger.warning(
                f"🛑 Call budget exhausted: {self.total_calls}/{self.max_calls} calls used."
            )
            return False
        return True

    def record_call(self):
        """Record a Gemini API call."""
        self.total_calls += 1
        if self.total_calls % 20 == 0:
            elapsed = time.time() - self._start_time
            logger.info(
                f"📊 Governor: {self.total_calls}/{self.max_calls} calls, "
                f"{self.total_papers_captured} papers captured, "
                f"{self.total_screenshots} screenshots, "
                f"{elapsed:.0f}s elapsed"
            )

    # ─── Rate limit handling ───

    def is_in_cooldown(self) -> bool:
        """Check if we're currently in a rate-limit cooldown period."""
        return time.time() < self._cooldown_until

    async def handle_rate_limit(self) -> float:
        """
        Handle a 429 rate limit error with exponential backoff + jitter.
        Returns the number of seconds waited.
        """
        self._rate_limit_hits += 1
        self._last_rate_limit_time = time.time()

        # Exponential backoff: base * 2^(hits-1), capped at max
        import random
        delay = min(
            DEFAULT_RATE_LIMIT_COOLDOWN_BASE * (2 ** (self._rate_limit_hits - 1)),
            DEFAULT_RATE_LIMIT_COOLDOWN_MAX,
        )
        # Add jitter (±20%)
        jitter = delay * random.uniform(-0.2, 0.2)
        actual_delay = delay + jitter

        self._cooldown_until = time.time() + actual_delay

        logger.warning(
            f"⏳ Rate limit hit #{self._rate_limit_hits}. "
            f"Cooling down for {actual_delay:.1f}s "
            f"(base={delay:.1f}s, jitter={jitter:.1f}s)"
        )
        await asyncio.sleep(actual_delay)
        return actual_delay

    def reset_rate_limit_counter(self):
        """Reset rate limit counter after a successful call."""
        if self._rate_limit_hits > 0:
            self._rate_limit_hits = max(0, self._rate_limit_hits - 1)

    # ─── Paper capture budget ───

    def get_paper_budget(self, url: str, title: str = "") -> PaperBudget:
        """Get or create a budget tracker for a paper."""
        if url not in self._paper_budgets:
            self._paper_budgets[url] = PaperBudget(url=url, title=title)
            self.total_papers_captured += 1
        return self._paper_budgets[url]

    def is_paper_capture_complete(self, url: str) -> bool:
        """Check if a paper's screenshot capture quota is met."""
        budget = self._paper_budgets.get(url)
        return budget.capture_complete if budget else False

    def can_capture_more_papers(self) -> bool:
        """Check if we can start capturing a new paper."""
        return self.total_papers_captured < self.max_papers

    def enqueue_paper_for_analysis(self, paper_data: dict):
        """
        Enqueue a captured paper for in-memory analysis.
        Called when PAPER_CAPTURE → HANDOFF_ANALYSIS transition fires.
        """
        url = paper_data.get("url", "")
        budget = self._paper_budgets.get(url)

        enriched = {
            **paper_data,
            "vision_screenshots": [],
            "capture_metadata": {},
        }

        if budget:
            enriched["vision_screenshots"] = [
                {"label": label, "jpeg_bytes": data}
                for label, data in budget.screenshot_data
            ]
            enriched["capture_metadata"] = {
                "screenshots_taken": budget.screenshots_taken,
                "screenshot_labels": budget.screenshot_labels,
                "scroll_positions": budget.scroll_positions,
                "capture_duration": time.time() - budget.start_time,
            }

        self.captured_papers.append(enriched)
        logger.info(
            f"📋 Paper enqueued for analysis: '{paper_data.get('title', url)[:50]}' "
            f"({budget.screenshots_taken if budget else 0} screenshots)"
        )

    # ─── Stagnation detection ───

    def record_progress(self, step: int):
        """Record that meaningful progress was made at this step."""
        self._last_progress_step = step
        self._stagnation_counter = 0

    def check_stagnation(self, current_step: int) -> bool:
        """
        Check if the agent is stagnating (no progress for N steps).
        Returns True if stagnation is detected.
        """
        steps_since_progress = current_step - self._last_progress_step
        if steps_since_progress >= DEFAULT_STAGNATION_THRESHOLD:
            self._stagnation_counter += 1
            logger.warning(
                f"⚠️ Stagnation detected: no progress for {steps_since_progress} steps "
                f"(counter: {self._stagnation_counter})"
            )
            return True
        return False

    # ─── Summary ───

    def get_summary(self) -> dict:
        """Return a summary of governor state for logging/debugging."""
        elapsed = time.time() - self._start_time
        return {
            "total_calls": self.total_calls,
            "max_calls": self.max_calls,
            "total_papers_captured": self.total_papers_captured,
            "total_screenshots": self.total_screenshots,
            "rate_limit_hits": self._rate_limit_hits,
            "papers_enqueued": len(self.captured_papers),
            "elapsed_seconds": round(elapsed, 1),
            "budget_remaining": self.max_calls - self.total_calls,
        }
