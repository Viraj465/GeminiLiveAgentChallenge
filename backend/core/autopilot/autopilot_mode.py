"""
autopilot_mode.py — Autopilot orchestrator (Day 4).

Manages the lifecycle of an autonomous browsing session:
  1. Launch headless Playwright browser
  2. Run the vision loop (screenshot → Gemini → action → repeat)
  3. Stream every action AND screenshot over WebSocket
  4. Handle cleanup on completion or error

This module is the bridge between the WebSocket layer and the core vision loop.
"""

import asyncio
import logging
import time
from fastapi import WebSocket
from core.browser import BrowserController
from core.stealth_browser import StealthBrowserController
from config import settings
from core.vision_loop import run_vision_loop
from core.vision_loop_computer_use import run_vision_loop_computer_use
from core.paper_discovery import extract_papers_from_page_content
from core.adk_pipeline import run_adk_pipeline
from ws_handlers.models import WSMessage, WSMessageType

logger = logging.getLogger(__name__)

# Global registry to expose active browsers to the websocket handler for manual overrides
active_browsers: dict[str, BrowserController] = {}


async def run_autopilot(
    session_id: str,
    task: str,
    websocket: WebSocket,
    max_steps: int = 50, # changed from 30
    start_url: str = "",
    pause_event: asyncio.Event = None,
    keep_browser_open_on_success: bool = True,
    prior_tasks: list[str] = None,
) -> dict:
    """
    ADK Tool: Run an autonomous browsing task end-to-end.
    Required task description string.
    prior_tasks: ordered list of previous user queries in this session (for context continuity).
    Returns dict with status key always present.
    """
    # If a previous successful run left a browser open, close it before starting a new task.
    previous_browser = active_browsers.pop(session_id, None)
    if previous_browser:
        try:
            await previous_browser.close()
        except Exception as e:
            logger.warning(f"Failed closing previous browser for session {session_id}: {e}")

    # Use StealthBrowserController if enabled in settings (always preferred in production)
    use_stealth = settings.USE_STEALTH_BROWSER
    browser = None
    completed_successfully = False
    partial_steps_done = 0

    # Register this browser so the websocket handler can access it for manual clicks
    # (will be updated once the browser is successfully created)
    active_browsers[session_id] = None  # placeholder

    try:
        # 1. Launch browser — stealth mode with automatic fallback to standard mode
        await _send_log(websocket, f"Launching browser for: {task}")

        if use_stealth:
            logger.info("Autopilot: Using StealthBrowserController")
            try:
                browser = StealthBrowserController()
                await browser.start(headless=settings.BROWSER_HEADLESS)
                logger.info("✅ Stealth browser started successfully")
            except Exception as stealth_err:
                err_lower = str(stealth_err).lower()
                is_binary_error = any(kw in err_lower for kw in (
                    "chrome browser binary", "browser binary", "executable",
                    "no such file", "not found", "cannot find", "chromedriver",
                ))
                logger.error(f"Stealth browser failed: {stealth_err}", exc_info=True)
                await _send_log(
                    websocket,
                    f"⚠️ Stealth browser failed ({stealth_err}). "
                    f"{'Chrome binary not found — ' if is_binary_error else ''}"
                    f"Falling back to standard Playwright browser..."
                )
                # Clean up the failed stealth browser before creating a new one
                try:
                    await browser.close()
                except Exception:
                    pass
                browser = BrowserController()
                await browser.start(headless=settings.BROWSER_HEADLESS)
                logger.info("✅ Fallback standard browser started successfully")
        else:
            logger.info("Autopilot: Using standard BrowserController")
            browser = BrowserController()
            await browser.start(headless=settings.BROWSER_HEADLESS)

        # Register the successfully started browser
        active_browsers[session_id] = browser
        await _send_log(websocket, "Browser ready")

        # Send initial blank screenshot
        initial_frame = await browser.screenshot_b64()
        await _send_frame(websocket, initial_frame, "about:blank")

        # 2. Navigate to start URL if provided
        if not start_url:
            start_url = "https://google.com"

        if start_url:
            await _send_log(websocket, f"Navigating to {start_url}")
            await browser.page.goto(start_url, wait_until="networkidle", timeout=15000)
            frame = await browser.screenshot_b64()
            await _send_frame(websocket, frame, start_url)

        # 3. Run vision loop — stream each action + screenshot
        # Choose Computer Use loop if enabled, else standard
        if settings.USE_COMPUTER_USE:
            vision_loop = run_vision_loop_computer_use
            logger.info("Autopilot: Using Computer Use (native tool calls) vision loop")
        else:
            vision_loop = run_vision_loop
            logger.info("Autopilot: Using standard vision loop")
        
        step_count = 0
        visited_urls: list[str] = []       # All URLs visited during browsing
        paper_found_urls: list[str] = []   # URLs where agent landed on a paper/PDF page
        hybrid_captured_papers: list[dict] = []  # Papers captured via hybrid mode

        logger.info("🔍 Autopilot vision loop starting — collecting URLs and paper pages...")
        await _send_log(websocket, "🔍 Vision loop started. Searching for papers...")

        async for action in vision_loop(browser, task, max_steps=max_steps, pause_event=pause_event, prior_tasks=prior_tasks or []):
            step_count += 1
            partial_steps_done = step_count

            # ── Collect visited URLs for multi-page paper discovery ──
            if action.get("action") == "url_visited":
                url = action.get("url", "")
                if url and url not in visited_urls:
                    visited_urls.append(url)
                    logger.info(f"📌 URL recorded [{len(visited_urls)}]: {url}")
                continue  # Don't stream url_visited events to the frontend

            # ── Hybrid mode events ──
            if action.get("action") == "hybrid_capture_start":
                url = action.get("url", "")
                await _send_log(
                    websocket,
                    f"📸 Hybrid capture: capturing screenshots for paper at {url}"
                )
                continue

            if action.get("action") == "hybrid_capture_complete":
                url = action.get("url", "")
                title = action.get("title", "")
                ss_count = action.get("screenshots_taken", 0)
                await _send_log(
                    websocket,
                    f"✅ Paper captured: '{title[:50]}' ({ss_count} screenshots) — "
                    f"will analyze via PDF download after browsing."
                )
                if url and url not in paper_found_urls:
                    paper_found_urls.append(url)
                continue

            if action.get("action") == "hybrid_capture_skipped":
                await _send_log(
                    websocket,
                    f"⏭️ Paper capture skipped: {action.get('reason', 'budget exhausted')}"
                )
                continue

            if action.get("action") == "hybrid_papers_ready":
                hybrid_captured_papers = action.get("captured_papers", [])
                governor_summary = action.get("governor_summary", {})
                await _send_log(
                    websocket,
                    f"📊 Hybrid browsing complete: {len(hybrid_captured_papers)} papers captured, "
                    f"{governor_summary.get('total_screenshots', 0)} screenshots taken"
                )
                continue

            # ── Paper found pivot: collect URL, log it, continue browsing ──
            # The vision loop emits paper_found when it lands on a PDF/paper page.
            # We collect the URL into paper_found_urls so the extraction pipeline
            # can prioritize these pages. We do NOT break — the agent may find
            # multiple papers across multiple pages.
            if action.get("action") == "paper_found":
                url = action.get("url", "")
                if url:
                    if url not in paper_found_urls:
                        paper_found_urls.append(url)
                        logger.info(
                            f"📄 Paper page detected [{len(paper_found_urls)}]: {url}"
                        )
                        await _send_log(
                            websocket,
                            f"📄 Paper located ({len(paper_found_urls)}): {url} — "
                            f"will extract via context cache after browsing."
                        )
                    # Also add to visited_urls if not already there
                    if url not in visited_urls:
                        visited_urls.append(url)
                continue  # Don't stream paper_found to frontend as an action

            # Stream the action to the frontend
            await _send_action(websocket, action)

            if action.get("action") == "error":
                # Still send a screenshot so the frontend shows current state
                try:
                    frame = await browser.screenshot_b64()
                    current_url = browser.page.url if browser.page else ""
                    await _send_frame(websocket, frame, current_url)
                except Exception:
                    pass
                return {"status": "error", "message": action.get("reason"), "steps": step_count}

            if action.get("action") == "done":
                break

            # Stream the screenshot AFTER every action — including wait,
            # ask_user, navigate_blocked, etc. This ensures the frontend
            # always shows the current browser state.
            try:
                frame = await browser.screenshot_b64()
                current_url = browser.page.url if browser.page else ""
                await _send_frame(websocket, frame, current_url)
            except Exception as e:
                logger.warning(f"Failed to capture/send post-action frame: {e}")

        # 4. Capture + send final screenshot
        final_frame = await browser.screenshot_b64()
        final_url = browser.page.url if browser.page else ""
        await _send_frame(websocket, final_frame, final_url)

        # ── Tell the frontend browsing is done BEFORE slow ADK pipeline.
        # Without this, the WebSocket goes silent for 10-30s and the frontend
        # interprets the silence as a session end, resetting the UI to blank.
        await websocket.send_json({
            "type": "session_status",
            "payload": {
                "status": "synthesis_started",
                "message": f"Browsing complete ({step_count} steps). Extracting papers & running research pipeline — please wait...",
                "steps_completed": step_count,
                "timestamp": time.time(),
            },
        })

        # 5. [PHASE 4] Extract papers discovered during browsing
        await _send_log(websocket, "Target paper(s) located. Preparing deep extraction pipeline...")
        await websocket.send_json({
            "type": "gcs_status",
            "payload": {
                "status": "uploading_to_gcs",
                "message": "Streaming document to Google Cloud Storage...",
                "timestamp": time.time(),
            },
        })

        # 5a. If hybrid mode captured papers, run in-memory analysis on them
        hybrid_analyzed_papers: list[dict] = []
        if hybrid_captured_papers and settings.USE_HYBRID_ANALYSIS:
            await _send_log(
                websocket,
                f"🔬 Running hybrid in-memory analysis on {len(hybrid_captured_papers)} captured papers..."
            )
            try:
                from core.paper_analyzer import analyze_papers_batch
                hybrid_analyzed_papers = await analyze_papers_batch(
                    hybrid_captured_papers, query=task, batch_size=3
                )
                success_count = sum(1 for p in hybrid_analyzed_papers if p.get("status") == "success")
                await _send_log(
                    websocket,
                    f"✅ Hybrid analysis complete: {success_count}/{len(hybrid_captured_papers)} papers "
                    f"analyzed via PDF download + screenshot evidence."
                )
            except Exception as e:
                logger.warning(f"Hybrid paper analysis failed (non-fatal): {e}")
                await _send_log(websocket, f"Hybrid analysis failed: {e}. Falling back to standard extraction.")

        # 5b. Standard paper discovery from browser screenshots
        await _send_log(websocket, "Extracting papers discovered during browsing...")
        discovered_papers = []
        try:
            # Pass the full URL history so paper_discovery can extract papers
            # from ALL pages visited, not just the final browser state.
            discovered_papers = await extract_papers_from_page_content(
                browser, task, visited_urls=visited_urls
            )
            await _send_log(
                websocket,
                f"Found {len(discovered_papers)} papers from browser session "
                f"({len(visited_urls)} URLs visited).",
            )
        except Exception as e:
            logger.warning(f"Paper discovery from browsing failed (non-fatal): {e}")
            await _send_log(websocket, f"Paper discovery from browser failed: {e}. Continuing with API search only.")

        # 5c. Merge hybrid-analyzed papers with vision-discovered papers
        # Hybrid papers take priority since they have richer data (screenshots + PDF text)
        all_discovered = list(hybrid_analyzed_papers)  # Start with hybrid (enriched)
        seen_urls = {p.get("url", "") for p in all_discovered if p.get("url")}
        seen_titles = {p.get("title", "").lower().strip()[:80] for p in all_discovered if p.get("title")}
        for p in discovered_papers:
            url = p.get("url", "")
            title_key = p.get("title", "").lower().strip()[:80]
            if url and url not in seen_urls and title_key not in seen_titles:
                all_discovered.append(p)
                seen_urls.add(url)
                if title_key:
                    seen_titles.add(title_key)

        if hybrid_analyzed_papers:
            await _send_log(
                websocket,
                f"📋 Total papers for pipeline: {len(all_discovered)} "
                f"({len(hybrid_analyzed_papers)} hybrid-analyzed, "
                f"{len(all_discovered) - len(hybrid_analyzed_papers)} from vision discovery)"
            )

        # 6. [PHASE 4] Run the full ADK pipeline: Search → Extract → Synthesize → Citation → Report
        await _send_log(websocket, "Starting ADK research pipeline...")
        pipeline_result = await run_adk_pipeline(
            task=task,
            discovered_papers=all_discovered,
            session_id=session_id,
            websocket=websocket,
        )

        # 7. Emit graph_update to trigger CitationGraph.tsx on frontend
        graph_data = pipeline_result.get("graph_data")
        if graph_data:
            await websocket.send_json({
                "type": "graph_update",
                "payload": {
                    "graph_data": graph_data,
                    "timestamp": time.time(),
                },
            })
            await _send_log(websocket, f"Citation network: {graph_data.get('node_count', 0)} nodes, {graph_data.get('edge_count', 0)} edges.")
        else:
            await _send_log(websocket, "Citation graph could not be generated.")

        # 8. Emit report_update to trigger ReportViewer.tsx on frontend
        report_markdown = pipeline_result.get("report_markdown")
        if report_markdown:
            await websocket.send_json({
                "type": "report_update",
                "payload": {
                    "report": report_markdown,
                    "timestamp": time.time(),
                },
            })
            await _send_log(websocket, f"Literature review generated ({len(report_markdown.split())} words).")
        else:
            await _send_log(websocket, "Literature review could not be generated.")

        # Summary
        papers_found = pipeline_result.get("papers_found", 0)
        papers_extracted = pipeline_result.get("papers_extracted", 0)
        await _send_log(
            websocket,
            f"Pipeline complete: {papers_found} papers found, {papers_extracted} extracted, "
            f"graph + report generated."
        )
        if keep_browser_open_on_success:
            await _send_log(websocket, "Task complete. Browser will remain open for inspection.")

        await _send_complete(websocket, f"Task finished in {step_count} steps — {papers_found} papers processed")
        completed_successfully = True

        return {
            "status": "success",
            "steps": step_count,
            "final_screenshot": final_frame,
        }

    except asyncio.CancelledError:
        # ── Bug 4 Fix: Don't silently discard progress.  Log how many steps ran,
        # notify the frontend so it can display partial results instead of going
        # blank, and include step count in the return dict.
        logger.info(f"Autopilot task cancelled after {partial_steps_done} steps")
        try:
            await _send_log(websocket, f"Session interrupted after {partial_steps_done} steps. Partial results preserved.")
            await websocket.send_json({
                "type": "session_status",
                "payload": {
                    "status": "cancelled",
                    "steps_completed": partial_steps_done,
                    "message": "Task was cancelled — browser kept open for inspection.",
                    "timestamp": time.time(),
                },
            })
        except Exception:
            pass
        return {"status": "cancelled", "message": "Task was cancelled", "steps": partial_steps_done}

    except Exception as e:
        error_msg = f"Autopilot error: {e}"
        logger.error(error_msg, exc_info=True)
        try:
            await _send_error(websocket, error_msg)
        except Exception:
            pass
        return {"status": "error", "message": str(e)}

    finally:
        # Guard: browser may be None if launch itself failed before assignment
        if browser is None:
            active_browsers.pop(session_id, None)
            return

        # Keep browser open if task succeeded OR if agent made meaningful progress
        keep_open = (completed_successfully or partial_steps_done > 3) and keep_browser_open_on_success
        if keep_open:
            logger.info(f"Keeping browser open (steps completed: {partial_steps_done})")
        else:
            if active_browsers.get(session_id) is browser:
                active_browsers.pop(session_id, None)
            try:
                await browser.close()
            except Exception as e:
                logger.warning(f"Error closing browser in finally block: {e}")


# ═══════════════════════════════════════════════
#  Structured Message Senders (WSMessage format)
# ═══════════════════════════════════════════════

async def _send_frame(websocket: WebSocket, frame_b64: str, url: str = ""):
    """Send a browser screenshot frame to the frontend."""
    try:
        await websocket.send_json({
            "type": WSMessageType.BROWSER_FRAME,
            "payload": {
                "frame": frame_b64,
                "url": url,
                "timestamp": time.time(),
            },
        })
    except Exception as e:
        logger.warning(f"Failed to send frame: {e}")


async def _send_action(websocket: WebSocket, action: dict):
    """Send an agent action to the frontend."""
    try:
        action_label = action.get("reason", action.get("action", ""))
        if action.get("action") == "click" and "x" in action and "y" in action:
            action_label = f"{action_label} (x={int(action['x'])}, y={int(action['y'])})"

        await websocket.send_json({
            "type": WSMessageType.AGENT_ACTION,
            "payload": {
                "action": action_label,
                "data": action,
                "timestamp": time.time(),
            },
        })
    except Exception as e:
        logger.warning(f"Failed to send action: {e}")


async def _send_log(websocket: WebSocket, message: str):
    """Send a log/status update to the frontend."""
    try:
        await websocket.send_json({
            "type": WSMessageType.LOG_UPDATE,
            "payload": {
                "log": message,
                "timestamp": time.time(),
            },
        })
    except Exception as e:
        logger.warning(f"Failed to send log: {e}")


async def _send_error(websocket: WebSocket, message: str):
    """Send an error to the frontend."""
    try:
        await websocket.send_json({
            "type": WSMessageType.ERROR,
            "payload": {
                "error": message,
                "timestamp": time.time(),
            },
        })
    except Exception as e:
        logger.warning(f"Failed to send error: {e}")


async def _send_complete(websocket: WebSocket, message: str):
    """Send a completion message to the frontend."""
    try:
        await websocket.send_json({
            "type": "complete",
            "payload": {
                "message": message,
                "timestamp": time.time(),
            },
        })
    except Exception as e:
        logger.warning(f"Failed to send complete: {e}")