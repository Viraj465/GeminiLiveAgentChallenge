"""
autopilot_mode.py — Autopilot orchestrator (Day 4).

Manages the lifecycle of an autonomous browsing session:
  1. Launch headless Playwright browser
  2. Run the vision loop (screenshot → Gemini → action → repeat)
  3. Stream every action AND screenshot over WebSocket
  4. Handle cleanup on completion or error

This module is the bridge between the WebSocket layer and the core vision loop.
"""

import json
import asyncio
import logging
import time
from fastapi import WebSocket
from core.browser import BrowserController
from core.vision_loop import run_vision_loop
from core.graph_builder import generate_citation_graph
from core.synthesis import generate_literature_review
from core.db import save_session_data
from websocket.models import WSMessage, WSMessageType

logger = logging.getLogger(__name__)

# Global registry to expose active browsers to the websocket handler for manual overrides
active_browsers: dict[str, BrowserController] = {}


async def run_autopilot(
    session_id: str,
    task: str,
    websocket: WebSocket,
    max_steps: int = 30,
    start_url: str = "",
    pause_event: asyncio.Event = None,
) -> dict:
    """
    ADK Tool: Run an autonomous browsing task end-to-end.
    Required task description string.
    Returns dict with status key always present.
    """
    browser = BrowserController()
    
    # Register this browser so the websocket handler can access it for manual clicks
    active_browsers[session_id] = browser

    try:
        # 1. Launch browser
        await _send_log(websocket, f"Launching browser for: {task}")
        await browser.start()
        await _send_log(websocket, "Browser ready")

        # Send initial blank screenshot
        initial_frame = await browser.screenshot_b64()
        await _send_frame(websocket, initial_frame, "about:blank")

        # 2. Navigate to start URL if provided
        if not start_url:
            start_url = "https://google.com"
            task = f"Start by navigating to https://google.com. {task}"

        if start_url:
            await _send_log(websocket, f"Navigating to {start_url}")
            await browser.page.goto(start_url, wait_until="networkidle", timeout=15000)
            frame = await browser.screenshot_b64()
            await _send_frame(websocket, frame, start_url)

        # 3. Run vision loop — stream each action + screenshot
        step_count = 0
        async for action in run_vision_loop(browser, task, max_steps=max_steps, pause_event=pause_event):
            step_count += 1

            if action.get("action") == "error":
                await _send_action(websocket, action)
                return {"status": "error", "message": action.get("reason"), "steps": step_count}

            # Stream the action
            await _send_action(websocket, action)

            if action.get("action") == "done":
                break

            # Stream the screenshot AFTER the action was executed
            frame = await browser.screenshot_b64()
            current_url = browser.page.url if browser.page else ""
            await _send_frame(websocket, frame, current_url)

        # 4. Capture + send final screenshot
        final_frame = await browser.screenshot_b64()
        final_url = browser.page.url if browser.page else ""
        await _send_frame(websocket, final_frame, final_url)
        
        # 5. [PHASE 4] Emit Mock Citation Graph
        await _send_log(websocket, "Generating citation network graph...")
        # Mocking 5 papers that were "found" during the research task
        mock_papers = [
            {"title": "Foundational Models in AI", "authors": "Smith et al."},
            {"title": f"Applications of {task[:20]}", "authors": "Johnson et al."},
            {"title": "Theoretical limits of Attention", "authors": "Williams & Chen"},
            {"title": "Scaling Laws Revisited", "authors": "Davis et al."},
            {"title": "Future Directions", "authors": "Miller"}
        ]
        
        graph_data = generate_citation_graph(mock_papers, topic=task[:20] + "...")
        
        # Save to Firestore
        asyncio.create_task(save_session_data(session_id, "graph_data", graph_data))
        
        # Emit the graph_update event to trigger CitationGraph.tsx on frontend
        await websocket.send_json({
            "type": "graph_update",
            "payload": {
                "graph_data": graph_data,
                "timestamp": time.time(),
            },
        })
        
        await _send_log(websocket, "Citation network generated and sent to UI.")

        # 6. [PHASE 4] Emit Mock Report (Simulating passing extracted texts)
        await _send_log(websocket, "Synthesizing literature review...")
        
        # In a fully integrated pipeline, you would pass real results from process_scraped_papers
        # For demonstration hitting the UI:
        mock_texts = {
            "Foundational Models in AI": "This paper discusses the emergence of large language models and their zero-shot capabilities...",
            f"Applications of {task[:20]}": f"We explore how {task[:20]} is applied in various industries including healthcare and finance...",
            "Theoretical limits of Attention": "An analysis of the quadratic complexity of standard attention mechanisms..."
        }
        
        report_markdown = await generate_literature_review(topic=task[:30], extracted_texts=mock_texts)
        
        # Save to Firestore
        asyncio.create_task(save_session_data(session_id, "report_markdown", report_markdown))
        
        # Emit the report_update event to trigger ReportViewer.tsx on frontend
        await websocket.send_json({
            "type": "report_update",
            "payload": {
                "report": report_markdown,
                "timestamp": time.time(),
            },
        })
        
        await _send_log(websocket, "Literature review complete and sent to UI.")

        await _send_complete(websocket, f"Task finished in {step_count} steps")

        return {
            "status": "success",
            "steps": step_count,
            "final_screenshot": final_frame,
        }

    except asyncio.CancelledError:
        logger.info("Autopilot task cancelled")
        await _send_log(websocket, "Task cancelled")
        return {"status": "cancelled", "message": "Task was cancelled"}

    except Exception as e:
        error_msg = f"Autopilot error: {e}"
        logger.error(error_msg, exc_info=True)
        try:
            await _send_error(websocket, error_msg)
        except Exception:
            pass
        return {"status": "error", "message": str(e)}

    finally:
        # Unregister and clean up the browser
        active_browsers.pop(session_id, None)
        await browser.close()


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
        await websocket.send_json({
            "type": WSMessageType.AGENT_ACTION,
            "payload": {
                "action": action.get("reason", action.get("action", "")),
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