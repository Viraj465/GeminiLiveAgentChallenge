"""
copilot_mode.py — Copilot guidance pipeline (Day 3).

Receives the user's screen frames (from getDisplayMedia) and a command,
sends them to Gemini Vision, and returns guidance text telling the user
what to do next.

Flow:
  Frame (base64) + User Command → Gemini 2.0 Flash Vision → Guidance text
"""

import json
import asyncio
import logging
import base64
from google import genai
from google.genai import types
from config import settings
from typing import Optional

logger = logging.getLogger(__name__)

from prompts import COPILOT_SYSTEM_PROMPT

# ─── Per-session copilot state ───
_session_state: dict[str, dict] = {}

MAX_FRAME_HISTORY = 5


def _get_session(session_id: str) -> dict:
    """Get or create per-session copilot state."""
    if session_id not in _session_state:
        _session_state[session_id] = {
            "command": "",
            "frame_history": [],   # list of {frame_b64, guidance}
            "client": None,
            "model_name": None,
        }
    return _session_state[session_id]


def _ensure_model(state: dict):
    """Lazy-init the Gemini model for this session."""
    if state["client"] is None:
        api_key = settings.GEMINI_API_KEY
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY is not set")
        state["client"] = genai.Client(api_key=api_key)
        state["model_name"] = settings.GOOGLE_VISION_MODEL


def set_user_command(session_id: str, command: str):
    """
    ADK Tool: Update the active user command for a copilot session.
    Required session_id and command string.
    Returns dict with status key always present.
    """
    try:
        state = _get_session(session_id)
        state["command"] = command
        logger.info(f"Copilot [{session_id}]: command set to '{command}'")
        return {"status": "success", "command": command}
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def analyze_frame(
    session_id: str,
    frame_b64: str,
    user_command: Optional[str] = None,
) -> dict:
    """
    ADK Tool: Analyze a user's screen frame and return guidance.
    Required session_id and frame_b64 (base64 PNG).
    Returns dict with status key always present.
    """
    try:
        state = _get_session(session_id)
        _ensure_model(state)

        # Update command if provided
        if user_command:
            state["command"] = user_command

        command = state["command"]
        if not command:
            return {
                "status": "error",
                "message": "No user command set — send a command first",
            }

        # Build context from recent frame history
        contents = [COPILOT_SYSTEM_PROMPT]

        # Add last 2 frames from history for continuity
        for prev in state["frame_history"][-2:]:
            contents.append(types.Part.from_bytes(data=base64.b64decode(prev["frame"]), mime_type="image/jpeg"))
            contents.append(f"(Previous guidance: {prev['guidance']})")

        # Add current frame + command
        contents.append(f"User's research task: {command}")
        contents.append(types.Part.from_bytes(data=base64.b64decode(frame_b64), mime_type="image/jpeg"))
        contents.append("What should the user do next?")

        # Call Gemini Vision
        response = await asyncio.wait_for(
            state["client"].aio.models.generate_content(
                model=state["model_name"],
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=300,
                    response_mime_type="application/json",
                ),
            ),
            timeout=15.0,
        )

        raw = response.text.strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {"guidance": raw, "status": "guiding"}

        guidance_text = result.get("guidance", "")
        guidance_status = result.get("status", "guiding")

        # Update frame history
        state["frame_history"].append({
            "frame": frame_b64,
            "guidance": guidance_text,
        })
        if len(state["frame_history"]) > MAX_FRAME_HISTORY:
            state["frame_history"] = state["frame_history"][-MAX_FRAME_HISTORY:]

        logger.info(f"Copilot [{session_id}]: {guidance_status} — {guidance_text[:80]}")

        return {
            "status": "success",
            "guidance": guidance_text,
            "copilot_status": guidance_status,
        }

    except asyncio.TimeoutError:
        logger.warning(f"Copilot [{session_id}]: Gemini call timed out")
        return {"status": "error", "message": "Vision analysis timed out"}
    except Exception as e:
        logger.error(f"Copilot [{session_id}]: error — {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


def cleanup_session(session_id: str):
    """Remove copilot state for a disconnected session."""
    _session_state.pop(session_id, None)
    logger.info(f"Copilot [{session_id}]: session cleaned up")
