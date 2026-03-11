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
import os

# ============================================================================
# HOW TO SWITCH BACK TO AI STUDIO
# ============================================================================
# 1. Update `_get_client()` below to return `genai.Client(api_key=...)`.
# 2. Ensure your .env file has GOOGLE_API_KEY or GEMINI_API_KEY set.
# ============================================================================

# --- Shared Imports ---
from google import genai
from google.genai import types

from core.browser import BrowserController
from config import settings
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

# ─── Valid actions and their required fields ───
VALID_ACTIONS = {"click", "type", "scroll", "navigate", "press", "wait", "done", "ask_user"}
ACTION_REQUIRES_XY = {"click"}
ACTION_REQUIRES_TEXT = {"type", "navigate", "press"}
ACTION_REQUIRES_DELTA = {"scroll"}

SYSTEM_PROMPT = """
You are a browser automation agent for academic research.
You receive a screenshot of a browser and a task.
You output ONE action at a time as valid JSON.

Action schema:
{"action": "click|type|scroll|navigate|press|wait|done", "x": int, "y": int, "text": str, "delta": int, "reason": str}

Rules:
- NEVER use DOM selectors or JavaScript
- Base all decisions on what you visually see in the screenshot
- A red/blue transparent coordinate grid (100x100 pixels per block) is injected over the image to help you precisely estimate 'x' and 'y' for clicks. Use the X: and Y: labels to calculate the exact center of the element you want to click.
- For type: provide the text to type (use after clicking input). IMPORTANT: If you are typing into a search bar, your very next action MUST be to use the "press" action for "Enter". Do NOT try to click the search button with your mouse.
- For press: provide the key name in the 'text' field (e.g., "Enter", "Escape")
- For navigate: provide full URL in text field
- For scroll: provide 'delta' (positive to scroll down, negative to scroll up)
- For ask_user: use when you encounter a CAPTCHA or are completely stuck. Provide the message to the user in the 'reason' field.
- For done: set when task is fully complete
- Keep reason short and descriptive
- If you see a cookie banner, dismiss it first
- If page is loading, use wait action

Respond ONLY with valid JSON. No markdown, no explanation.
"""

# ─── Retry Configuration ───
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds
GEMINI_TIMEOUT = 30.0   # seconds per call


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


def validate_action(action: dict) -> str | None:
    """
    Validate an action dict against the expected schema.
    Returns None if valid, or an error string if invalid.
    """
    act = action.get("action")
    if not act:
        return "Missing 'action' field"
    if act not in VALID_ACTIONS:
        return f"Unknown action '{act}' — must be one of {VALID_ACTIONS}"
    if act in ACTION_REQUIRES_XY:
        if "x" not in action or "y" not in action:
            return f"Action '{act}' requires 'x' and 'y' coordinates"
        if not isinstance(action["x"], (int, float)) or not isinstance(action["y"], (int, float)):
            return f"Action '{act}' requires numeric 'x' and 'y'"
    if act in ACTION_REQUIRES_TEXT:
        if not action.get("text"):
            return f"Action '{act}' requires a non-empty 'text' field"
    if act in ACTION_REQUIRES_DELTA:
        if "delta" not in action:
            return f"Action '{act}' requires 'delta'"
        if not isinstance(action["delta"], (int, float)):
            return f"Action '{act}' requires numeric 'delta'"
    return None


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


async def run_vision_loop(
    browser: BrowserController,
    task: str,
    max_steps: int = 30,
    pause_event: asyncio.Event = None,
) -> AsyncGenerator[dict, None]:
    """
    Core ReAct loop: screenshot → Gemini → action → repeat.
    Yields each action for real-time streaming to frontend.
    """
    import base64
    client = _get_client()
    model_name = settings.GOOGLE_VISION_MODEL
    logger.info(f"Using vision model: {model_name}")
    
    history = []          # text summaries of past steps
    recent_frames = []    # last N screenshots for visual grounding
    step = 0
    consecutive_clicks = 0
    last_click_coords = None

    generation_config = types.GenerateContentConfig(
        temperature=0.4,
        max_output_tokens=2048,
        response_mime_type="application/json",
        system_instruction=SYSTEM_PROMPT,
    )

    while step < max_steps:
        # Pre-capture checks
        await browser.wait_for_visual_stability()
        await browser.inject_grid(cell_size=100)
        
        # 1. Capture current screen
        screenshot_b64 = await browser.screenshot_b64()
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
                        types.Part.from_text(text=f"Task: {task}\n\nStep {prev_frame['step']}. What is the next single action?")
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
                    types.Part.from_text(text=f"Task: {task}\n\nStep {step + 1}. What is the next single action?")
                ]
            )
        )

        # 4. Call Gemini with retry + timeout
        try:
            raw = await _call_gemini_with_retry(client, model_name, contents, generation_config)
        except RuntimeError as e:
            yield {"action": "error", "reason": str(e)}
            break

        # 5. Parse action JSON
        try:
            action = json.loads(raw)
        except json.JSONDecodeError:
            yield {"action": "error", "reason": f"Gemini returned invalid JSON: {raw[:200]}"}
            break

        # 6. Validate action schema
        validation_error = validate_action(action)
        if validation_error:
            logger.warning(f"Step {step + 1}: invalid action — {validation_error}")
            yield {"action": "error", "reason": f"Invalid action from Gemini: {validation_error}", "raw": action}
            break

        # --- Anti-stuck mechanism ---
        if action.get("action") == "click":
            coords = (action.get("x"), action.get("y"))
            
            # Check if clicks are within 15 pixels of each other
            is_same_area = False
            if last_click_coords:
                import math
                dist = math.hypot(coords[0] - last_click_coords[0], coords[1] - last_click_coords[1])
                is_same_area = dist < 15
                
            if is_same_area:
                consecutive_clicks += 1
            else:
                consecutive_clicks = 1
                last_click_coords = coords
            
            if consecutive_clicks >= 3:
                logger.warning(f"Anti-stuck triggered near {coords}")
                action_summary = json.dumps({
                    "action": "error", 
                    "reason": f"You already clicked near {coords} multiple times and nothing changed. DON'T click there again. Try a different strategy or navigate somewhere else."
                })
                yield {"action": "wait", "step": step + 1, "reason": "Anti-stuck: Injecting dynamic error to model"}
                
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
                    
                step += 1
                continue

        # 7. Yield action to frontend (real-time stream)
        action["step"] = step + 1
        yield action

        # 8. Check if done
        if action.get("action") == "done":
            break

        # 8.5 Check if ask_user pause needed
        if action.get("action") == "ask_user":
            if pause_event:
                logger.info(f"Vision loop paused at step {step + 1}. Waiting for user input.")
                await pause_event.wait()
                pause_event.clear()
                logger.info("Vision loop resumed by user.")
                
                action_summary = json.dumps({"action": "ask_user", "reason": action.get("reason", "")})
                history.append(f"Step {step + 1}: {action_summary} -> (User resolved)")
                if len(history) > 10:
                    history = history[-10:]
                
                recent_frames.append({
                    "step": step + 1,
                    "frame": screenshot_b64,
                    "action_summary": action_summary + " (Resolved by user)",
                })
                if len(recent_frames) > 3:
                    recent_frames = recent_frames[-3:]
                    
                step += 1
                continue
            else:
                yield {"action": "error", "reason": "ask_user action requested, but no pause_event configured."}
                break

        # 9. Execute action
        result = await browser.execute_action(action)
        if result.startswith("ERROR"):
            yield {"action": "error", "reason": f"Browser action failed: {result}"}
            break
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

        step += 1

    logger.info(f"Vision loop completed after {step} steps")