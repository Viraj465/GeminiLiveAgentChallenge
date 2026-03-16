"""
action_corrector.py — Self-healing logic for imperfect LLM outputs.

Enhanced to support Google Computer Use actions:
- hover_at, go_back, go_forward, scroll_at
"""

import logging

logger = logging.getLogger(__name__)

def correct_action(action: dict) -> dict:
    """
    Attempt to fix common AI mistakes in action JSON.
    
    Enhanced to support:
    - New Google Computer Use actions
    - Better coordinate correction
    - Direction validation for scroll_at
    """
    act = action.get("action")
    
    if act == "scroll":
        if "delta" not in action:
            # Fallback for "direction" or just default to scroll down
            direction = action.get("direction", "down").lower()
            if "up" in direction:
                action["delta"] = -600
                logger.info("Correction: Added default up-scroll delta=-600")
            else:
                action["delta"] = 600
                logger.info("Correction: Added default down-scroll delta=600")
                
    elif act == "wait":
        if "seconds" not in action:
            action["seconds"] = 1
            logger.info("Correction: Added default wait seconds=1")
            
    elif act == "type":
        if "text" not in action:
            # Try to infer text or use empty string
            action["text"] = ""
            logger.info("Correction: Added empty text to type action")
    
    
    # Google Computer Use Actions Corrections
    
    
    elif act == "scroll_at":
        # Ensure direction is present
        if "direction" not in action:
            action["direction"] = "down"
            logger.info("Correction: Added default scroll_at direction='down'")
        
        # Ensure magnitude is present
        if "magnitude" not in action:
            action["magnitude"] = 300
            logger.info("Correction: Added default scroll_at magnitude=300")
        
        # Validate direction
        valid_directions = ["up", "down", "left", "right"]
        if action["direction"] not in valid_directions:
            action["direction"] = "down"
            logger.info(f"Correction: Invalid direction, defaulting to 'down'")
    
    elif act == "hover_at":
        # Ensure coordinates are present
        if "x" not in action or "y" not in action:
            logger.warning("Correction: hover_at missing coordinates, cannot fix")
    
    
    # Coordinate Bounds Correction
    
    
    # Clamp coordinates to valid bounds
    if "x" in action:
        x = action["x"]
        if x < 0:
            action["x"] = 0
            logger.info(f"Correction: Clamped x from {x} to 0")
        elif x > 1280:
            action["x"] = 1280
            logger.info(f"Correction: Clamped x from {x} to 1280")
    
    if "y" in action:
        y = action["y"]
        if y < 0:
            action["y"] = 0
            logger.info(f"Correction: Clamped y from {y} to 0")
        elif y > 800:
            action["y"] = 800
            logger.info(f"Correction: Clamped y from {y} to 800")

    return action
