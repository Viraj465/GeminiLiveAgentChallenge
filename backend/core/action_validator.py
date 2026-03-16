"""
action_validator.py — Logic to check if an action matches the defined schema.

Enhanced to support Google Computer Use actions:
- hover_at, go_back, go_forward, scroll_at
"""

from core.action_schema import ACTION_SCHEMA, OPTIONAL_FIELDS

def validate_action(action: dict) -> str | None:
    """
    Validate an action dict against the expected schema.
    Returns None if valid, or an error string if invalid.
    
    Enhanced to support:
    - New Google Computer Use actions
    - Optional fields validation
    """
    act = action.get("action")
    if not act:
        return "Missing 'action' field"
    
    if act not in ACTION_SCHEMA:
        return f"Unknown action '{act}' — must be one of {list(ACTION_SCHEMA.keys())}"
    
    required_fields = ACTION_SCHEMA[act]
    
    # Check required fields
    for field in required_fields:
        if field not in action:
            return f"Action '{act}' requires '{field}' field"
            
        # Numeric checks
        if field in ["x", "y", "delta", "seconds", "magnitude"]:
            if not isinstance(action[field], (int, float)):
                return f"Action '{act}' requires numeric '{field}'"
        
        # String checks
        if field in ["text", "direction"]:
            if not isinstance(action[field], str):
                return f"Action '{act}' requires string '{field}'"
    
    # Validate direction for scroll_at
    if act == "scroll_at":
        direction = action.get("direction", "")
        valid_directions = ["up", "down", "left", "right"]
        if direction not in valid_directions:
            return f"scroll_at direction must be one of {valid_directions}, got '{direction}'"
    
    # Validate coordinate bounds (0-1280 for x, 0-800 for y)
    if "x" in action:
        x = action["x"]
        if not (0 <= x <= 1280):
            return f"x coordinate {x} out of bounds (must be 0-1280)"
    
    if "y" in action:
        y = action["y"]
        if not (0 <= y <= 800):
            return f"y coordinate {y} out of bounds (must be 0-800)"
                
    return None
