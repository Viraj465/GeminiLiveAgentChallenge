
"""
action_schema.py — Definition of valid actions and their required fields.

Enhanced with Google Computer Use actions:
- hover_at: Hover over element (useful for dropdowns/menus)
- go_back: Browser back navigation
- go_forward: Browser forward navigation
- scroll_at: Scroll at specific coordinates (element-specific scrolling)
"""

ACTION_SCHEMA = {
    # Original actions (kept for compatibility)
    "click": ["x", "y"],
    "type": ["x", "y", "text"],
    "scroll": ["delta"],
    "navigate": ["text"],
    "press": ["text"],
    "wait": ["seconds"],
    "done": [],
    "ask_user": ["reason"],
    
    # Google Computer Use actions
    # Reference: https://ai.google.dev/gemini-api/docs/computer-use
    "double_click": ["x", "y"],  # Double-click at coordinates
    "right_click": ["x", "y"],  # Right/context click at coordinates
    "long_press": ["x", "y"],  # Long-press (click and hold)
    "hover_at": ["x", "y"],  # Hover over element to reveal dropdowns/menus
    "go_back": [],  # Browser back button
    "go_forward": [],  # Browser forward button
    "scroll_at": ["x", "y", "direction"],  # Scroll at specific element (direction: "up"/"down"/"left"/"right")
    "drag": ["start_x", "start_y", "end_x", "end_y"],  # Mouse drag from start to end
}

VALID_ACTIONS = set(ACTION_SCHEMA.keys())

# Optional fields for actions
OPTIONAL_FIELDS = {
    "type": ["clear", "press_enter"],  # clear: clear before typing, press_enter: press enter after typing
    "scroll_at": ["magnitude"],  # magnitude: scroll amount (default: 300)
    "long_press": ["duration"],  # duration: hold time in seconds (default: 1)
}
