"""
action_cache.py — Intelligent action caching and retry logic for cost-efficient vision loops.

This module implements a 2-try execution strategy similar to Claude's computer use:
1. First try: Execute action and verify result
2. Second try: If failed, use cached context to retry with correction

Key features:
- Action result caching to avoid redundant API calls
- Visual verification of action success
- Smart retry logic with context awareness
- Cost optimization through reduced API calls
"""

import logging
import hashlib
import json
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class ActionResult:
    """Stores the result of an executed action."""
    action: dict
    success: bool
    visual_change: float  # 0.0 to 1.0
    screenshot_before: str  # base64
    screenshot_after: str  # base64
    timestamp: datetime
    error_message: Optional[str] = None
    retry_count: int = 0


@dataclass
class ActionContext:
    """Context for action execution including history and state."""
    current_url: str
    task: str
    recent_actions: list = field(default_factory=list)
    failed_actions: list = field(default_factory=list)
    successful_patterns: dict = field(default_factory=dict)


class ActionCache:
    """
    Manages action execution history and provides intelligent retry logic.
    
    Implements the 2-try strategy:
    - Try 1: Execute action, verify visually
    - Try 2: If failed, analyze failure and retry with correction
    """
    
    def __init__(self, max_cache_size: int = 100):
        self.cache: Dict[str, ActionResult] = {}
        self.max_cache_size = max_cache_size
        self.context = ActionContext(current_url="", task="")
        
    def _generate_action_key(self, action: dict, context: ActionContext) -> str:
        """Generate a unique key for an action in a given context."""
        # Include action type, coordinates/text, and current URL
        key_data = {
            "action": action.get("action"),
            "x": action.get("x"),
            "y": action.get("y"),
            "text": action.get("text"),
            "url": context.current_url,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def update_context(self, url: str, task: str):
        """Update the current execution context."""
        self.context.current_url = url
        self.context.task = task
    
    def should_retry_action(self, action: dict) -> Tuple[bool, Optional[dict]]:
        """
        Determine if an action should be retried and provide correction.
        
        Returns:
            (should_retry, corrected_action)
        """
        action_key = self._generate_action_key(action, self.context)
        
        # Check if this exact action failed recently
        if action_key in self.cache:
            cached = self.cache[action_key]
            
            # If it failed within last 30 seconds, don't retry the same way
            if not cached.success and (datetime.now() - cached.timestamp) < timedelta(seconds=30):
                logger.warning(f"Action {action.get('action')} failed recently, suggesting alternative")
                return True, self._suggest_alternative_action(action, cached)
            
            # If it succeeded recently, use the same approach
            if cached.success and (datetime.now() - cached.timestamp) < timedelta(seconds=60):
                logger.info(f"Action {action.get('action')} succeeded recently, reusing pattern")
                return False, None
        
        return False, None
    
    def _suggest_alternative_action(self, original_action: dict, failed_result: ActionResult) -> dict:
        """
        Suggest an alternative action based on failure analysis.
        
        This implements intelligent fallback strategies:
        - Click failures → Try double-click or wait for page load
        - Scroll failures → Stop scrolling, try clicking visible links
        - Type failures → Try clicking first to ensure focus
        - Navigate failures → Try waiting for page load
        """
        action_type = original_action.get("action")
        
        # CRITICAL: Detect scroll loops and force different action
        if action_type == "scroll":
            # Count recent scroll actions
            recent_scrolls = sum(1 for a in self.context.recent_actions[-5:] if a.get("action") == "scroll")
            if recent_scrolls >= 3:
                logger.warning(f"Scroll loop detected ({recent_scrolls} recent scrolls), forcing click action")
                # Force a click on a visible link instead of continuing to scroll
                return {
                    "action": "wait",
                    "seconds": 1,
                    "reason": "Breaking scroll loop - waiting for model to identify clickable element"
                }
        
        if action_type == "click":
            # If click failed, try waiting for page to load first
            # Don't scroll - that causes loops. Instead, wait and retry the click.
            return {
                "action": "wait",
                "seconds": 1,
                "reason": "Waiting for page to load before retrying click"
            }
        
        elif action_type == "type":
            # If type failed, ensure field is focused first
            if "x" in original_action and "y" in original_action:
                return {
                    "action": "click",
                    "x": original_action["x"],
                    "y": original_action["y"],
                    "reason": "Clicking to focus field before typing"
                }
        
        elif action_type == "navigate":
            # If navigation failed, wait for page to settle
            return {
                    "action": "wait",
                "seconds": 2,
                "reason": "Waiting for page to settle before navigation"
            }
        
        # Default: wait and retry
        return {
            "action": "wait",
            "seconds": 1,
            "reason": "Waiting before retry"
        }
    
    def record_action_result(
        self,
        action: dict,
        success: bool,
        visual_change: float,
        screenshot_before: str,
        screenshot_after: str,
        error_message: Optional[str] = None
    ):
        """Record the result of an action execution."""
        action_key = self._generate_action_key(action, self.context)
        
        result = ActionResult(
            action=action,
            success=success,
            visual_change=visual_change,
            screenshot_before=screenshot_before,
            screenshot_after=screenshot_after,
            timestamp=datetime.now(),
            error_message=error_message,
            retry_count=0
        )
        
        # Update cache
        self.cache[action_key] = result
        
        # Maintain cache size
        if len(self.cache) > self.max_cache_size:
            # Remove oldest entries
            sorted_keys = sorted(
                self.cache.keys(),
                key=lambda k: self.cache[k].timestamp
            )
            for key in sorted_keys[:len(self.cache) - self.max_cache_size]:
                del self.cache[key]
        
        # Update context
        if success:
            self.context.successful_patterns[action.get("action", "")] = action
            self.context.recent_actions.append(action)
        else:
            self.context.failed_actions.append(action)
        
        # Keep recent actions limited
        if len(self.context.recent_actions) > 10:
            self.context.recent_actions = self.context.recent_actions[-10:]
        if len(self.context.failed_actions) > 5:
            self.context.failed_actions = self.context.failed_actions[-5:]
        
        logger.info(
            f"Recorded action {action.get('action')}: "
            f"success={success}, visual_change={visual_change:.2%}"
        )
    
    def get_success_rate(self, action_type: str) -> float:
        """Calculate success rate for a specific action type."""
        relevant_results = [
            r for r in self.cache.values()
            if r.action.get("action") == action_type
        ]
        
        if not relevant_results:
            return 0.5  # Default 50% if no data
        
        successful = sum(1 for r in relevant_results if r.success)
        return successful / len(relevant_results)
    
    def get_recent_failures(self, limit: int = 5) -> list:
        """Get recent failed actions for analysis."""
        return self.context.failed_actions[-limit:]
    
    def clear_cache(self):
        """Clear the action cache (useful for new tasks)."""
        self.cache.clear()
        self.context = ActionContext(current_url="", task="")
        logger.info("Action cache cleared")
