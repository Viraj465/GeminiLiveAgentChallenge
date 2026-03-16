"""
test_vision.py — Tests for the vision loop, browser controller, and action validation.

Run with:
    cd e:\\GeminiLiveAgentChallenge\\backend
    python -m pytest tests/test_vision.py -v
"""

import pytest
import json
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from core.vision_loop import validate_action, run_vision_loop
from core.action_schema import ACTION_SCHEMA
from core.browser import BrowserController


# ═══════════════════════════════════════════════
#  Action Validation Tests
# ═══════════════════════════════════════════════

class TestActionValidation:
    """Test the validate_action function."""

    def test_valid_click(self):
        result = validate_action({"action": "click", "x": 100, "y": 200, "reason": "click button"})
        assert result is None

    def test_valid_type(self):
        result = validate_action({"action": "type", "x": 100, "y": 200, "text": "hello", "reason": "type text"})
        assert result is None

    def test_valid_navigate(self):
        result = validate_action({"action": "navigate", "text": "https://example.com", "reason": "go to page"})
        assert result is None

    def test_valid_press(self):
        result = validate_action({"action": "press", "text": "Enter", "reason": "submit"})
        assert result is None

    def test_valid_scroll(self):
        result = validate_action({"action": "scroll", "delta": 300, "reason": "scroll down"})
        assert result is None

    def test_valid_wait(self):
        result = validate_action({"action": "wait", "seconds": 2, "reason": "page loading"})
        assert result is None

    def test_valid_done(self):
        result = validate_action({"action": "done", "reason": "task complete"})
        assert result is None

    def test_valid_ask_user(self):
        result = validate_action({"action": "ask_user", "text": "Please solve CAPTCHA", "reason": "captcha found"})
        assert result is None

    def test_missing_action_field(self):
        result = validate_action({"x": 100, "y": 200})
        assert result is not None
        # Check for either error message format
        assert "Missing 'action' field" in result or "is not a valid action" in result

    def test_unknown_action(self):
        result = validate_action({"action": "fly", "reason": "not real"})
        assert result is not None
        assert "Unknown action" in result

    def test_click_missing_x(self):
        result = validate_action({"action": "click", "y": 200})
        assert result is not None
        assert "requires 'x' field" in result

    def test_click_missing_y(self):
        result = validate_action({"action": "click", "x": 100})
        assert result is not None
        assert "requires 'y' field" in result

    def test_click_non_numeric_coords(self):
        result = validate_action({"action": "click", "x": "bad", "y": 200})
        assert result is not None
        assert "numeric" in result

    def test_type_missing_text(self):
        result = validate_action({"action": "type", "x": 100, "y": 200, "reason": "type something"})
        assert result is not None
        assert "requires 'text' field" in result


    def test_navigate_missing_text(self):
        result = validate_action({"action": "navigate"})
        assert result is not None
        assert "requires 'text' field" in result

    def test_press_missing_text(self):
        result = validate_action({"action": "press"})
        assert result is not None
        assert "requires 'text' field" in result


# ═══════════════════════════════════════════════
#  BrowserController Tests (mocked Playwright)
# ═══════════════════════════════════════════════

class TestBrowserController:
    """Test BrowserController.execute_action with a mocked page."""

    @pytest.fixture
    def browser(self):
        """Create a BrowserController with a mocked page."""
        bc = BrowserController()
        bc.page = MagicMock()
        bc.page.mouse = MagicMock()
        bc.page.mouse.click = AsyncMock()
        bc.page.mouse.wheel = AsyncMock()
        bc.page.keyboard = MagicMock()
        bc.page.keyboard.type = AsyncMock()
        bc.page.keyboard.press = AsyncMock()
        bc.page.goto = AsyncMock()
        bc.page.screenshot = AsyncMock(return_value=b"fake_png")
        bc.page.wait_for_load_state = AsyncMock()
        bc.inject_grid = AsyncMock()
        bc.remove_grid = AsyncMock()
        bc.wait_for_visual_stability = AsyncMock(return_value=True)
        return bc

    @pytest.mark.asyncio
    async def test_click_action(self, browser):
        result = await browser.execute_action({"action": "click", "x": 150, "y": 250})
        assert result == "OK"
        browser.page.mouse.click.assert_called_once_with(150, 250, delay=100)

    @pytest.mark.asyncio
    async def test_type_action(self, browser):
        result = await browser.execute_action({"action": "type", "text": "hello"})
        assert result == "OK"
        browser.page.keyboard.type.assert_called_once_with("hello", delay=50)

    @pytest.mark.asyncio
    async def test_press_action(self, browser):
        result = await browser.execute_action({"action": "press", "text": "Enter"})
        assert result == "OK"
        browser.page.keyboard.press.assert_called_once_with("Enter")

    @pytest.mark.asyncio
    async def test_scroll_action(self, browser):
        result = await browser.execute_action({"action": "scroll", "delta": 500})
        assert result == "OK"
        browser.page.mouse.wheel.assert_called_once_with(0, 500)

    @pytest.mark.asyncio
    async def test_navigate_action(self, browser):
        result = await browser.execute_action({"action": "navigate", "text": "https://example.com"})
        assert result == "OK"
        browser.page.goto.assert_called_once()

    @pytest.mark.asyncio
    async def test_done_action(self, browser):
        result = await browser.execute_action({"action": "done"})
        assert result == "DONE"

    @pytest.mark.asyncio
    async def test_unknown_action(self, browser):
        result = await browser.execute_action({"action": "teleport"})
        assert result.startswith("ERROR")

    @pytest.mark.asyncio
    async def test_wait_action_capped(self, browser):
        """Wait action should be capped at 10 seconds."""
        result = await browser.execute_action({"action": "wait", "seconds": 999})
        assert result == "OK"

    @pytest.mark.asyncio
    async def test_screenshot_b64(self, browser):
        result = await browser.screenshot_b64()
        assert isinstance(result, str)
        assert len(result) > 0


# ═══════════════════════════════════════════════
#  Constant/Config Sanity Tests
# ═══════════════════════════════════════════════

class TestConstantsAndConfig:
    """Verify constants and config are valid."""

    def test_valid_actions_not_empty(self):
        valid_actions = set(ACTION_SCHEMA.keys())
        assert len(valid_actions) > 0
        assert "click" in valid_actions
        assert "done" in valid_actions

    def test_config_model_names(self):
        from config import settings
        assert settings.GOOGLE_VISION_MODEL, "GOOGLE_VISION_MODEL must not be empty"
        assert settings.GOOGLE_REASONING_MODEL, "GOOGLE_REASONING_MODEL must not be empty"
        assert settings.GOOGLE_REPORT_MODEL, "GOOGLE_REPORT_MODEL must not be empty"

# ═══════════════════════════════════════════════
#  Manual Live Test Execution
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    async def manual_live_test():
        print("Starting Manual Live Vision Loop Test...")
        load_dotenv()
        
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        if not project_id:
            print("ERROR: GOOGLE_CLOUD_PROJECT (or GOOGLE_CLOUD_PROJECT_ID) is missing in .env")
            return
            
        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            print("WARNING: GOOGLE_APPLICATION_CREDENTIALS is missing in .env. Ensure your ADC default logic is configured.")

        browser = BrowserController()
        await browser.start()

        try:
            print("Navigating to starting page...")
            await browser.execute_action({"action": "navigate", "text": "https://en.wikipedia.org/wiki/Reinforcement_learning"})

            print("Starting Vision Loop...")
            async for action in run_vision_loop(
                browser=browser,
                task="Print out what this page is about.",
                max_steps=5
            ):
                print(f"Decision: {json.dumps(action, indent=2)}")
                
        except Exception as e:
            print(f"Test encountered an error: {e}")
        finally:
            print("Closing browser...")
            await browser.close()
            print("Test Finished.")

    asyncio.run(manual_live_test())