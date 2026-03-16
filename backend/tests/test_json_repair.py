import json
import pytest
from core.vision_loop import try_repair_json as repair_standard
from core.vision_loop_optimized import try_repair_json as repair_optimized

@pytest.mark.parametrize("repair_func", [repair_standard, repair_optimized])
def test_repair_valid_json(repair_func):
    """Test standard valid JSON (should not be changed)."""
    raw = '{"action": "click", "x": 100, "y": 200, "reason": "test"}'
    result = repair_func(raw)
    assert result["action"] == "click"
    assert result["x"] == 100

@pytest.mark.parametrize("repair_func", [repair_standard, repair_optimized])
def test_repair_backticks_complete(repair_func):
    """Test JSON wrapped in complete markdown backticks."""
    raw = '```json\n{"action": "wait", "seconds": 2, "reason": "test"}\n```'
    result = repair_func(raw)
    assert result["action"] == "wait"

@pytest.mark.parametrize("repair_func", [repair_standard, repair_optimized])
def test_repair_backticks_truncated(repair_func):
    """Test JSON wrapped in truncated markdown backticks."""
    raw = '```json\n{"action": "wait", "seconds": 2, "reason": "test"'
    result = repair_func(raw)
    assert result["action"] == "wait"

@pytest.mark.parametrize("repair_func", [repair_standard, repair_optimized])
def test_repair_truncated_string(repair_func):
    """Test JSON truncated in the middle of a string value."""
    raw = '{"action": "type", "x": 485, "y": 441, "text": "Vision Transformers for medical image analysis and cancer detection", "press_enter": true, "reason": "typing search query into Google search bar and sub'
    result = repair_func(raw)
    assert result is not None
    assert result["action"] == "type"
    assert result["text"] == "Vision Transformers for medical image analysis and cancer detection"

@pytest.mark.parametrize("repair_func", [repair_standard, repair_optimized])
def test_repair_missing_multiple_braces(repair_func):
    """Test JSON with multiple missing closing braces."""
    raw = '{"action": "click", "data": {"coords": {"x": 10, "y": 20}'
    result = repair_func(raw)
    assert result["action"] == "click"

@pytest.mark.parametrize("repair_func", [repair_standard, repair_optimized])
def test_repair_interior_braces(repair_func):
    """Test truncation with interior braces in strings."""
    raw = '{"action": "press", "text": "Enter", "reason": "Found something {like this}"'
    result = repair_func(raw)
    assert result["action"] == "press"

@pytest.mark.parametrize("repair_func", [repair_standard, repair_optimized])
def test_repair_escaped_quotes(repair_func):
    """Test quote balancing with escaped quotes."""
    raw = '{"action": "type", "text": "Quotes like \\"this\\" are tricky", "reason": "typing'
    result = repair_func(raw)
    assert result is not None
    assert result["text"] == 'Quotes like "this" are tricky'

@pytest.mark.parametrize("repair_func", [repair_standard, repair_optimized])
def test_repair_with_preamble(repair_func):
    """Test JSON with text preamble."""
    raw = 'Based on the screen, I will take this action: {"action": "scroll", "delta": 400}'
    result = repair_func(raw)
    assert result["action"] == "scroll"

@pytest.mark.parametrize("repair_func", [repair_standard, repair_optimized])
def test_repair_empty(repair_func):
    """Test empty input."""
    assert repair_func("") is None
    assert repair_func("   ") is None

@pytest.mark.parametrize("repair_func", [repair_standard, repair_optimized])
def test_repair_complete_bullshit(repair_func):
    """Test non-JSON input."""
    assert repair_func("This is not JSON at all") is None

def test_optimized_defaults():
    """Test that optimized loop adds specific defaults."""
    raw = '{"action": "type", "x": 10, "y": 20, "text": "abc"' # Truncated
    result = repair_optimized(raw)
    assert result["action"] == "type"
    assert result["press_enter"] is False
    assert "executing type" in result["reason"]
    
    raw_scroll = '{"action": "scroll_at", "x": 10, "y": 20' # Truncated
    result_scroll = repair_optimized(raw_scroll)
    assert result_scroll["action"] == "scroll_at"
    assert result_scroll["magnitude"] == 300
