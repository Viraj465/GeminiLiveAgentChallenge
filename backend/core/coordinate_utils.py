"""
coordinate_utils.py — Coordinate normalization/denormalization utilities.

Google's Computer Use API uses normalized coordinates (0-999) that are resolution-independent.
This module provides utilities to convert between normalized and absolute pixel coordinates.
"""

# Standard viewport dimensions
VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 800

# Google's normalized coordinate range
NORMALIZED_MAX = 1000


def normalize_x(absolute_x: int, viewport_width: int = VIEWPORT_WIDTH) -> int:
    """
    Convert absolute pixel X coordinate to normalized (0-999).
    
    Args:
        absolute_x: Pixel coordinate (0-1280)
        viewport_width: Viewport width in pixels
        
    Returns:
        Normalized coordinate (0-999)
    """
    return int((absolute_x / viewport_width) * NORMALIZED_MAX)


def normalize_y(absolute_y: int, viewport_height: int = VIEWPORT_HEIGHT) -> int:
    """
    Convert absolute pixel Y coordinate to normalized (0-999).
    
    Args:
        absolute_y: Pixel coordinate (0-800)
        viewport_height: Viewport height in pixels
        
    Returns:
        Normalized coordinate (0-999)
    """
    return int((absolute_y / viewport_height) * NORMALIZED_MAX)


def denormalize_x(normalized_x: int, viewport_width: int = VIEWPORT_WIDTH) -> int:
    """
    Convert normalized X coordinate (0-999) to absolute pixels.
    
    Args:
        normalized_x: Normalized coordinate (0-999)
        viewport_width: Viewport width in pixels
        
    Returns:
        Absolute pixel coordinate (0-1280)
    """
    return int((normalized_x / NORMALIZED_MAX) * viewport_width)


def denormalize_y(normalized_y: int, viewport_height: int = VIEWPORT_HEIGHT) -> int:
    """
    Convert normalized Y coordinate (0-999) to absolute pixels.
    
    Args:
        normalized_y: Normalized coordinate (0-999)
        viewport_height: Viewport height in pixels
        
    Returns:
        Absolute pixel coordinate (0-800)
    """
    return int((normalized_y / NORMALIZED_MAX) * viewport_height)


def normalize_coordinates(x: int, y: int, 
                         viewport_width: int = VIEWPORT_WIDTH,
                         viewport_height: int = VIEWPORT_HEIGHT) -> tuple[int, int]:
    """
    Convert absolute pixel coordinates to normalized (0-999, 0-999).
    
    Args:
        x: Absolute X coordinate
        y: Absolute Y coordinate
        viewport_width: Viewport width in pixels
        viewport_height: Viewport height in pixels
        
    Returns:
        Tuple of (normalized_x, normalized_y)
    """
    return (
        normalize_x(x, viewport_width),
        normalize_y(y, viewport_height)
    )


def denormalize_coordinates(normalized_x: int, normalized_y: int,
                           viewport_width: int = VIEWPORT_WIDTH,
                           viewport_height: int = VIEWPORT_HEIGHT) -> tuple[int, int]:
    """
    Convert normalized coordinates (0-999, 0-999) to absolute pixels.
    
    Args:
        normalized_x: Normalized X coordinate (0-999)
        normalized_y: Normalized Y coordinate (0-999)
        viewport_width: Viewport width in pixels
        viewport_height: Viewport height in pixels
        
    Returns:
        Tuple of (absolute_x, absolute_y)
    """
    return (
        denormalize_x(normalized_x, viewport_width),
        denormalize_y(normalized_y, viewport_height)
    )
