"""
analysis.py — Gemini Vision utility for figure and table analysis.
"""

import logging
import asyncio
from google.genai import types
from config import settings
from prompts import FIGURE_ANALYSIS_TEMPLATE, TABLE_ANALYSIS_TEMPLATE
from core.vision_loop import _get_client

logger = logging.getLogger(__name__)

async def analyze_figure(base64_image: str, context: str = "") -> str:
    """
    Analyze a figure/graph image using Gemini 2.0 Flash.
    """
    client = _get_client()
    model_name = settings.GOOGLE_VISION_MODEL
    
    prompt = FIGURE_ANALYSIS_TEMPLATE.format(context=context)
    
    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=model_name,
                contents=[
                    types.Content(role="user", parts=[
                        types.Part.from_bytes(data=base64_image, mime_type="image/png"), # MIME might vary but PyMuPDF extracts png/jpeg
                        types.Part.from_text(text=prompt)
                    ])
                ],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=1024
                )
            ),
            timeout=30.0
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Figure analysis failed: {e}")
        return f"[Figure Analysis Error]: {str(e)}"

async def analyze_table(table_data: list[list], context: str = "") -> str:
    """
    Interprets tabular data using Gemini.
    """
    client = _get_client()
    model_name = settings.GOOGLE_REASONING_MODEL # Usually a Pro model for better reasoning
    
    # Convert nested list table to markdown-ish string for the prompt
    table_str = "\n".join([" | ".join([str(cell) for cell in row]) for row in table_data])
    
    prompt = TABLE_ANALYSIS_TEMPLATE.format(context=context, table_data=table_str)
    
    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=1024,
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=True,
                    ),
                )
            ),
            timeout=30.0
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Table analysis failed: {e}")
        return f"[Table Analysis Error]: {str(e)}"
