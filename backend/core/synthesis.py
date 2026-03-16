"""
synthesis.py — Academic Literature Review Generator using Gemini.
"""

import asyncio
import logging
import os
from google import genai
from google.genai import types
from config import settings
from prompts import REPORT_PROMPT_TEMPLATE


logger = logging.getLogger(__name__)


def _get_client():
    """Get a Gemini/Vertex AI client for synthesis."""
    from dotenv import load_dotenv
    load_dotenv()

    project_id = settings.VERTEX_AI_PROJECT or settings.PROJECT_ID
    if not project_id:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    location = settings.VERTEX_AI_LOCATION or "global"

    if project_id:
        return genai.Client(vertexai=True, project=project_id, location=location)

    api_key = settings.GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)

    raise EnvironmentError("No Gemini API credentials available for synthesis")


async def generate_literature_review(
    topic: str,
    extracted_texts: dict[str, str],
    synthesis: str = "",
    graph: dict = None,
) -> str:
    """
    Generates a publication-ready literature review using REPORT_PROMPT_TEMPLATE.

    Args:
        topic:           The research query / topic string.
        extracted_texts: Mapping of 'Paper Title' → 'Raw Extracted Text'.
        synthesis:       Optional pre-computed synthesis from synthesize_findings().
                         If provided, it is passed directly into the report prompt
                         instead of re-synthesising from raw text.
        graph:           Optional citation graph dict with 'edge_count' key.

    Returns:
        Markdown-formatted literature review string.
    """
    if not extracted_texts:
        return "No text provided to generate a literature review."

    try:
        client = _get_client()
    except EnvironmentError as e:
        logger.error(str(e))
        return f"Error: {e}"

    # Use a model from settings
    model_name = (
        settings.GOOGLE_REPORT_MODEL
        if hasattr(settings, "GOOGLE_REPORT_MODEL")
        else "gemini-2.5-pro"
    )

    paper_count = len(extracted_texts)
    edge_count = (graph or {}).get("edge_count", 0)

    # If no pre-computed synthesis was passed, build a combined text block
    # so the report prompt still has something meaningful in {synthesis}.
    if not synthesis:
        combined_parts = []
        for i, (title, text) in enumerate(extracted_texts.items(), start=1):
            combined_parts.append(f"[Paper {i}] {title}\n{text}")
        synthesis = "\n\n---\n\n".join(combined_parts)

    # Fill in the REPORT_PROMPT_TEMPLATE
    user_prompt = REPORT_PROMPT_TEMPLATE.format(
        query=topic,
        paper_count=paper_count,
        edge_count=edge_count,
        synthesis=synthesis,
    )

    logger.info(
        f"Generating literature review for '{topic}' "
        f"({paper_count} papers, {edge_count} citation edges) …"
    )

    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=model_name,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=user_prompt)],
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.2,  # Low temp for factual synthesis
                ),
            ),
            timeout=180.0,  # Report generation can take longer than synthesis
        )

        logger.info("Successfully generated literature review.")
        return response.text.strip()

    except asyncio.TimeoutError:
        error_msg = "Gemini API timed out during literature review generation."
        logger.error(error_msg)
        return f"Error: {error_msg}"

    except Exception as e:
        logger.error(f"Failed to generate literature review: {e}", exc_info=True)
        return f"Error generating literature review: {str(e)}"
