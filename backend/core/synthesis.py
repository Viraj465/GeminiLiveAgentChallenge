"""
synthesis.py — Academic Literature Review Generator using Gemini 2.0 Flash.
"""

import asyncio
import logging
from google import genai
from google.genai import types
from config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are an expert academic researcher and writer.
You will be provided with the raw extracted text from several academic papers on a specific topic.
Your task is to synthesize these papers into a comprehensive, professional Literature Review in Markdown format.

Your literature review MUST include:
1. **Introduction**: A clear overview of the topic and the scope of the papers provided.
2. **Thematic Analysis**: Group the findings by themes or methodologies, rather than just listing papers one by one.
3. **Methodology Comparison**: Compare and contrast the different approaches or methods used in the provided papers.
4. **Conclusion & Future Directions**: Summarize the current state of the field based on these papers and identify gaps or future research directions.

Rules:
- Format the output strictly in Markdown.
- Use proper heading hierarchies (#, ##, ###).
- Cite the papers in-text using the provided paper titles (e.g., [Title of Paper]).
- Do NOT hallucinate information. If the provided texts do not cover a specific aspect, do not invent details.
- Keep the tone academic, objective, and analytical.
"""


async def generate_literature_review(topic: str, extracted_texts: dict[str, str]) -> str:
    """
    ADK Tool: Generates a literature review from a dictionary of academic texts.
    Takes a 'topic' string and a dictionary mapping 'Paper Title' to 'Raw Extracted Text'.
    Returns a Markdown formatted string.
    """
    if not extracted_texts:
        return "No text provided to generate a literature review."

    api_key = settings.GEMINI_API_KEY
    if not api_key:
        logger.error("GOOGLE_API_KEY is not set.")
        return "Error: GOOGLE_API_KEY not configured."

    try:
        # Initialize the client. Under the hood, this integrates with Vertex AI if configured.
        from core.vision_loop import _get_client
        client = _get_client()
        
        # Use a model from settings
        model_name = settings.GOOGLE_REPORT_MODEL if hasattr(settings, "GOOGLE_REPORT_MODEL") else "gemini-2.5-pro"

        # Construct the user prompt with the extracted texts
        user_prompt = f"Topic: {topic}\n\nHere are the extracted texts from the papers:\n\n"
        
        for title, text in extracted_texts.items():
            user_prompt += f"--- Paper: {title} ---\n{text}\n\n"
            
        user_prompt += "Please generate the literature review based ONLY on the texts above."

        logger.info(f"Sending {len(extracted_texts)} papers to Gemini for synthesis...")

        # We use wait_for so the large extraction doesn't hang the loop infinitely
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=model_name,
                contents=[
                    types.Content(role="user", parts=[
                        types.Part.from_text(text=SYSTEM_PROMPT),
                        types.Part.from_text(text=user_prompt)
                    ])
                ],
                config=types.GenerateContentConfig(
                    temperature=0.2, # Low temp for factual synthesis
                ),
            ),
            timeout=120.0, # Synthesis takes longer
        )

        logger.info("Successfully generated literature review.")
        return response.text.strip()

    except asyncio.TimeoutError:
        error_msg = "Gemini API timed out during synthesis."
        logger.error(error_msg)
        return f"Error: {error_msg}"
        
    except Exception as e:
        logger.error(f"Failed to generate literature review: {e}", exc_info=True)
        return f"Error generating literature review: {str(e)}"
