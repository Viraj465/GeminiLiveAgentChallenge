"""
synthesis_agent.py — ADK Tool: Cross-paper analysis using Gemini 2.5 Pro.

Uses Gemini's 1M token context to analyze all papers simultaneously.
Produces structured synthesis covering findings, methods, agreements,
contradictions, gaps, and trends.
"""

import os
import logging

from google import genai
from google.genai import types
from config import settings

logger = logging.getLogger(__name__)

# ─── Lazy Gemini init ───
_genai_client = None


def _get_client():
    """Configure genai once using project config."""
    global _genai_client
    if _genai_client is None:
        api_key = settings.GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY is not set")
        _genai_client = genai.Client(api_key=api_key)
    return _genai_client


def synthesize_findings(extractions: list, query: str) -> dict:
    """
    ADK Tool: Uses Gemini 2.5 Pro (1M context) to reason across all papers.
    Produces a structured synthesis of findings, methods, and research gaps.

    Args:
        extractions: List of paper extraction dicts from extraction_agent.
        query: The original research query.

    Returns:
        dict with status, synthesis text, and paper count.
    """
    client = _get_client()

    # Combine all paper texts + figure descriptions into one context
    combined_text = ""
    paper_count = 0
    for i, paper in enumerate(extractions):
        if "error" in paper:
            continue

        paper_count += 1
        combined_text += f"\n\n=== PAPER {paper_count}: {paper.get('name', 'Unknown')} ===\n"
        combined_text += paper.get("text", "")

        # Include figure descriptions if available
        figures = paper.get("figure_descriptions", [])
        if figures:
            combined_text += "\n\n--- FIGURES ---\n"
            for fig in figures:
                combined_text += f"Page {fig.get('page', '?')}: {fig.get('description', '')}\n"

    if not combined_text.strip():
        return {
            "status": "success",
            "synthesis": "No paper text available for synthesis.",
            "papers_analyzed": 0,
            "total_chars_processed": 0,
        }

    # Gemini 2.5 Pro handles up to 1M tokens
    model_name = "gemini-2.5-pro"

    synthesis_prompt = f"""
    You are a research synthesis expert. Below are full texts from multiple academic papers.
    
    Research Query: {query}
    
    Analyze ALL papers and provide:
    1. KEY FINDINGS: What are the main findings across all papers?
    2. METHODOLOGIES: What methods do they use? How do they compare?
    3. AGREEMENTS: Where do all papers agree?
    4. CONTRADICTIONS: Where do papers disagree? Why?
    5. RESEARCH GAPS: What questions remain unanswered?
    6. TRENDS: How has this field evolved across the papers?
    7. FIGURE INSIGHTS: If figure descriptions are provided, what do the visual data show?
    
    Be specific. Cite paper numbers when referencing. Be thorough.
    
    PAPERS:
    {combined_text[:900000]}
    """

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=synthesis_prompt,
            config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=8192),
        )

        return {
            "status": "success",
            "synthesis": response.text,
            "papers_analyzed": paper_count,
            "total_chars_processed": len(combined_text),
            "token_usage": {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "candidates_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
            }
        }

    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return {
            "status": "error",
            "message": str(e),
            "papers_analyzed": paper_count,
            "total_chars_processed": len(combined_text),
        }