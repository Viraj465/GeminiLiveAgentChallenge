"""
report_agent.py — ADK Tool: Generate publication-ready literature review.

Uses Gemini 2.5 Pro to produce a structured Markdown literature review
from the synthesis and citation graph data.
"""

import os
import logging
from datetime import datetime

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


def generate_report(synthesis: str, citation_graph: dict, query: str) -> dict:
    """
    ADK Tool: Generates a publication-ready literature review using Gemini 2.5 Pro.
    Output is a structured Markdown document.

    Args:
        synthesis: Synthesis text from synthesis_agent.
        citation_graph: Graph data dict from citation_agent.
        query: The original research query.

    Returns:
        dict with status, report markdown, word count, and timestamp.
    """
    client = _get_client()

    model_name = "gemini-2.5-pro"

    report_prompt = f"""
    You are an expert academic writer. Using the synthesis below, write a complete
    literature review in academic style.
    
    Research Query: {query}
    Papers Analyzed: {citation_graph.get('nodes', 0)}
    Citation Relationships Found: {citation_graph.get('edges', 0)}
    
    SYNTHESIS:
    {synthesis}
    
    Write the literature review with these sections:
    # Literature Review: [Topic]
    ## Abstract (150 words)
    ## 1. Introduction
    ## 2. Methodology of Review
    ## 3. Key Findings
    ## 4. Comparative Analysis
    ## 5. Research Gaps & Future Directions
    ## 6. Conclusion
    ## References
    
    Use formal academic language. Be comprehensive. Use markdown formatting.
    """

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=report_prompt,
            config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=8192),
        )

        report = response.text

        return {
            "status": "success",
            "report": report,
            "word_count": len(report.split()),
            "generated_at": datetime.utcnow().isoformat(),
            "query": query,
            "token_usage": {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "candidates_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
            }
        }

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return {
            "status": "error",
            "message": str(e),
            "query": query,
        }