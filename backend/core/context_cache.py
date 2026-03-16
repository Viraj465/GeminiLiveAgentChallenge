"""
context_cache.py — Gemini Context Caching for full-paper deep analysis.

Replaces the 30-50 screenshot scroll loop with:
  1. One cache creation call  (loads entire PDF into Gemini memory)
  2. 5-8 targeted Q&A calls   (extract structured provenance data)
  3. One cache deletion call  (cleanup + stop billing)

The cache holds the entire multimodal PDF — text, figures, tables, diagrams —
so every query gets full-document context at a fraction of the token cost.

Provenance schema per extracted claim:
  {
    "claim":           str,   # The factual statement
    "section":         str,   # Section heading (e.g. "Results", "Methodology")
    "page":            int,   # Page number in the PDF
    "paragraph_index": int,   # Paragraph index within the section (0-based)
    "evidence_type":   str,   # "metric" | "method" | "limitation" | "figure" | "table"
    "dataset":         str,   # Dataset name if applicable
    "hardware":        str,   # Hardware/compute if applicable
    "limitation":      str,   # Limitation text if evidence_type == "limitation"
    "raw_quote":       str,   # Verbatim sentence from the paper
    "bounding_box":    dict   # {x, y, width, height} normalized 0-1, for figures/tables
  }
"""

import asyncio
import logging
import json
import os
from typing import Any

from google import genai
from google.genai import types

from config import settings

logger = logging.getLogger(__name__)

# ─── Gemini client (lazy singleton) ───
_genai_client: genai.Client | None = None

# ─── Targeted extraction questions ───
# Each question is sent as a separate query against the cached paper.
# Structured JSON output is requested for every call.
_EXTRACTION_QUERIES = [
    {
        "key": "methodology",
        "question": (
            "Extract the complete methodology of this paper as structured JSON. "
            "Return a JSON object with these fields:\n"
            "{\n"
            '  "approach": "one-sentence description of the core method",\n'
            '  "datasets": ["list of dataset names used"],\n'
            '  "hardware": "GPU/TPU/compute description if mentioned",\n'
            '  "training_details": "batch size, epochs, optimizer, learning rate if mentioned",\n'
            '  "evaluation_protocol": "how results were measured",\n'
            '  "baseline_comparisons": ["list of baselines compared against"],\n'
            '  "section": "section heading where methodology is described",\n'
            '  "page": page_number_integer,\n'
            '  "problem_statement": "explain the Why - what gap did this specific paper fill?",\n'
            '  "contribution_type": "empirical or theoretical",\n'
            '  "research_theme": "the overarching sub-field or topic (e.g. AI, Math, CV, NLP)"\n'
            "}\n"
            "Return ONLY the JSON object. No markdown fences."
        ),
    },
    {
        "key": "key_claims",
        "question": (
            "Extract ALL quantitative claims and key findings from this paper as a JSON array. "
            "For each claim return:\n"
            "[\n"
            "  {\n"
            '    "claim": "exact factual statement",\n'
            '    "section": "section heading",\n'
            '    "page": page_number_integer,\n'
            '    "paragraph_index": paragraph_index_integer,\n'
            '    "evidence_type": "metric|method|finding",\n'
            '    "dataset": "dataset name or null",\n'
            '    "hardware": "hardware description or null",\n'
            '    "raw_quote": "verbatim sentence from the paper"\n'
            "  }\n"
            "]\n"
            "Include ALL numeric results (accuracy, F1, BLEU, perplexity, latency, etc.). "
            "Return ONLY the JSON array. No markdown fences."
        ),
    },
    {
        "key": "limitations",
        "question": (
            "Extract ALL limitations, failure cases, and future work items explicitly stated "
            "by the authors. Return a JSON array:\n"
            "[\n"
            "  {\n"
            '    "text": "limitation description",\n'
            '    "section": "section heading",\n'
            '    "page": page_number_integer,\n'
            '    "raw_quote": "verbatim sentence from the paper"\n'
            "  }\n"
            "]\n"
            "Return ONLY the JSON array. No markdown fences."
        ),
    },
    {
        "key": "figures_tables",
        "question": (
            "Extract metadata for ALL figures and tables in this paper. Return a JSON array:\n"
            "[\n"
            "  {\n"
            '    "label": "Figure 3 or Table 2",\n'
            '    "type": "figure|table",\n'
            '    "page": page_number_integer,\n'
            '    "caption": "full caption text",\n'
            '    "key_finding": "one sentence: what does this figure/table prove?",\n'
            '    "data_points": ["list of key numeric values visible"],\n'
            '    "bounding_box": {"x": 0.0, "y": 0.0, "width": 1.0, "height": 0.3}\n'
            "  }\n"
            "]\n"
            "For bounding_box: x,y is top-left corner, all values normalized 0.0-1.0 relative to page size. "
            "Return ONLY the JSON array. No markdown fences."
        ),
    },
    {
        "key": "abstract_and_conclusion",
        "question": (
            "Extract the abstract and conclusion of this paper. Return a JSON object:\n"
            "{\n"
            '  "title": "full paper title",\n'
            '  "authors": ["list of author names"],\n'
            '  "year": publication_year_integer_or_null,\n'
            '  "abstract": "full abstract text",\n'
            '  "abstract_page": page_number_integer,\n'
            '  "conclusion": "full conclusion text",\n'
            '  "conclusion_page": page_number_integer,\n'
            '  "primary_contribution": "one sentence: what is the main contribution?",\n'
            '  "keywords": ["list of keywords if present"]\n'
            "}\n"
            "Return ONLY the JSON object. No markdown fences."
        ),
    },
    {
        "key": "citations_in_text",
        "question": (
            "Extract ALL in-text citations from this paper's Related Work and Introduction sections. "
            "For each citation return:\n"
            "[\n"
            "  {\n"
            '    "cited_work": "author name(s) and year as they appear in text, e.g. Smith et al. (2023)",\n'
            '    "context_sentence": "the full sentence containing this citation",\n'
            '    "section": "section heading",\n'
            '    "page": page_number_integer,\n'
            '    "relationship": "builds_on|contrasts_with|uses_dataset_from|improves_upon|cites_for_background"\n'
            "  }\n"
            "]\n"
            "Return ONLY the JSON array. No markdown fences."
        ),
    },
]


def _get_client() -> genai.Client:
    global _genai_client
    if _genai_client is None:
        project_id = settings.VERTEX_AI_PROJECT or settings.PROJECT_ID or os.getenv("GOOGLE_CLOUD_PROJECT")
        location = settings.VERTEX_AI_LOCATION or "global"
        if project_id:
            _genai_client = genai.Client(vertexai=True, project=project_id, location=location)
        else:
            api_key = settings.GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise EnvironmentError("No Gemini credentials available for context caching.")
            _genai_client = genai.Client(api_key=api_key)
    return _genai_client


async def create_paper_cache(gcs_uri: str, title: str = "") -> str | None:
    """
    Load a PDF from GCS into a Gemini Context Cache.

    Args:
        gcs_uri: GCS URI of the PDF, e.g. "gs://bucket/papers/abc.pdf"
        title:   Paper title for logging.

    Returns:
        Cache name string (e.g. "cachedContents/abc123xyz") or None on failure.
    """
    if not settings.ENABLE_CONTEXT_CACHING:
        logger.info("Context caching disabled via ENABLE_CONTEXT_CACHING=false")
        return None

    client = _get_client()
    model_name = settings.GOOGLE_REASONING_MODEL  # gemini-3.1-pro-preview

    try:
        cache = await asyncio.to_thread(
            client.caches.create,
            model=model_name,
            config=types.CreateCachedContentConfig(
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_uri(
                                file_uri=gcs_uri,
                                mime_type="application/pdf",
                            )
                        ],
                    )
                ],
                system_instruction=(
                    "You are an expert academic researcher analyzing a research paper. "
                    "The full PDF of the paper is loaded in your context. "
                    "Answer all questions with precise page numbers, section headings, "
                    "and verbatim quotes. Always return valid JSON as instructed."
                ),
                ttl=f"{settings.CONTEXT_CACHE_TTL_SECONDS}s",
                display_name=f"paper_cache_{title[:40].replace(' ', '_')}",
            ),
        )
        logger.info(f"Context cache created: {cache.name} for '{title}'")
        return cache.name

    except Exception as e:
        logger.error(f"Failed to create context cache for {gcs_uri}: {e}")
        return None


async def query_cached_paper(cache_name: str, question: str) -> str:
    """
    Query a cached paper with a single question.

    Args:
        cache_name: Cache name from create_paper_cache().
        question:   The question to ask about the paper.

    Returns:
        Response text string (expected to be JSON per our prompts).
    """
    client = _get_client()
    model_name = settings.GOOGLE_REASONING_MODEL

    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=model_name,
                contents=question,
                config=types.GenerateContentConfig(
                    cached_content=cache_name,
                    temperature=0.0,
                    max_output_tokens=8192,
                ),
            ),
            timeout=120.0,
        )
        return response.text.strip() if response.text else ""

    except asyncio.TimeoutError:
        logger.warning(f"Cache query timed out for cache {cache_name}")
        return ""
    except Exception as e:
        logger.warning(f"Cache query failed for cache {cache_name}: {e}")
        return ""


async def delete_paper_cache(cache_name: str) -> bool:
    """
    Delete a Gemini Context Cache to stop storage billing.

    Args:
        cache_name: Cache name from create_paper_cache().

    Returns:
        True if deleted, False otherwise.
    """
    if not cache_name:
        return False

    client = _get_client()
    try:
        await asyncio.to_thread(client.caches.delete, name=cache_name)
        logger.info(f"Deleted context cache: {cache_name}")
        return True
    except Exception as e:
        logger.warning(f"Failed to delete context cache {cache_name}: {e}")
        return False


def _safe_parse_json(text: str, fallback: Any = None) -> Any:
    """
    Safely parse JSON from a Gemini response.
    Strips markdown fences if present before parsing.
    """
    if not text:
        return fallback

    # Strip markdown code fences
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```json or ```) and last line (```)
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        cleaned = "\n".join(inner).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON array or object within the text
        for start_char, end_char in [("[", "]"), ("{", "}")]:
            start = cleaned.find(start_char)
            end = cleaned.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(cleaned[start : end + 1])
                except json.JSONDecodeError:
                    pass
        logger.warning(f"Could not parse JSON from response: {cleaned[:200]}")
        return fallback


async def extract_full_paper_via_cache(
    gcs_uri: str,
    title: str = "",
    url: str = "",
) -> dict:
    """
    Full pipeline: create cache → run all targeted queries → delete cache.

    Args:
        gcs_uri: GCS URI of the uploaded PDF.
        title:   Paper title for logging and display.
        url:     Original paper URL (stored in provenance).

    Returns:
        Structured extraction dict with provenance data:
        {
          "title":          str,
          "authors":        list,
          "year":           int | None,
          "abstract":       str,
          "conclusion":     str,
          "methodology":    dict,
          "key_claims":     list[dict],   ← with page + section + raw_quote
          "limitations":    list[dict],   ← with page + section + raw_quote
          "figures_tables": list[dict],   ← with bounding_box + page
          "citations_in_text": list[dict],← with context_sentence + page
          "text":           str,          ← combined text for synthesis
          "cache_name":     str,          ← for cleanup
          "gcs_uri":        str,          ← for cleanup
          "extraction_method": "context_cache",
          "status":         "success" | "partial" | "error"
        }
    """
    result: dict = {
        "title": title,
        "authors": [],
        "year": None,
        "abstract": "",
        "conclusion": "",
        "methodology": {},
        "key_claims": [],
        "limitations": [],
        "figures_tables": [],
        "citations_in_text": [],
        "text": "",
        "cache_name": None,
        "gcs_uri": gcs_uri,
        "url": url,
        "extraction_method": "context_cache",
        "status": "error",
    }

    # ── Step 1: Create the context cache ──
    cache_name = await create_paper_cache(gcs_uri, title)
    if not cache_name:
        logger.warning(f"Cache creation failed for '{title}' — falling back to PyMuPDF path")
        result["status"] = "cache_failed"
        return result

    result["cache_name"] = cache_name

    # ── Step 2: Run all targeted extraction queries concurrently ──
    # We run them concurrently to minimize latency, but each is a separate
    # API call so the model can focus on one structured task at a time.
    logger.info(f"Running {len(_EXTRACTION_QUERIES)} targeted queries against cache '{cache_name}'")

    query_tasks = {
        q["key"]: query_cached_paper(cache_name, q["question"])
        for q in _EXTRACTION_QUERIES
    }

    # Gather all responses
    keys = list(query_tasks.keys())
    responses = await asyncio.gather(*query_tasks.values(), return_exceptions=True)
    raw_responses: dict[str, str] = {}
    for key, resp in zip(keys, responses):
        if isinstance(resp, Exception):
            logger.warning(f"Query '{key}' raised exception: {resp}")
            raw_responses[key] = ""
        else:
            raw_responses[key] = resp or ""

    # ── Step 3: Parse each response ──

    # abstract_and_conclusion
    meta = _safe_parse_json(raw_responses.get("abstract_and_conclusion", ""), {})
    if isinstance(meta, dict):
        result["title"] = meta.get("title") or title
        result["authors"] = meta.get("authors") or []
        result["year"] = meta.get("year")
        result["abstract"] = meta.get("abstract") or ""
        result["conclusion"] = meta.get("conclusion") or ""

    # methodology
    methodology = _safe_parse_json(raw_responses.get("methodology", ""), {})
    if isinstance(methodology, dict):
        result["methodology"] = methodology

    # key_claims
    key_claims = _safe_parse_json(raw_responses.get("key_claims", ""), [])
    if isinstance(key_claims, list):
        result["key_claims"] = key_claims

    # limitations
    limitations = _safe_parse_json(raw_responses.get("limitations", ""), [])
    if isinstance(limitations, list):
        result["limitations"] = limitations

    # figures_tables
    figures_tables = _safe_parse_json(raw_responses.get("figures_tables", ""), [])
    if isinstance(figures_tables, list):
        result["figures_tables"] = figures_tables

    # citations_in_text
    citations = _safe_parse_json(raw_responses.get("citations_in_text", ""), [])
    if isinstance(citations, list):
        result["citations_in_text"] = citations

    # ── Step 4: Build combined text for synthesis pipeline ──
    # This is what synthesis_agent.py and report_agent.py consume.
    # We build a rich structured text block from all extracted data.
    text_parts = []

    if result["abstract"]:
        text_parts.append(f"## Abstract\n{result['abstract']}")

    if result["methodology"]:
        m = result["methodology"]
        text_parts.append(
            f"## Methodology\n"
            f"Approach: {m.get('approach', '')}\n"
            f"Datasets: {', '.join(m.get('datasets', []))}\n"
            f"Hardware: {m.get('hardware', '')}\n"
            f"Evaluation: {m.get('evaluation_protocol', '')}\n"
            f"Baselines: {', '.join(m.get('baseline_comparisons', []))}"
        )

    if result["key_claims"]:
        claims_text = "\n".join(
            f"- [{c.get('section', '?')}, p.{c.get('page', '?')}] "
            f"{c.get('claim', '')} "
            f"(quote: \"{c.get('raw_quote', '')}\")"
            for c in result["key_claims"]
        )
        text_parts.append(f"## Key Claims & Results\n{claims_text}")

    if result["figures_tables"]:
        figs_text = "\n".join(
            f"- {f.get('label', '?')} [p.{f.get('page', '?')}]: "
            f"{f.get('caption', '')} — {f.get('key_finding', '')}"
            for f in result["figures_tables"]
        )
        text_parts.append(f"## Figures & Tables\n{figs_text}")

    if result["limitations"]:
        lims_text = "\n".join(
            f"- [{l.get('section', '?')}, p.{l.get('page', '?')}] "
            f"{l.get('text', '')} "
            f"(quote: \"{l.get('raw_quote', '')}\")"
            for l in result["limitations"]
        )
        text_parts.append(f"## Limitations\n{lims_text}")

    if result["conclusion"]:
        text_parts.append(f"## Conclusion\n{result['conclusion']}")

    result["text"] = "\n\n".join(text_parts)
    result["char_count"] = len(result["text"])

    # Determine status
    if result["text"] and len(result["text"]) > 200:
        result["status"] = "success"
    elif result["abstract"]:
        result["status"] = "partial"
    else:
        result["status"] = "error"

    logger.info(
        f"Cache extraction complete for '{result['title']}': "
        f"status={result['status']}, "
        f"{len(result['key_claims'])} claims, "
        f"{len(result['figures_tables'])} figures/tables, "
        f"{len(result['limitations'])} limitations, "
        f"{len(result['citations_in_text'])} in-text citations"
    )

    # NOTE: Cache is NOT deleted here — it is deleted by adk_pipeline.py
    # after synthesis and report generation are complete, so all agents
    # can query it if needed. Pass cache_name back in the result for cleanup.
    return result
