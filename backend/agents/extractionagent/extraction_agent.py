"""
extraction_agent.py — ADK Tool: In-memory PDF text extraction with provenance.

Extraction priority order:
  1. PyMuPDF fast-path  — httpx download → fitz text extraction (zero API cost)
  2. Context Cache path — RAM → GCS → Gemini Context Cache → targeted Q&A
                          (replaces the old 30-50 screenshot scroll loop)
                          Returns full provenance: page numbers, sections, raw quotes,
                          bounding boxes for figures/tables, in-text citations.
  3. Abstract-only fallback — uses snippet/abstract from search metadata.

The visual scrolling fallback (_visual_fallback) has been removed.
It required 30-50 Gemini API calls per paper and produced no provenance data.
The context cache path achieves better results in 6-8 targeted API calls.
"""

import asyncio
import logging
import os

import httpx
import fitz  # noqa: F401 — PyMuPDF

from core.pdf_processor import extract_multimodal_from_bytes
from core.analysis import analyze_figure, analyze_table
from core.gcs_handler import fetch_pdf_to_gcs, delete_gcs_object
from core.context_cache import extract_full_paper_via_cache
from config import settings

logger = logging.getLogger(__name__)

SAFE_DOMAINS = [
    "arxiv.org",
    "europepmc.org",
    "semanticscholar.org",
    "pubmed.ncbi.nlm.nih.gov",
    "core.ac.uk",
]


async def _download_pdf_bytes(url: str) -> bytes | None:
    """Download PDF bytes into RAM. Returns None if not a valid PDF."""
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code == 200 and b"%PDF" in resp.content[:1024]:
                return resp.content
    except Exception as e:
        logger.warning(f"Failed to download PDF from {url}: {e}")
    return None


async def _cache_based_extraction(url: str, title: str) -> dict | None:
    """
    Phase 2+3+4 of the new plan:
      RAM → GCS → Gemini Context Cache → targeted provenance Q&A → cleanup GCS

    Returns a full extraction dict with provenance fields, or None if the
    GCS upload fails (cache creation failure is handled inside context_cache.py
    and returns a partial result rather than None).

    The cache_name is kept alive in the result dict so adk_pipeline.py can
    delete it after synthesis + report generation are complete.
    """
    if not settings.ENABLE_CONTEXT_CACHING:
        logger.info(f"Context caching disabled — skipping cache path for '{title}'")
        return None

    if not settings.CLOUD_STORAGE_BUCKET:
        logger.warning("CLOUD_STORAGE_BUCKET not set — cannot use context cache path.")
        return None

    logger.info(f"Cache extraction path: uploading '{title}' to GCS...")

    # ── Phase 2: RAM → GCS ──
    gcs_uri = await fetch_pdf_to_gcs(url, settings.CLOUD_STORAGE_BUCKET)
    if not gcs_uri:
        logger.warning(f"GCS upload failed for '{title}' — falling back to abstract-only")
        return None

    # ── Phase 3+4: GCS → Context Cache → targeted Q&A ──
    result = await extract_full_paper_via_cache(gcs_uri=gcs_uri, title=title, url=url)

    # GCS object is deleted here immediately after cache creation.
    # The cache itself holds the document in Gemini's memory — GCS is no longer needed.
    # Cache deletion happens in adk_pipeline.py after all agents finish.
    await delete_gcs_object(gcs_uri)
    logger.info(f"GCS object deleted after cache creation: {gcs_uri}")

    if result.get("status") in ("success", "partial"):
        return result

    logger.warning(f"Cache extraction returned status='{result.get('status')}' for '{title}'")
    return None


async def _extract_single(paper: dict) -> dict:
    """
    Extract a single paper using the priority cascade:
      0. Hybrid-analyzed (already has text + vision insights — pass through)
      1. PyMuPDF (fast, free, no provenance)
      2. Context Cache (slower, costs tokens, full provenance)
      3. Abstract-only (always available)
    """
    url = paper.get("url")
    title = paper.get("title", "Unknown")
    abstract = paper.get("snippet", paper.get("abstract", ""))

    # ── Path 0: Hybrid-analyzed papers (already processed by paper_analyzer.py) ──
    # These papers come from the hybrid vision + in-memory analysis pipeline.
    # They already have extracted text, hierarchical summaries, and vision insights.
    # Pass them through without re-extracting.
    extraction_method = paper.get("extraction_method", "")
    if extraction_method in ("hybrid_analysis",) and paper.get("text") and len(paper.get("text", "")) > 200:
        logger.info(f"Hybrid-analyzed paper '{title[:40]}' — passing through (already extracted)")
        return {
            "url": url,
            "title": title,
            "name": title,
            "authors": paper.get("authors", []),
            "year": paper.get("year"),
            "abstract": paper.get("abstract", abstract),
            "text": paper.get("text", ""),
            "char_count": paper.get("char_count", len(paper.get("text", ""))),
            "status": paper.get("status", "success"),
            "extraction_method": "hybrid_analysis",
            "methodology": paper.get("methodology", {}),
            "key_claims": paper.get("key_claims", []),
            "limitations": paper.get("limitations", []),
            "figures_tables": paper.get("figures_tables", []),
            "citations_in_text": paper.get("citations_in_text", []),
            "citation_strings": paper.get("citation_strings", []),
            "section_map": paper.get("section_map", []),
            "hierarchical_summary": paper.get("hierarchical_summary", ""),
            "vision_insights": paper.get("vision_insights", []),
            "capture_metadata": paper.get("capture_metadata", {}),
            "cache_name": None,
            "gcs_uri": None,
        }

    result = {
        "url": url,
        "title": title,
        "name": title,           # backward compat
        "authors": paper.get("authors", []),
        "year": paper.get("year"),
        "abstract": abstract,
        "text": "",
        "char_count": 0,
        "status": "error",
        "extraction_method": "abstract_only",
        # Provenance fields (populated by context cache path)
        "methodology": {},
        "key_claims": [],
        "limitations": [],
        "figures_tables": [],
        "citations_in_text": [],
        "cache_name": None,      # kept for adk_pipeline.py cleanup
        "gcs_uri": None,
    }

    if not url:
        if abstract:
            result["text"] = abstract
            result["char_count"] = len(abstract)
            result["status"] = "success"
        return result

    # ── Path 1: PyMuPDF fast-path ──
    pdf_bytes = await _download_pdf_bytes(url)
    if pdf_bytes:
        try:
            data = await asyncio.to_thread(extract_multimodal_from_bytes, pdf_bytes)
            text = data.get("text", "")
            figures = data.get("figures", [])
            tables = data.get("tables", [])
            del pdf_bytes

            if len(text.strip()) > 200:
                result["text"] = text[:40000]

                # Analyze figures and tables (limit to top 3 each)
                visual_insights = []

                figure_tasks = [
                    analyze_figure(fig["base64"], context=f"Paper: {title}")
                    for fig in figures[:3]
                ]
                if figure_tasks:
                    fig_results = await asyncio.gather(*figure_tasks)
                    for i, res in enumerate(fig_results):
                        visual_insights.append(
                            f"Figure {figures[i]['page']}.{figures[i]['index']}: {res}"
                        )

                table_tasks = [
                    analyze_table(tab["data"], context=f"Paper: {title}")
                    for tab in tables[:3]
                ]
                if table_tasks:
                    tab_results = await asyncio.gather(*table_tasks)
                    for i, res in enumerate(tab_results):
                        visual_insights.append(
                            f"Table {tables[i]['page']}.{tables[i]['index']}: {res}"
                        )

                if visual_insights:
                    result["text"] += (
                        "\n\n### VISUAL & TABULAR INSIGHTS ###\n"
                        + "\n".join(visual_insights)
                    )

                result["char_count"] = len(result["text"])
                result["status"] = "success"
                result["extraction_method"] = "pdf_multimodal"
                return result

        except Exception as e:
            logger.warning(f"PyMuPDF extraction error for {url}: {e}")

    # ── Path 2: Context Cache (replaces visual scrolling fallback) ──
    # Only attempt for safe/open-access domains to avoid wasting GCS quota
    # on paywalled sites that will return HTML error pages instead of PDFs.
    is_safe = any(domain in (url or "") for domain in SAFE_DOMAINS)
    if is_safe or os.getenv("CACHE_ALL_DOMAINS", "false").lower() == "true":
        cache_result = await _cache_based_extraction(url, title)
        if cache_result and cache_result.get("status") in ("success", "partial"):
            # Merge provenance fields into result
            result.update({
                "text": cache_result.get("text", ""),
                "char_count": len(cache_result.get("text", "")),
                "status": cache_result["status"],
                "extraction_method": "context_cache",
                "title": cache_result.get("title") or title,
                "authors": cache_result.get("authors") or result["authors"],
                "year": cache_result.get("year") or result["year"],
                "abstract": cache_result.get("abstract") or abstract,
                "methodology": cache_result.get("methodology", {}),
                "key_claims": cache_result.get("key_claims", []),
                "limitations": cache_result.get("limitations", []),
                "figures_tables": cache_result.get("figures_tables", []),
                "citations_in_text": cache_result.get("citations_in_text", []),
                "cache_name": cache_result.get("cache_name"),
                "gcs_uri": cache_result.get("gcs_uri"),
            })
            return result
    else:
        logger.info(f"Skipping cache path for non-safe domain: {url}")

    # ── Path 3: Abstract-only fallback ──
    if abstract:
        result["text"] = abstract
        result["char_count"] = len(abstract)
        result["status"] = "success"
        result["extraction_method"] = "abstract_only"

    return result


async def _extract_in_batches(papers: list) -> dict:
    """Extract papers in small concurrent batches to respect API rate limits."""
    batch_size = int(os.getenv("BATCH_SIZE", "3"))
    all_extractions = []

    for i in range(0, len(papers), batch_size):
        batch = papers[i : i + batch_size]
        logger.info(
            f"Extracting batch {i // batch_size + 1} "
            f"({len(batch)} papers, {i + len(batch)}/{len(papers)} total)"
        )

        tasks = [_extract_single(p) for p in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, dict):
                all_extractions.append(r)
            else:
                logger.error(f"Extraction task failed in batch: {r}")

        # Brief pause between batches to avoid rate-limiting GCS/Gemini APIs
        if i + batch_size < len(papers):
            await asyncio.sleep(1)

    success_count = sum(1 for e in all_extractions if e.get("status") in ("success", "partial"))
    cache_count = sum(1 for e in all_extractions if e.get("extraction_method") == "context_cache")
    pymupdf_count = sum(1 for e in all_extractions if e.get("extraction_method") == "pdf_multimodal")

    logger.info(
        f"Extraction complete: {success_count}/{len(papers)} succeeded "
        f"({pymupdf_count} PyMuPDF, {cache_count} context cache)"
    )

    return {
        "status": "success",
        "papers_extracted": success_count,
        "extractions": all_extractions,
    }


def extract_papers(papers: list = None, urls: list = None) -> dict:
    """
    ADK Tool: Extract text and provenance data from academic papers.

    Priority cascade per paper:
      1. PyMuPDF (fast, free, no provenance)
      2. Gemini Context Cache via GCS (full provenance: page, section, raw quote, bounding box)
      3. Abstract-only fallback

    Safe to call from both sync and async contexts.
    """
    if papers is None:
        papers = [{"url": u} for u in (urls or [])]

    if not papers:
        return {"status": "error", "message": "No papers provided"}

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(_extract_in_batches(papers), loop)
            return future.result(timeout=600)
        else:
            return loop.run_until_complete(_extract_in_batches(papers))
    except RuntimeError:
        return asyncio.run(_extract_in_batches(papers))
    except Exception as e:
        return {"status": "error", "message": f"Extraction failed: {str(e)}"}
