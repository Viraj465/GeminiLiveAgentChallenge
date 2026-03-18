"""
adk_pipeline.py — Runs the full ADK research pipeline after autopilot browsing.

This bridges the vision loop (which discovers papers via browser) with the
ADK sub-agents (which process, synthesize, and report on them).

Pipeline flow (matching Diagram 4):
  1. SearchAgent — enriches discovered papers with API data (Semantic Scholar,
     arXiv, Europe PMC, OpenAlex, Crossref, CORE) and reranks by relevance.
  2. ExtractionAgent — downloads PDFs, extracts text + figures + tables.
  3. SynthesisAgent — cross-paper analysis using Gemini 2.5 Pro (1M context).
  4. CitationAgent — builds citation graph from extracted texts.
  5. ReportAgent — generates publication-ready literature review.

Each step streams status updates to the frontend via WebSocket.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import WebSocket

logger = logging.getLogger(__name__)


def _ws_is_open(websocket: WebSocket) -> bool:
    """Check if the WebSocket connection is still open."""
    try:
        return websocket.client_state.value == 1  # CONNECTED=1
    except Exception:
        return False


async def _send_pipeline_status(websocket: WebSocket, stage: str, message: str, progress: float = 0.0):
    """Send a pipeline progress update to the frontend."""
    if not _ws_is_open(websocket):
        return
    try:
        await websocket.send_json({
            "type": "log_update",
            "payload": {
                "log": f"[Pipeline: {stage}] {message}",
                "pipeline_stage": stage,
                "pipeline_progress": progress,
                "timestamp": time.time(),
            },
        })
    except Exception as e:
        logger.debug(f"Failed to send pipeline status: {e}")


@asynccontextmanager
async def _stage_keepalive(websocket: WebSocket, stage: str, interval: float = 25.0):
    """
    Context manager that sends a lightweight heartbeat every `interval` seconds
    while a slow pipeline stage (synthesis, report generation) is running.

    This prevents Cloud Run's 300 s request timeout from killing the container
    during heavy Gemini API calls that can take 2-5 minutes per stage.

    Usage:
        async with _stage_keepalive(websocket, "SynthesisAgent"):
            result = await slow_gemini_call(...)
    """
    _running = True
    _elapsed = 0

    async def _ping():
        nonlocal _elapsed
        try:
            while _running:
                await asyncio.sleep(interval)
                if not _running:
                    break
                _elapsed += int(interval)
                if _ws_is_open(websocket):
                    try:
                        await websocket.send_json({
                            "type": "pipeline_heartbeat",
                            "payload": {
                                "message": f"[{stage}] Still working… ({_elapsed}s)",
                                "stage": stage,
                                "elapsed_seconds": _elapsed,
                                "timestamp": time.time(),
                            },
                        })
                        logger.debug(f"Stage keepalive [{stage}] sent ({_elapsed}s)")
                    except Exception:
                        break  # WebSocket closed — stop pinging
        except asyncio.CancelledError:
            pass

    ping_task = asyncio.create_task(_ping())
    try:
        yield
    finally:
        _running = False
        ping_task.cancel()
        try:
            await asyncio.wait_for(ping_task, timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass


async def run_adk_pipeline(
    task: str,
    discovered_papers: list[dict],
    session_id: str,
    websocket: WebSocket,
) -> dict:
    """
    Run the full ADK research pipeline.

    Args:
        task: The original research query.
        discovered_papers: Papers found during autopilot browsing.
        session_id: Session ID for Firestore persistence.
        websocket: WebSocket for streaming status updates.

    Returns:
        dict with graph_data and report_markdown.
    """
    result = {
        "graph_data": None,
        "report_markdown": None,
        "papers_found": 0,
        "papers_extracted": 0,
        "status": "error",
    }

    try:
        
        # Stage 1: SearchAgent — Enrich with API data
        
        await _send_pipeline_status(websocket, "SearchAgent", "Phase 1-5 Deep Search: Planning sub-queries across 9 source tiers (arXiv, Semantic Scholar, Europe PMC, OpenAlex, Crossref, CORE, BASE, DOAJ, Google CSE)...", 0.1)

        from agents.searchagent.search_agent import search_papers_async

        # Run the 5-phase deep search
        search_result = await search_papers_async(task, max_papers=20)

        api_papers = search_result.get("papers", [])
        sources = search_result.get("sources_searched", [])
        subqueries = search_result.get("subqueries_used", [])

        # Merge vision-discovered papers with API-found papers
        # Vision-discovered papers get priority (they came from the user's browsing)
        all_papers = list(discovered_papers)  # Start with browsing discoveries
        seen_titles = {p.get("title", "").lower().strip()[:80] for p in all_papers}

        for p in api_papers:
            key = p.get("title", "").lower().strip()[:80]
            if key and key not in seen_titles:
                seen_titles.add(key)
                all_papers.append(p)

        # Limit to top 15 papers total
        all_papers = all_papers[:15]
        result["papers_found"] = len(all_papers)

        await _send_pipeline_status(
            websocket, "SearchAgent",
            f"Found {len(all_papers)} papers ({len(discovered_papers)} from browsing, "
            f"{len(api_papers)} from APIs: {', '.join(sources)})",
            0.2,
        )

        if not all_papers:
            await _send_pipeline_status(websocket, "SearchAgent", "No papers found. Pipeline cannot continue.", 1.0)
            result["status"] = "no_papers"
            return result

        
        # Stage 2: ExtractionAgent — Download & extract PDFs
        
        await _send_pipeline_status(
            websocket, "ExtractionAgent",
            f"Downloading and extracting {len(all_papers)} papers (PyMuPDF → Context Cache cascade)...",
            0.3,
        )

        # Notify frontend that GCS + context caching may be used
        if _ws_is_open(websocket):
            try:
                await websocket.send_json({
                    "type": "gcs_status",
                    "payload": {
                        "status": "uploading_to_gcs",
                        "message": "Streaming PDFs to Google Cloud Storage for context caching...",
                        "timestamp": time.time(),
                    },
                })
            except Exception:
                pass

        from agents.extractionagent.extraction_agent import _extract_in_batches

        extraction_result = await _extract_in_batches(all_papers)
        extractions = extraction_result.get("extractions", [])
        extracted_count = extraction_result.get("papers_extracted", 0)
        result["papers_extracted"] = extracted_count

        # Count how many used each extraction method
        cache_count = sum(1 for e in extractions if e.get("extraction_method") == "context_cache")
        pymupdf_count = sum(1 for e in extractions if e.get("extraction_method") == "pdf_multimodal")
        hybrid_count = sum(1 for e in extractions if e.get("extraction_method") == "hybrid_analysis")

        if _ws_is_open(websocket):
            try:
                method_parts = []
                if pymupdf_count:
                    method_parts.append(f"{pymupdf_count} via PyMuPDF")
                if cache_count:
                    method_parts.append(f"{cache_count} via Gemini Context Cache")
                if hybrid_count:
                    method_parts.append(f"{hybrid_count} via Hybrid Analysis (PDF + Vision)")
                method_summary = ", ".join(method_parts) if method_parts else "abstract-only fallback"

                await websocket.send_json({
                    "type": "cache_status",
                    "payload": {
                        "status": "cache_ready",
                        "message": f"Extraction complete: {method_summary}.",
                        "cache_count": cache_count,
                        "pymupdf_count": pymupdf_count,
                        "hybrid_count": hybrid_count,
                        "timestamp": time.time(),
                    },
                })
            except Exception:
                pass

        await _send_pipeline_status(
            websocket, "ExtractionAgent",
            f"Extracted {extracted_count}/{len(all_papers)} papers "
            f"({pymupdf_count} PyMuPDF, {cache_count} context cache, {hybrid_count} hybrid analysis).",
            0.5,
        )

        if extracted_count == 0:
            # Fallback: use snippets/abstracts as minimal text
            logger.warning("No papers had extractable text. Using abstracts as fallback.")
            extractions = []
            for p in all_papers:
                snippet = p.get("snippet", p.get("abstract", ""))
                if snippet:
                    extractions.append({
                        "title": p.get("title", "Unknown"),
                        "name": p.get("title", "Unknown"),
                        "text": snippet,
                        "authors": p.get("authors", []),
                        "year": p.get("year"),
                        "char_count": len(snippet),
                        "status": "success",
                        "extraction_method": "abstract_only",
                    })
            extracted_count = len(extractions)

        
        # Stage 3: SynthesisAgent — Cross-paper analysis
        
        await _send_pipeline_status(
            websocket, "SynthesisAgent",
            f"Analyzing {extracted_count} papers with Gemini 2.5 Pro...",
            0.6,
        )

        from agents.synthesisagent.synthesis_agent import synthesize_findings

        # Wrap in keepalive — synthesis can take 2-4 min on large corpora
        async with _stage_keepalive(websocket, "SynthesisAgent"):
            synthesis_result = await asyncio.to_thread(
                synthesize_findings, extractions, task
            )
        synthesis_text = synthesis_result.get("synthesis", "")

        await _send_pipeline_status(
            websocket, "SynthesisAgent",
            f"Synthesis complete ({synthesis_result.get('papers_analyzed', 0)} papers analyzed).",
            0.7,
        )

        
        # Stage 4: CitationAgent — Build citation graph
        
        await _send_pipeline_status(
            websocket, "CitationAgent",
            "Building citation network from extracted texts...",
            0.8,
        )

        # Use the lightweight graph_builder for D3 visualization
        # (The ADK citation_agent needs Firestore; graph_builder works standalone)
        from core.graph_builder import generate_citation_graph

        # Format papers for graph_builder — pass ALL provenance fields so
        # nodes get methodology, key_claims, limitations, figures_tables,
        # citations_in_text, year, url, and extraction_method.
        graph_papers = []
        for ext in extractions:
            graph_papers.append({
                "title": ext.get("title", ext.get("name", "Unknown")),
                "authors": ext.get("authors", []),
                "year": ext.get("year"),
                "url": ext.get("url", ""),
                "text": ext.get("text", ""),
                "abstract": ext.get("abstract", ""),
                "snippet": ext.get("abstract", ""),
                "extraction_method": ext.get("extraction_method", "unknown"),
                "char_count": ext.get("char_count", 0),
                # Provenance fields from context_cache path
                "methodology": ext.get("methodology", {}),
                "key_claims": ext.get("key_claims", []),
                "limitations": ext.get("limitations", []),
                "figures_tables": ext.get("figures_tables", []),
                "citations_in_text": ext.get("citations_in_text", []),
            })

        # Also add papers that had no extracted text but were found
        for p in all_papers:
            title = p.get("title", "")
            if title and not any(gp.get("title") == title for gp in graph_papers):
                graph_papers.append({
                    "title": title,
                    "authors": p.get("authors", []),
                    "year": p.get("year"),
                    "url": p.get("url", ""),
                    "text": "",
                    "abstract": p.get("snippet", p.get("abstract", "")),
                    "snippet": p.get("snippet", p.get("abstract", "")),
                    "extraction_method": "not_extracted",
                    "methodology": {},
                    "key_claims": [],
                    "limitations": [],
                    "figures_tables": [],
                    "citations_in_text": [],
                })

        graph_data = generate_citation_graph(graph_papers, topic=task[:50])
        result["graph_data"] = graph_data

        await _send_pipeline_status(
            websocket, "CitationAgent",
            f"Citation network: {graph_data.get('node_count', 0)} nodes, {graph_data.get('edge_count', 0)} edges.",
            0.85,
        )

        
        # Stage 5: ReportAgent — Generate literature review
        
        await _send_pipeline_status(
            websocket, "ReportAgent",
            "Generating publication-ready literature review...",
            0.9,
        )

        # Use the synthesis.py module which handles Gemini 2.5 Pro async
        from core.synthesis import generate_literature_review

        # Build extracted_texts dict for the review generator
        extracted_texts = {}
        for ext in extractions:
            title = ext.get("title", ext.get("name", "Unknown"))
            text = ext.get("text", "")
            if text and len(text) > 50:
                extracted_texts[title] = text

        if not extracted_texts:
            # Fallback: use synthesis text itself
            extracted_texts["Synthesis Summary"] = synthesis_text

        # Wrap in keepalive — report generation can take 3-6 min for large reviews
        async with _stage_keepalive(websocket, "ReportAgent"):
            report_markdown = await generate_literature_review(
                topic=task,
                extracted_texts=extracted_texts,
                synthesis=synthesis_text,
                graph=graph_data,
            )
        result["report_markdown"] = report_markdown

        await _send_pipeline_status(
            websocket, "ReportAgent",
            f"Literature review generated ({len(report_markdown.split())} words).",
            1.0,
        )

        
        # Phase 5: Cleanup — delete all Gemini context caches
        # Context caches are billed per second while alive.
        # Delete them immediately after the report is generated.
        
        cache_names_to_delete = [
            e.get("cache_name")
            for e in extractions
            if e.get("cache_name")
        ]
        if cache_names_to_delete:
            await _send_pipeline_status(
                websocket, "Cleanup",
                f"Deleting {len(cache_names_to_delete)} Gemini context cache(s)...",
                1.0,
            )
            from core.context_cache import delete_paper_cache
            cleanup_tasks = [delete_paper_cache(name) for name in cache_names_to_delete]
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

            if _ws_is_open(websocket):
                try:
                    await websocket.send_json({
                        "type": "cache_status",
                        "payload": {
                            "status": "cache_deleted",
                            "message": f"Cleaned up {len(cache_names_to_delete)} context cache(s). Billing stopped.",
                            "timestamp": time.time(),
                        },
                    })
                except Exception:
                    pass

            logger.info(f"Cleanup complete: deleted {len(cache_names_to_delete)} context cache(s).")

        # ── Persist results to Firestore (always runs, even if WebSocket is closed) ──
        # This ensures results survive client disconnects and are available on reconnect.
        try:
            from core.db import save_session_data
            await asyncio.gather(
                save_session_data(session_id, "query", task),
                save_session_data(session_id, "graph_data", graph_data),
                save_session_data(session_id, "report_markdown", report_markdown),
                save_session_data(session_id, "papers_found", len(all_papers)),
                save_session_data(session_id, "papers_extracted", extracted_count),
                return_exceptions=True,
            )
            logger.info(f"Session {session_id}: results persisted to Firestore.")
        except Exception as e:
            logger.warning(f"Firestore save failed (non-fatal): {e}")

        result["status"] = "success"
        return result

    except Exception as e:
        logger.error(f"ADK pipeline error: {e}", exc_info=True)
        await _send_pipeline_status(websocket, "Error", f"Pipeline failed: {e}", 1.0)
        result["status"] = "error"
        result["error"] = str(e)
        return result
