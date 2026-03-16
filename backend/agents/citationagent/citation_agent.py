"""
citation_agent.py — ADK Tool: Build citation graph with three-tier matching.

Strategy priority:
  A. citations_in_text  (context_cache provenance data — highest quality)
  B. LLM batch matching (Gemini reads the reference tail-chunk + corpus list)
  C. Heuristic matching (title-word overlap + author last-name + year signals)

The graph is stored in Firestore and returned as graph_data for the frontend
D3.js CitationGraph component.
"""

import logging
import os
from google.cloud import firestore

from core.graph_builder import generate_citation_graph

logger = logging.getLogger(__name__)


def build_citation_graph(extractions: list, query: str) -> dict:
    """
    ADK Tool: Build a citation graph from extracted papers.

    Delegates all graph construction to core/graph_builder.py which implements
    the three-tier citation detection strategy:
      A. citations_in_text from context_cache (structured provenance)
      B. LLM batch matching on the reference tail-chunk
      C. Robust heuristic (title-word overlap + author + year)

    Stores the resulting graph in Firestore and returns it for the frontend.

    Args:
        extractions: List of paper extraction dicts from extraction_agent.
                     Each dict should contain at minimum:
                       - title, authors, year, url
                       - text  (raw extracted text — used for B and C)
                       - citations_in_text (list — used for A, may be empty)
        query: The original research query string.

    Returns:
        dict with keys: status, graph_id, nodes (count), edges (count), graph_data.
    """
    if not extractions:
        logger.warning("build_citation_graph called with empty extractions list.")
        return {
            "status": "error",
            "message": "No extractions provided.",
            "graph_data": {"nodes": [], "edges": [], "node_count": 0, "edge_count": 0},
        }

    # Filter out hard-error extractions (keep partial successes — they still
    # have text and metadata that can contribute to the graph).
    usable = [
        e for e in extractions
        if isinstance(e, dict) and e.get("status") in ("success", "partial")
        or (isinstance(e, dict) and e.get("text"))
    ]

    if not usable:
        logger.warning("No usable extractions (all errored). Returning empty graph.")
        return {
            "status": "error",
            "message": "All extractions failed — cannot build graph.",
            "graph_data": {"nodes": [], "edges": [], "node_count": 0, "edge_count": 0},
        }

    logger.info(
        f"Building citation graph: {len(usable)}/{len(extractions)} usable extractions "
        f"for query='{query[:60]}'"
    )

    # ── Core graph construction (three-tier strategy) ──
    graph_data = generate_citation_graph(usable, query)

    # ── Persist to Firestore ──
    graph_id = _save_to_firestore(graph_data, query)

    paper_edge_count = sum(
        1 for e in graph_data["edges"]
        if e.get("source") != "topic_root" and e.get("target") != "topic_root"
    )

    logger.info(
        f"Citation graph stored: graph_id={graph_id}, "
        f"nodes={graph_data['node_count']}, "
        f"total_edges={graph_data['edge_count']}, "
        f"paper-to-paper_edges={paper_edge_count}"
    )

    return {
        "status": "success",
        "graph_id": graph_id,
        "nodes": graph_data["node_count"],
        "edges": graph_data["edge_count"],
        "paper_edges": paper_edge_count,
        "graph_data": graph_data,
    }


def _save_to_firestore(graph_data: dict, query: str) -> str:
    """
    Persist the graph to Firestore.
    Returns the document ID, or 'local' if Firestore is unavailable.
    """
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        logger.warning("GOOGLE_CLOUD_PROJECT_ID not set — skipping Firestore write.")
        return "local"

    try:
        db = firestore.Client(project=project_id)
        doc_ref = db.collection("citation_graphs").document()
        doc_ref.set({
            "query": query,
            **graph_data,
        })
        return doc_ref.id
    except Exception as e:
        logger.error(f"Firestore write failed: {e}")
        return "local"
