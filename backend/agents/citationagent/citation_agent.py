"""
citation_agent.py — ADK Tool: Build citation graph with improved heuristic matching.

Uses fuzzy string matching + author/year signals to detect citations.
Zero API cost — all matching is local.
Stores graph in Firestore for D3.js visualization.
"""

import logging
import os
import re
from difflib import SequenceMatcher
from google.cloud import firestore

logger = logging.getLogger(__name__)


def build_citation_graph(extractions: list, query: str) -> dict:
    """
    ADK Tool: Extracts citation relationships using improved heuristic matching.
    Stores the graph in Firestore for frontend D3.js visualization.

    Args:
        extractions: List of paper extraction dicts from extraction_agent.
        query: The original research query.

    Returns:
        dict with status, graph_id, node/edge counts, and graph data.
    """
    db = firestore.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT"))

    nodes = []
    edges = []

    # ── Build node list with metadata ──
    paper_meta = []
    for i, paper in enumerate(extractions):
        if "error" in paper:
            continue

        text = paper.get("text", "")
        title = _extract_title(text)
        authors = _extract_authors(text)
        year = _extract_year(text)

        paper_id = f"paper_{i}"
        node = {
            "id": paper_id,
            "label": title or f"Paper {i + 1}",
            "authors": authors,
            "year": year,
            "name": paper.get("name", ""),
            "char_count": paper.get("char_count", 0),
            "images_found": paper.get("images_found", 0),
        }
        nodes.append(node)
        paper_meta.append({
            "index": i,
            "id": paper_id,
            "title": title,
            "authors": authors,
            "year": year,
            "text": text,
        })

    # ── Find citation edges using improved heuristics ──
    for meta in paper_meta:
        refs_section = _extract_references_section(meta["text"])
        text_to_search = refs_section if refs_section else meta["text"]

        for other_meta in paper_meta:
            if meta["index"] == other_meta["index"]:
                continue

            confidence = _calculate_citation_confidence(
                text_to_search,
                other_meta["title"],
                other_meta["authors"],
                other_meta["year"],
            )

            if confidence >= 0.4:  # Threshold for citation edge
                edges.append({
                    "source": meta["id"],
                    "target": other_meta["id"],
                    "type": "cites",
                    "confidence": round(confidence, 3),
                })

    # ── Store in Firestore ──
    graph_data = {
        "query": query,
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
    }

    doc_ref = db.collection("citation_graphs").document()
    doc_ref.set(graph_data)

    return {
        "status": "success",
        "graph_id": doc_ref.id,
        "nodes": len(nodes),
        "edges": len(edges),
        "graph_data": graph_data,
    }


def _extract_references_section(text: str) -> str:
    """Extract the references/bibliography section from paper text."""
    # Look for common section headers
    patterns = [
        r"(?i)\n\s*references?\s*\n",
        r"(?i)\n\s*bibliography\s*\n",
        r"(?i)\n\s*works?\s+cited\s*\n",
        r"(?i)\n\s*literature\s+cited\s*\n",
    ]

    best_pos = -1
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            pos = match.start()
            if best_pos == -1 or pos > best_pos:
                best_pos = pos

    if best_pos > 0:
        return text[best_pos:]

    return ""


def _calculate_citation_confidence(
    search_text: str,
    target_title: str,
    target_authors: list,
    target_year: str,
) -> float:
    """
    Calculate confidence score (0-1) that search_text cites the target paper.
    Uses fuzzy title matching + author/year signals.
    """
    if not target_title or len(target_title) < 10:
        return 0.0

    score = 0.0
    search_lower = search_text.lower()
    title_lower = target_title.lower()

    # ── Signal 1: Fuzzy title match (weight: 0.6) ──
    # Check if title (or significant substring) appears in text
    if title_lower in search_lower:
        score += 0.6
    else:
        # Try fuzzy matching on chunks of the reference section
        best_ratio = 0.0
        # Slide a window of title length across the text
        title_len = len(title_lower)
        # Only search in reasonable chunks to avoid O(n²)
        search_sample = search_lower[:30000]
        for start in range(0, len(search_sample) - title_len, title_len // 3 + 1):
            chunk = search_sample[start:start + title_len + 20]
            ratio = SequenceMatcher(None, title_lower, chunk).ratio()
            if ratio > best_ratio:
                best_ratio = ratio

        if best_ratio > 0.75:
            score += 0.6 * best_ratio

    # ── Signal 2: Author name match (weight: 0.25) ──
    if target_authors:
        authors_found = 0
        for author in target_authors[:3]:  # Check first 3 authors
            # Extract last name
            parts = author.strip().split()
            if parts:
                last_name = parts[-1].lower()
                if len(last_name) > 2 and last_name in search_lower:
                    authors_found += 1

        if target_authors:
            author_ratio = authors_found / min(len(target_authors), 3)
            score += 0.25 * author_ratio

    # ── Signal 3: Year match (weight: 0.15) ──
    if target_year and str(target_year) in search_text:
        score += 0.15

    return min(score, 1.0)


def _extract_title(text: str) -> str:
    """Extract likely title from the first 500 chars of paper text."""
    if not text:
        return ""
    first_lines = text[:500].split("\n")
    for line in first_lines:
        line = line.strip()
        if 10 < len(line) < 200:
            return line
    return ""


def _extract_authors(text: str) -> list:
    """Extract likely author names from the header area of a paper."""
    if not text:
        return []

    # Look in first 2000 chars (title + author block)
    header = text[:2000]
    authors = []

    # Common patterns: "Name1, Name2, and Name3" or "Name1 · Name2"
    # Look for lines with multiple capitalized words separated by commas
    lines = header.split("\n")
    for line in lines[1:10]:  # Skip title (first line), check next 9
        line = line.strip()
        if not line or len(line) < 5:
            continue

        # Skip lines that look like abstracts or section headers
        if any(kw in line.lower() for kw in ["abstract", "introduction", "keywords", "doi"]):
            continue

        # Check if line looks like author names (multiple capitalized words)
        words = line.split()
        cap_words = sum(1 for w in words if w[0].isupper() and len(w) > 1)
        if cap_words >= 2 and len(words) <= 20:
            # Split by common separators
            for sep in [",", "·", ";", " and "]:
                if sep in line:
                    parts = [p.strip() for p in line.split(sep) if p.strip()]
                    if 2 <= len(parts) <= 10:
                        authors = parts[:5]
                        break
            if authors:
                break

    return authors


def _extract_year(text: str) -> str:
    """Extract publication year from paper text."""
    if not text:
        return ""

    # Look in first 3000 chars for 4-digit year
    header = text[:3000]
    years = re.findall(r'\b(19[89]\d|20[0-2]\d)\b', header)

    if years:
        # Return the most common year, or the first one
        from collections import Counter
        year_counts = Counter(years)
        return year_counts.most_common(1)[0][0]

    return ""