"""
graph_builder.py — Provenance-enriched citation and idea network generator.

Transforms extracted papers into a D3-compatible JSON graph where:
  - Nodes carry full provenance: methodology, key claims with page+section+raw_quote,
    limitations, figures/tables with bounding boxes, year, URL.
  - Edges carry citation context: the exact sentence that contains the citation,
    the section and page it appears on, and the relationship type.

Citation edges are derived from three strategies (in priority order):
  1. citations_in_text field (from context_cache extraction) — highest quality.
  2. LLM-batch matching — pass the last 20% of raw text + all paper titles to
     Gemini and ask it to return a structured list of which papers are cited.
  3. Robust tail-chunk heuristic — isolate the last 20% of the paper text
     unconditionally (bypasses fragile "References" header regex), then use
     multi-signal fuzzy matching (title words + author last-names + year).
"""

import logging
import os
import re
import json

logger = logging.getLogger(__name__)


def _clean_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _truncate(text: str, n: int = 220) -> str:
    t = _clean_text(text)
    if not t:
        return ""
    return t[:n] + ("..." if len(t) > n else "")


def _fallback_problem_statement(abstract: str, text: str) -> str:
    candidate = _truncate(abstract, 260)
    if candidate:
        return candidate
    return _truncate(text, 260)


def _fallback_key_finding(abstract: str, text: str, extraction_method: str) -> str:
    candidate = _truncate(abstract, 200)
    if candidate:
        return candidate
    candidate = _truncate(text, 200)
    if candidate:
        return candidate
    if extraction_method:
        return f"Detailed findings unavailable (extraction: {extraction_method})."
    return "Detailed findings unavailable."


# ─── Robust reference-block extraction ──────────────────────────────────────

def _get_tail_chunk(text: str, fraction: float = 0.20) -> str:
    """
    Return the last `fraction` of the paper text unconditionally.
    This is where references almost always live, regardless of whether the
    "References" header is cleanly formatted.
    Falls back to the full text if the paper is very short.
    """
    if not text:
        return ""
    cutoff = max(0, int(len(text) * (1.0 - fraction)))
    return text[cutoff:]


def _get_references_section(text: str) -> str:
    """
    Try to find a cleanly-labelled references section first.
    If not found, fall back to the last 20% of the text.
    This fixes the original fragile regex that required perfect newline formatting.
    """
    if not text:
        return ""

    # Broad pattern — tolerates missing newlines, page numbers merged into header,
    # all-caps headers, and Unicode dashes.
    patterns = [
        r"(?:^|\n|\r)\s*(?:references?|bibliography|works?\s+cited|literature\s+cited"
        r"|cited\s+works?|citations?)\s*(?:\n|\r|$)",
        # All-caps variant
        r"(?:^|\n|\r)\s*(?:REFERENCES?|BIBLIOGRAPHY|WORKS?\s+CITED)\s*(?:\n|\r|$)",
        # Numbered section header like "9. References" or "10 References"
        r"(?:^|\n|\r)\s*\d+\.?\s+(?:references?|bibliography)\s*(?:\n|\r|$)",
    ]

    best_pos = -1
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            pos = m.start()
            if pos > best_pos:
                best_pos = pos

    if best_pos > 0 and best_pos > len(text) * 0.4:
        # Only trust the header if it appears in the latter half of the paper
        return text[best_pos:]

    # Fallback: last 20% of text
    return _get_tail_chunk(text, fraction=0.20)


# ─── Individual reference entry parser ──────────────────────────────────────

_REF_ENTRY_RE = re.compile(
    r"(?:^\s*\[?\d+\]?\.?\s+|\n\s*\[?\d+\]?\.?\s+)"
    r"(.+?)(?=\n\s*\[?\d+\]?\.?\s+|\Z)",
    re.DOTALL,
)


def _parse_ref_entries(ref_block: str) -> list[str]:
    """Split a reference block into individual reference strings."""
    entries = _REF_ENTRY_RE.findall(ref_block[:12000])
    cleaned = [" ".join(e.split()) for e in entries if e.strip()]
    if cleaned:
        return cleaned
    # Fallback: split by newlines and return non-empty lines
    lines = [l.strip() for l in ref_block.split("\n") if len(l.strip()) > 20]
    return lines[:80]


# ─── Heuristic matching (Strategy C) ────────────────────────────────────────

def _title_word_overlap(title: str, ref_text: str) -> float:
    """
    Compute the fraction of significant title words (len > 3) that appear
    in the reference text.  Returns 0.0–1.0.
    """
    if not title:
        return 0.0
    words = [w.lower() for w in re.split(r"\W+", title) if len(w) > 3]
    if not words:
        return 0.0
    ref_lower = ref_text.lower()
    hits = sum(1 for w in words if w in ref_lower)
    return hits / len(words)


def _find_cited_paper_heuristic(
    ref_text: str,
    paper_titles: list[str],
    paper_authors: list[list],
    paper_years: list[str],
) -> int | None:
    """
    Multi-signal heuristic matcher.
    Returns the index of the best-matching paper, or None.

    Scoring:
      - Title word overlap ≥ 0.40  → moderate title signal (0.45 pts)
      - Title word overlap ≥ 0.60  → strong title signal   (0.55 pts)
      - Exact title substring       → full title signal     (0.70 pts)
      - Author last-name match      → +0.25 pts (max 0.25)
      - Year match                  → +0.15 pts
    Threshold: 0.40  (lowered from 0.50 to catch more matches)
    """
    ref_lower = ref_text.lower()
    best_score = 0.0
    best_idx = None

    for idx, (title, authors, year) in enumerate(
        zip(paper_titles, paper_authors, paper_years)
    ):
        if not title:
            continue

        score = 0.0
        title_lower = title.lower()

        # Exact substring — highest confidence
        if title_lower in ref_lower:
            score += 0.70
        else:
            overlap = _title_word_overlap(title, ref_text)
            if overlap >= 0.60:
                score += 0.55 * overlap
            elif overlap >= 0.40:
                score += 0.45 * overlap

        # Author last-name signal (works even when authors list is empty)
        authors_found = 0
        for author in (authors or [])[:3]:
            parts = str(author).strip().split()
            if parts:
                last = parts[-1].lower()
                if len(last) > 2 and last in ref_lower:
                    authors_found += 1
        if authors_found:
            score += 0.25 * (authors_found / max(min(len(authors or [1]), 3), 1))

        # Year signal
        if year and str(year) in ref_text:
            score += 0.15

        # Title-only path: if no author data, lower threshold for title overlap
        if not (authors or []) and score == 0.0:
            overlap = _title_word_overlap(title, ref_text)
            if overlap >= 0.50:
                score += 0.50 * overlap

        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx if best_score >= 0.40 else None


# ─── LLM-batch matching (Strategy B) ────────────────────────────────────────

def _llm_match_citations(
    ref_block: str,
    paper_titles: list[str],
    paper_authors: list[list],
    paper_years: list[str],
    source_title: str = "",
) -> list[int]:
    """
    Ask Gemini to identify which papers from our corpus appear in the
    reference block.  Returns a list of matched paper indices.

    Falls back to an empty list if the API call fails.
    """
    try:
        from google import genai
        from config import settings

        client = genai.Client(api_key=settings.GEMINI_API_KEY)

        # Build a compact numbered list of our corpus papers
        corpus_lines = []
        for i, (t, a, y) in enumerate(zip(paper_titles, paper_authors, paper_years)):
            authors_str = ", ".join((a or [])[:2]) if a else "Unknown"
            corpus_lines.append(f"[{i}] {t} — {authors_str} ({y or 'n/a'})")
        corpus_text = "\n".join(corpus_lines)

        # Truncate ref_block to stay within token budget
        ref_sample = ref_block[:6000]

        prompt = f"""You are a citation matching assistant for academic literature.

Below is the REFERENCE SECTION (or tail text) extracted from a paper titled: "{source_title}"

REFERENCE SECTION:
{ref_sample}

---

Below is a numbered list of papers from our research corpus:

CORPUS:
{corpus_text}

---

TASK: Identify which papers from the CORPUS appear to be cited or referenced in the REFERENCE SECTION above.

MATCHING RULES (be LIBERAL — err on the side of inclusion):
- Match on ANY of: partial title words, author last names, publication year, or topic similarity
- A paper is a match if 2 or more significant title words appear anywhere in the reference section
- A paper is a match if an author's last name AND year both appear near each other in the reference section
- Do NOT require an exact or complete title match
- If a paper's topic is clearly related and its author or year appears, include it

Return ONLY a JSON array of the integer indices (from the CORPUS list) that are cited or likely cited.
Example output: [0, 3, 7]
If none are found, return: []
Return ONLY the JSON array, no explanation, no markdown."""

        response = client.models.generate_content(
            model=settings.GOOGLE_REASONING_MODEL,
            contents=prompt,
        )
        raw = response.text.strip()

        # Extract JSON array from response
        match = re.search(r"\[[\d,\s]*\]", raw)
        if match:
            indices = json.loads(match.group(0))
            # Validate indices are in range
            valid = [i for i in indices if isinstance(i, int) and 0 <= i < len(paper_titles)]
            logger.info(
                f"LLM citation matching for '{source_title[:40]}': "
                f"found {len(valid)} matches → {valid}"
            )
            return valid

    except Exception as e:
        logger.warning(f"LLM citation matching failed: {e}")

    return []


# ─── cited_work string → paper index (for Strategy A) ───────────────────────

def _match_cited_work_to_paper(
    cited_work_str: str,
    paper_titles: list[str],
    paper_authors: list[list],
    paper_years: list[str],
) -> int | None:
    """
    Match a cited_work string (e.g. "Smith et al. (2023)") to a paper index
    using author last-name + year + title signals.
    """
    cited_lower = cited_work_str.lower()

    year_match = re.search(r"\b(19[89]\d|20[0-2]\d)\b", cited_work_str)
    cited_year = year_match.group(1) if year_match else None

    best_score = 0.0
    best_idx = None

    for idx, (title, authors, year) in enumerate(
        zip(paper_titles, paper_authors, paper_years)
    ):
        if not title:
            continue

        score = 0.0

        # Author last-name match
        for author in (authors or [])[:3]:
            parts = str(author).strip().split()
            if parts:
                last_name = parts[-1].lower()
                if len(last_name) > 2 and last_name in cited_lower:
                    score += 0.40
                    break

        if score == 0.0:
            continue

        # Year match
        if cited_year and str(year) == cited_year:
            score += 0.35

        # Title word overlap bonus
        overlap = _title_word_overlap(title, cited_work_str)
        if overlap >= 0.40:
            score += 0.25 * overlap

        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx if best_score >= 0.40 else None


# ─── Main graph generator ────────────────────────────────────────────────────

def generate_citation_graph(papers: list[dict], topic: str) -> dict:
    """
    Generate a provenance-enriched D3 JSON graph from extracted papers.

    Args:
        papers: List of paper dicts from extraction_agent.
        topic:  Research topic string (used for the central concept node).

    Returns:
        D3-compatible graph dict:
        {
          'nodes': [...],
          'edges': [...],
          'node_count': int,
          'edge_count': int
        }
    """
    logger.info(
        f"Generating citation graph for {len(papers)} papers on '{topic}'…"
    )

    nodes: list[dict] = []
    links: list[dict] = []

    # ── 1. Central Topic Node ──
    nodes.append({
        "id": "topic_root",
        "name": topic.title(),
        "label": topic.title(),
        "group": 0,
        "type": "concept",
    })

    # ── 2. Build paper nodes with full provenance ──
    paper_titles: list[str] = []
    paper_authors: list[list] = []
    paper_years: list[str] = []

    for i, paper in enumerate(papers):
        paper_id = f"paper_{i}"
        title = paper.get("title", f"Unknown Paper {i}")
        authors = paper.get("authors", [])
        if isinstance(authors, str):
            authors = [authors]
        year = str(paper.get("year", "")) if paper.get("year") else ""

        paper_titles.append(title)
        paper_authors.append(authors)
        paper_years.append(year)

        methodology = paper.get("methodology", {})
        key_claims = paper.get("key_claims", [])
        limitations = paper.get("limitations", [])
        figures_tables = paper.get("figures_tables", [])

        methodology_summary = {}
        if isinstance(methodology, dict) and methodology:
            methodology_summary = {
                "approach": methodology.get("approach", ""),
                "datasets": methodology.get("datasets", []),
                "hardware": methodology.get("hardware", ""),
                "evaluation_protocol": methodology.get("evaluation_protocol", ""),
                "baseline_comparisons": methodology.get("baseline_comparisons", []),
            }

        top_claims = []
        for c in (key_claims or [])[:5]:
            if isinstance(c, dict):
                top_claims.append({
                    "claim": c.get("claim", ""),
                    "section": c.get("section", ""),
                    "page": c.get("page"),
                    "raw_quote": c.get("raw_quote", ""),
                    "evidence_type": c.get("evidence_type", ""),
                    "dataset": c.get("dataset"),
                })

        top_limitations = []
        for lim in (limitations or [])[:5]:
            if isinstance(lim, dict):
                top_limitations.append({
                    "text": lim.get("text", ""),
                    "section": lim.get("section", ""),
                    "page": lim.get("page"),
                    "raw_quote": lim.get("raw_quote", ""),
                })

        top_figures = []
        for fig in (figures_tables or [])[:8]:
            if isinstance(fig, dict):
                top_figures.append({
                    "label": fig.get("label", ""),
                    "type": fig.get("type", "figure"),
                    "page": fig.get("page"),
                    "caption": fig.get("caption", ""),
                    "key_finding": fig.get("key_finding", ""),
                    "data_points": fig.get("data_points", []),
                    "bounding_box": fig.get("bounding_box"),
                })

        # Derive display fields for the frontend CitationGraph component
        research_theme = (
            methodology.get("research_theme", "General")
            if isinstance(methodology, dict) else "General"
        )
        contribution_type = (
            methodology.get("contribution_type", "empirical")
            if isinstance(methodology, dict) else "empirical"
        )
        abstract = paper.get("abstract", paper.get("snippet", ""))
        problem_statement = (
            methodology.get("problem_statement", "")
            if isinstance(methodology, dict) else ""
        )
        if not problem_statement:
            problem_statement = _fallback_problem_statement(abstract, paper.get("text", ""))
        key_finding = (
            top_claims[0]["claim"] if top_claims else
            _fallback_key_finding(
                abstract,
                paper.get("text", ""),
                paper.get("extraction_method", "unknown"),
            )
        )
        methodology_display = methodology_summary.get("approach", "") or "Not extracted"

        node = {
            "id": paper_id,
            "name": title,
            "label": title,
            "authors": authors,
            "year": year,
            "url": paper.get("url", ""),
            "group": 1,
            "type": "paper",
            "extraction_method": paper.get("extraction_method", "unknown"),
            "citation_count": paper.get("citation_count", 0),
            "research_theme": research_theme,
            "contribution_type": contribution_type,
            "problem_statement": problem_statement,
            "key_finding": key_finding,
            "methodology": methodology_display,
            "limitations": [
                lim.get("text", "") if isinstance(lim, dict) else str(lim)
                for lim in (limitations or [])[:3]
            ],
            "figures_tables": top_figures,
            # Full provenance payload
            "methodology_full": methodology_summary,
            "key_claims": top_claims,
            "limitations_full": top_limitations,
            "char_count": paper.get("char_count", 0),
        }
        nodes.append(node)

        # Every paper links to the central topic node
        links.append({
            "source": "topic_root",
            "target": paper_id,
            "value": 1,
            "confidence": 1.0,
            "relationship": "explores",
            "citation_context": "",
            "citing_section": "",
            "citing_page": None,
        })

    # ── 3. Build inter-paper citation edges ──
    papers_with_edges = 0
    strategy_a_count = 0
    strategy_b_count = 0
    strategy_c_count = 0

    # Determine whether LLM matching is available
    use_llm_matching = bool(
        os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    )

    for i, paper in enumerate(papers):
        source_id = f"paper_{i}"
        cited_indices_added: set[int] = set()

        # ── Strategy A: citations_in_text from context_cache (highest quality) ──
        citations_in_text = paper.get("citations_in_text", [])
        if citations_in_text and isinstance(citations_in_text, list):
            for cit in citations_in_text:
                if not isinstance(cit, dict):
                    continue
                cited_work = cit.get("cited_work", "")
                if not cited_work:
                    continue

                target_idx = _match_cited_work_to_paper(
                    cited_work, paper_titles, paper_authors, paper_years
                )
                if (
                    target_idx is not None
                    and target_idx != i
                    and target_idx not in cited_indices_added
                ):
                    cited_indices_added.add(target_idx)
                    links.append({
                        "source": source_id,
                        "target": f"paper_{target_idx}",
                        "value": 3,
                        "confidence": 0.95,
                        "relationship": cit.get("relationship", "cites"),
                        "citation_context": cit.get("context_sentence", ""),
                        "citing_section": cit.get("section", ""),
                        "citing_page": cit.get("page"),
                    })
                    strategy_a_count += 1
                    logger.debug(
                        f"[A] paper_{i} → paper_{target_idx} via '{cited_work}'"
                    )

        # ── Strategy B: LLM batch matching on reference tail chunk ──
        raw_text = paper.get("text", "")
        if raw_text and use_llm_matching:
            ref_block = _get_references_section(raw_text)
            if ref_block and len(ref_block) > 100:
                matched_indices = _llm_match_citations(
                    ref_block,
                    paper_titles,
                    paper_authors,
                    paper_years,
                    source_title=paper_titles[i],
                )
                for target_idx in matched_indices:
                    if target_idx != i and target_idx not in cited_indices_added:
                        cited_indices_added.add(target_idx)
                        links.append({
                            "source": source_id,
                            "target": f"paper_{target_idx}",
                            "value": 2,
                            "confidence": 0.80,
                            "relationship": "cites",
                            "citation_context": "",
                            "citing_section": "References (LLM-matched)",
                            "citing_page": None,
                        })
                        strategy_b_count += 1
                        logger.debug(
                            f"[B] paper_{i} → paper_{target_idx} (LLM match)"
                        )

        # ── Strategy C: Robust heuristic on tail chunk (zero API cost fallback) ──
        if raw_text:
            ref_block = _get_references_section(raw_text)
            if ref_block and len(ref_block) > 100:
                ref_entries = _parse_ref_entries(ref_block)

                # Also search the full tail chunk for author+year patterns
                # when individual entry parsing yields nothing
                search_space = ref_entries if ref_entries else [ref_block[:8000]]

                for ref_text_chunk in search_space:
                    target_idx = _find_cited_paper_heuristic(
                        ref_text_chunk,
                        paper_titles,
                        paper_authors,
                        paper_years,
                    )
                    if (
                        target_idx is not None
                        and target_idx != i
                        and target_idx not in cited_indices_added
                    ):
                        cited_indices_added.add(target_idx)
                        links.append({
                            "source": source_id,
                            "target": f"paper_{target_idx}",
                            "value": 1,
                            "confidence": 0.60,
                            "relationship": "cites",
                            "citation_context": ref_text_chunk[:200],
                            "citing_section": "References (heuristic)",
                            "citing_page": None,
                        })
                        strategy_c_count += 1
                        logger.debug(
                            f"[C] paper_{i} → paper_{target_idx} (heuristic)"
                        )

        if cited_indices_added:
            papers_with_edges += 1

    total_paper_edges = strategy_a_count + strategy_b_count + strategy_c_count
    logger.info(
        f"Graph complete: {len(nodes)} nodes, {len(links)} edges "
        f"({papers_with_edges}/{len(papers)} papers had citation edges). "
        f"Strategy breakdown — A(context_cache):{strategy_a_count}, "
        f"B(LLM):{strategy_b_count}, C(heuristic):{strategy_c_count}. "
        f"Total paper-to-paper edges: {total_paper_edges}."
    )

    return {
        "nodes": nodes,
        "edges": links,
        "node_count": len(nodes),
        "edge_count": len(links),
    }
