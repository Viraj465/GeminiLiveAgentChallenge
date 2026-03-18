"""
search_agent.py — ADK Tool: 5-Phase Deep Search Strategy.

Phase 1 — Planning:
    Gemini decomposes the query into sub-queries covering different angles
    (methodology, applications, comparisons, recent advances, critiques).

Phase 2 — Iterative API Search:
    Searches 10 source tiers concurrently:
      Academic:   Semantic Scholar, arXiv, Europe PMC, OpenAlex, Crossref, CORE,
                  BASE (Bielefeld), DOAJ (Directory of Open Access Journals)
      Preprints:  bioRxiv / medRxiv (via Europe PMC)
      Web:        SerpAPI / Google Custom Search (if key available)

Phase 3 — Deep Browsing (gap-fill):
    For sub-queries that returned < MIN_RESULTS papers, fires targeted
    follow-up searches against the same source tier that had the most hits.

Phase 4 — Gap Identification:
    Gemini reviews the collected titles/abstracts and identifies missing
    perspectives, then generates 1-2 additional targeted sub-queries.

Phase 5 — Synthesis / Reranking:
    Local cross-encoder reranks all unique papers by relevance.
    Returns the top N with full metadata.

Zero browser / zero CAPTCHA risk — all API-based.
"""

import asyncio
import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from typing import Optional

import httpx
from sentence_transformers import CrossEncoder

from prompts import SEARCH_PLANNING_PROMPT, SEARCH_GAP_IDENTIFICATION_PROMPT

logger = logging.getLogger(__name__)

# Constants 
MIN_RESULTS_PER_SUBQUERY = 3   # trigger gap-fill if a sub-query returns fewer
MAX_SUBQUERIES = 5             # max sub-queries from planning phase
MAX_GAP_QUERIES = 2            # max additional queries from gap identification
DEFAULT_MAX_PAPERS = 20        # default final paper count

# Local reranker (loaded once, zero API cost) 
_reranker: Optional[CrossEncoder] = None

def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info("Loading local reranker model (one-time)…")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker



# PHASE 1 — PLANNING: Decompose query into sub-queries


async def _plan_subqueries(query: str) -> list[str]:
    """
    Use Gemini to decompose the research query into focused sub-queries that
    cover different angles: methodology, applications, comparisons, recent
    advances, limitations/critiques.

    Falls back to a simple list of the original query if the API call fails.
    """
    try:
        from google import genai
        from config import settings

        client = genai.Client(api_key=settings.GEMINI_API_KEY)

        prompt = SEARCH_PLANNING_PROMPT.format(max_subqueries=MAX_SUBQUERIES, query=query)

        response = client.models.generate_content(
            model=settings.GOOGLE_REASONING_MODEL,
            contents=prompt,
        )
        raw = response.text.strip()
        # Extract JSON array
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            subqueries = json.loads(match.group(0))
            if isinstance(subqueries, list) and subqueries:
                # Always include the original query as the first entry
                result = [query] + [s for s in subqueries if s != query]
                logger.info(f"[Phase 1] Planning: {len(result)} sub-queries generated")
                return result[:MAX_SUBQUERIES + 1]
    except Exception as e:
        logger.warning(f"[Phase 1] Planning failed, using original query only: {e}")

    return [query]



# PHASE 2 — SOURCE TIER SEARCH FUNCTIONS


async def _search_semantic_scholar(query: str, limit: int = 15) -> list:
    papers = []
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query": query,
                    "limit": limit,
                    "fields": "title,abstract,url,year,authors,citationCount,openAccessPdf",
                },
            )
            if resp.status_code == 200:
                for item in resp.json().get("data", []):
                    # Prefer open-access PDF URL
                    oa = item.get("openAccessPdf") or {}
                    url = oa.get("url") or item.get("url", "")
                    papers.append({
                        "title": item.get("title", ""),
                        "url": url,
                        "snippet": (item.get("abstract") or "")[:400],
                        "year": item.get("year"),
                        "citation_count": item.get("citationCount", 0),
                        "authors": [a.get("name", "") for a in (item.get("authors") or [])[:5]],
                        "source": "semantic_scholar",
                    })
    except Exception as e:
        logger.warning(f"Semantic Scholar error: {e}")
    return papers


async def _search_arxiv(query: str, limit: int = 15) -> list:
    papers = []
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "http://export.arxiv.org/api/query",
                params={
                    "search_query": f"all:{query}",
                    "max_results": limit,
                    "sortBy": "relevance",
                    "sortOrder": "descending",
                },
            )
            if resp.status_code == 200:
                root = ET.fromstring(resp.text)
                ns = "{http://www.w3.org/2005/Atom}"
                for entry in root.findall(f"{ns}entry"):
                    title_el = entry.find(f"{ns}title")
                    summary_el = entry.find(f"{ns}summary")
                    id_el = entry.find(f"{ns}id")
                    pub_el = entry.find(f"{ns}published")
                    authors = [
                        a.find(f"{ns}name").text
                        for a in entry.findall(f"{ns}author")
                        if a.find(f"{ns}name") is not None
                    ][:5]
                    if title_el is not None and id_el is not None:
                        arxiv_id = id_el.text.strip()
                        # Convert abs URL to PDF URL
                        pdf_url = arxiv_id.replace("/abs/", "/pdf/") + ".pdf"
                        year = int(pub_el.text[:4]) if pub_el is not None else None
                        papers.append({
                            "title": (title_el.text or "").replace("\n", " ").strip(),
                            "url": pdf_url,
                            "snippet": (summary_el.text or "").replace("\n", " ")[:400],
                            "year": year,
                            "authors": authors,
                            "citation_count": 0,
                            "source": "arxiv",
                        })
    except Exception as e:
        logger.warning(f"arXiv error: {e}")
    return papers


async def _search_europe_pmc(query: str, limit: int = 15) -> list:
    """Covers PubMed, PMC, bioRxiv, medRxiv, and preprints."""
    papers = []
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params={
                    "query": query,
                    "format": "json",
                    "pageSize": limit,
                    "resultType": "core",
                    "sort": "CITED desc",
                },
            )
            if resp.status_code == 200:
                for item in resp.json().get("resultList", {}).get("result", []):
                    pmcid = item.get("pmcid")
                    url = (
                        f"https://europepmc.org/articles/{pmcid}?pdf=render"
                        if pmcid
                        else f"https://europepmc.org/article/MED/{item.get('pmid', '')}"
                    )
                    authors = [
                        a.get("fullName", "")
                        for a in item.get("authorList", {}).get("author", [])
                    ][:5]
                    papers.append({
                        "title": item.get("title", ""),
                        "url": url,
                        "snippet": (item.get("abstractText") or "")[:400],
                        "year": int(item["pubYear"]) if item.get("pubYear") else None,
                        "authors": authors,
                        "citation_count": item.get("citedByCount", 0),
                        "source": "europe_pmc",
                    })
    except Exception as e:
        logger.warning(f"Europe PMC error: {e}")
    return papers


async def _search_openalex(query: str, limit: int = 15) -> list:
    papers = []
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://api.openalex.org/works",
                params={
                    "search": query,
                    "per-page": limit,
                    "filter": "has_fulltext:true",
                    "sort": "cited_by_count:desc",
                },
                headers={"User-Agent": "ResearchAgent/1.0 (mailto:research@example.com)"},
            )
            if resp.status_code == 200:
                for item in resp.json().get("results", []):
                    # Reconstruct abstract from inverted index
                    abs_index = item.get("abstract_inverted_index") or {}
                    abstract = ""
                    if abs_index:
                        max_pos = max(
                            (max(pos) for pos in abs_index.values() if pos),
                            default=0,
                        )
                        abs_list = [""] * (max_pos + 1)
                        for word, positions in abs_index.items():
                            for pos in positions:
                                if pos <= max_pos:
                                    abs_list[pos] = word
                        abstract = " ".join(abs_list)

                    url = (
                        item.get("open_access", {}).get("oa_url")
                        or item.get("doi", "")
                    )
                    authors = [
                        a.get("author", {}).get("display_name", "")
                        for a in item.get("authorships", [])
                    ][:5]
                    if url:
                        papers.append({
                            "title": item.get("title", ""),
                            "url": url,
                            "snippet": abstract[:400],
                            "year": item.get("publication_year"),
                            "authors": authors,
                            "citation_count": item.get("cited_by_count", 0),
                            "source": "openalex",
                        })
    except Exception as e:
        logger.warning(f"OpenAlex error: {e}")
    return papers


async def _search_crossref(query: str, limit: int = 15) -> list:
    papers = []
    email = os.getenv("CROSSREF_EMAIL", "bot@example.com")
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://api.crossref.org/works",
                params={
                    "query": query,
                    "rows": limit,
                    "filter": "has-full-text:true",
                    "mailto": email,
                    "sort": "is-referenced-by-count",
                    "order": "desc",
                },
            )
            if resp.status_code == 200:
                for item in resp.json().get("message", {}).get("items", []):
                    title = (item.get("title") or [""])[0]
                    url = item.get("URL", "")
                    for link in item.get("link", []):
                        if link.get("content-type") == "application/pdf":
                            url = link.get("URL", url)
                            break
                    authors = [
                        f"{a.get('given', '')} {a.get('family', '')}".strip()
                        for a in item.get("author", [])
                    ][:5]
                    year = None
                    try:
                        year = item["created"]["date-parts"][0][0]
                    except (KeyError, IndexError, TypeError):
                        pass
                    papers.append({
                        "title": title,
                        "url": url,
                        "snippet": (item.get("abstract") or "")[:400],
                        "year": year,
                        "authors": authors,
                        "citation_count": item.get("is-referenced-by-count", 0),
                        "source": "crossref",
                    })
    except Exception as e:
        logger.warning(f"Crossref error: {e}")
    return papers


async def _search_core(query: str, limit: int = 15) -> list:
    """CORE — world's largest open-access aggregator."""
    papers = []
    api_key = os.getenv("CORE_API_KEY")
    if not api_key:
        return papers
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://api.core.ac.uk/v3/search/works",
                params={"q": query, "limit": limit},
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if resp.status_code == 200:
                for item in resp.json().get("results", []):
                    urls = item.get("downloadUrl") or item.get("sourceFulltextUrls", [])
                    url = urls if isinstance(urls, str) else (urls[0] if urls else item.get("abstractUrl", ""))
                    authors = [a.get("name", "") for a in item.get("authors", [])][:5]
                    if url:
                        papers.append({
                            "title": item.get("title", ""),
                            "url": url,
                            "snippet": (item.get("abstract") or "")[:400],
                            "year": item.get("yearPublished"),
                            "authors": authors,
                            "citation_count": item.get("citationCount", 0),
                            "source": "core",
                        })
    except Exception as e:
        logger.warning(f"CORE error: {e}")
    return papers


async def _search_base(query: str, limit: int = 15) -> list:
    """
    BASE — Bielefeld Academic Search Engine.
    One of the largest academic web search engines; indexes 300M+ documents.
    No API key required.
    """
    papers = []
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://api.base-search.net/cgi-bin/BaseHttpSearchInterface.fcgi",
                params={
                    "func": "PerformSearch",
                    "query": query,
                    "hits": limit,
                    "offset": 0,
                    "format": "json",
                    "boost": "oa",          # prefer open-access
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                docs = data.get("response", {}).get("docs", [])
                for item in docs:
                    url = ""
                    links = item.get("dclink", [])
                    if isinstance(links, list) and links:
                        url = links[0]
                    elif isinstance(links, str):
                        url = links

                    authors_raw = item.get("dccontributor") or item.get("dccreator") or []
                    if isinstance(authors_raw, str):
                        authors_raw = [authors_raw]

                    year = None
                    date_str = item.get("dcyear") or item.get("dcdate", "")
                    if date_str:
                        m = re.search(r"\b(19[89]\d|20[0-2]\d)\b", str(date_str))
                        if m:
                            year = int(m.group(1))

                    papers.append({
                        "title": item.get("dctitle", ""),
                        "url": url,
                        "snippet": (item.get("dcdescription") or "")[:400],
                        "year": year,
                        "authors": authors_raw[:5],
                        "citation_count": 0,
                        "source": "base",
                    })
    except Exception as e:
        logger.warning(f"BASE error: {e}")
    return papers


async def _search_doaj(query: str, limit: int = 15) -> list:
    """
    DOAJ — Directory of Open Access Journals.
    Community-curated; everything is freely readable without login.
    """
    papers = []
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://doaj.org/api/search/articles/" + httpx.URL("", params={"q": query}).params.get("q", query),
                params={"q": query, "pageSize": limit, "page": 1},
            )
            if resp.status_code == 200:
                for item in resp.json().get("results", []):
                    bib = item.get("bibjson", {})
                    title = bib.get("title", "")
                    # Build URL from DOI or journal link
                    doi = bib.get("identifier", [{}])
                    url = ""
                    for ident in doi:
                        if ident.get("type") == "doi":
                            url = f"https://doi.org/{ident.get('id', '')}"
                            break
                    if not url:
                        links = bib.get("link", [])
                        for lnk in links:
                            if lnk.get("type") == "fulltext":
                                url = lnk.get("url", "")
                                break

                    authors = [
                        a.get("name", "") for a in bib.get("author", [])
                    ][:5]
                    year = bib.get("year")
                    abstract = bib.get("abstract", "")

                    if title:
                        papers.append({
                            "title": title,
                            "url": url,
                            "snippet": abstract[:400],
                            "year": int(year) if year else None,
                            "authors": authors,
                            "citation_count": 0,
                            "source": "doaj",
                        })
    except Exception as e:
        logger.warning(f"DOAJ error: {e}")
    return papers


async def _search_google_custom(query: str, limit: int = 10) -> list:
    """
    Google Custom Search API (requires GOOGLE_CSE_ID + GOOGLE_CSE_API_KEY).
    Used as a web-tier fallback to find high-authority news, industry, and
    niche community sources (Reddit, Stack Overflow, blogs).
    Only activated when both env vars are set.
    """
    papers = []
    api_key = os.getenv("GOOGLE_CSE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    if not api_key or not cse_id:
        return papers
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": api_key,
                    "cx": cse_id,
                    "q": query,
                    "num": min(limit, 10),
                },
            )
            if resp.status_code == 200:
                for item in resp.json().get("items", []):
                    papers.append({
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", "")[:400],
                        "year": None,
                        "authors": [],
                        "citation_count": 0,
                        "source": "google_cse",
                    })
    except Exception as e:
        logger.warning(f"Google CSE error: {e}")
    return papers



# PHASE 2 — ITERATIVE SEARCH: Run all sources for a single sub-query


_SOURCE_NAMES = [
    "Semantic Scholar", "arXiv", "Europe PMC", "OpenAlex",
    "Crossref", "CORE", "BASE", "DOAJ", "Google CSE",
]

async def _search_all_sources(query: str, per_source_limit: int = 15) -> dict:
    """Run all source tiers concurrently for a single query."""
    results = await asyncio.gather(
        _search_semantic_scholar(query, per_source_limit),
        _search_arxiv(query, per_source_limit),
        _search_europe_pmc(query, per_source_limit),
        _search_openalex(query, per_source_limit),
        _search_crossref(query, per_source_limit),
        _search_core(query, per_source_limit),
        _search_base(query, per_source_limit),
        _search_doaj(query, per_source_limit),
        _search_google_custom(query, 10),
        return_exceptions=True,
    )

    all_papers: list[dict] = []
    sources_hit: list[str] = []
    source_counts: dict[str, int] = {}

    for i, res in enumerate(results):
        name = _SOURCE_NAMES[i] if i < len(_SOURCE_NAMES) else f"source_{i}"
        if isinstance(res, list):
            all_papers.extend(res)
            if res:
                sources_hit.append(name)
                source_counts[name] = len(res)
        elif isinstance(res, Exception):
            logger.warning(f"Source '{name}' failed: {res}")

    return {
        "papers": all_papers,
        "sources_hit": sources_hit,
        "source_counts": source_counts,
    }



# PHASE 3 — DEEP BROWSING: Gap-fill for under-served sub-queries


async def _deep_browse_gap_fill(
    subquery: str,
    existing_titles: set[str],
    best_source_fn,
    limit: int = 10,
) -> list[dict]:
    """
    For a sub-query that returned fewer than MIN_RESULTS_PER_SUBQUERY papers,
    fire a targeted follow-up search against the best-performing source.
    Returns only papers not already in existing_titles.
    """
    logger.info(f"[Phase 3] Gap-fill for sub-query: '{subquery[:60]}'")
    papers = await best_source_fn(subquery, limit)
    new_papers = []
    for p in papers:
        key = (p.get("title") or "").lower().strip()[:80]
        if key and key not in existing_titles:
            existing_titles.add(key)
            new_papers.append(p)
    logger.info(f"[Phase 3] Gap-fill found {len(new_papers)} new papers")
    return new_papers



# PHASE 4 — GAP IDENTIFICATION: Gemini reviews corpus and finds blind spots


async def _identify_gaps(query: str, papers: list[dict]) -> list[str]:
    """
    Ask Gemini to review the collected paper titles/abstracts and identify
    missing perspectives. Returns up to MAX_GAP_QUERIES additional sub-queries.
    """
    if not papers:
        return []
    try:
        from google import genai
        from config import settings

        client = genai.Client(api_key=settings.GEMINI_API_KEY)

        # Build a compact summary of what we have
        corpus_summary = "\n".join(
            f"- {p.get('title', 'Unknown')} ({p.get('year', 'n/a')}) [{p.get('source', '')}]"
            for p in papers[:30]
        )

        prompt = SEARCH_GAP_IDENTIFICATION_PROMPT.format(
            query=query,
            corpus_summary=corpus_summary,
            max_gap_queries=MAX_GAP_QUERIES
        )

        response = client.models.generate_content(
            model=settings.GOOGLE_REASONING_MODEL,
            contents=prompt,
        )
        raw = response.text.strip()
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            gap_queries = json.loads(match.group(0))
            if isinstance(gap_queries, list):
                logger.info(f"[Phase 4] Gap identification: {len(gap_queries)} gap queries → {gap_queries}")
                return [q for q in gap_queries if isinstance(q, str)][:MAX_GAP_QUERIES]
    except Exception as e:
        logger.warning(f"[Phase 4] Gap identification failed: {e}")
    return []



# PHASE 5 — SYNTHESIS / RERANKING


def _deduplicate(papers: list[dict]) -> list[dict]:
    """Deduplicate by normalized title (first 80 chars)."""
    seen: set[str] = set()
    unique: list[dict] = []
    for p in papers:
        key = (p.get("title") or "").strip().lower()[:80]
        if key and key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _filter_paywalled(papers: list[dict]) -> list[dict]:
    """
    Remove papers from known paywalled domains that cannot be extracted.
    Keep everything else — including niche community sources.
    """
    BLOCKED_DOMAINS = {
        "ieeexplore.ieee.org", "ieee.org",
        "sciencedirect.com", "linkinghub.elsevier.com",
        "springer.com", "link.springer.com",
        "dl.acm.org",
        "wiley.com", "onlinelibrary.wiley.com",
        "tandfonline.com",
        "jstor.org",
        "nature.com",
        "science.org",
        "researchgate.net",
        "academia.edu",
    }
    clean = []
    for p in papers:
        url = p.get("url", "")
        blocked = any(domain in url for domain in BLOCKED_DOMAINS)
        if not blocked:
            clean.append(p)
        else:
            logger.debug(f"Filtered paywalled URL: {url}")
    return clean


def _rerank_papers(query: str, papers: list[dict]) -> list[dict]:
    """Rerank papers by relevance using a local cross-encoder (zero API cost)."""
    if not papers:
        return papers
    reranker = _get_reranker()
    pairs = [
        (query, f"{p.get('title', '')}. {p.get('snippet', '')}")
        for p in papers
    ]
    scores = reranker.predict(pairs)
    for i, p in enumerate(papers):
        p["relevance_score"] = round(float(scores[i]), 4)
    papers.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    return papers



# MAIN ORCHESTRATOR


async def search_papers_async(query: str, max_papers: int = DEFAULT_MAX_PAPERS) -> dict:
    """
    5-Phase Deep Search:
      1. Planning   — decompose query into sub-queries
      2. Iterative  — search all 9 source tiers per sub-query
      3. Deep Browse— gap-fill under-served sub-queries
      4. Gap ID     — Gemini identifies missing perspectives
      5. Synthesis  — deduplicate, filter paywalled, rerank
    """
    all_papers: list[dict] = []
    all_sources_hit: set[str] = set()
    seen_titles: set[str] = set()

    # ── Phase 1: Planning ──────────────────────────────────────────────────
    subqueries = await _plan_subqueries(query)
    logger.info(f"[Phase 1] Sub-queries: {subqueries}")

    # ── Phase 2: Iterative search across all source tiers ─────────────────
    per_source = max(10, max_papers)
    for sq in subqueries:
        result = await _search_all_sources(sq, per_source_limit=per_source)
        sq_papers = result["papers"]
        all_sources_hit.update(result["sources_hit"])

        new_count = 0
        for p in sq_papers:
            key = (p.get("title") or "").lower().strip()[:80]
            if key and key not in seen_titles:
                seen_titles.add(key)
                all_papers.append(p)
                new_count += 1

        logger.info(
            f"[Phase 2] Sub-query '{sq[:50]}': "
            f"{new_count} new papers (sources: {result['sources_hit']})"
        )

        # ── Phase 3: Deep browse gap-fill ─────────────────────────────────
        if new_count < MIN_RESULTS_PER_SUBQUERY:
            # Use arXiv as the reliable fallback for gap-fill
            gap_papers = await _deep_browse_gap_fill(
                sq, seen_titles, _search_arxiv, limit=10
            )
            all_papers.extend(gap_papers)
            if gap_papers:
                all_sources_hit.add("arXiv (gap-fill)")

    # ── Phase 4: Gap identification ────────────────────────────────────────
    gap_queries = await _identify_gaps(query, all_papers)
    for gq in gap_queries:
        result = await _search_all_sources(gq, per_source_limit=10)
        all_sources_hit.update(result["sources_hit"])
        for p in result["papers"]:
            key = (p.get("title") or "").lower().strip()[:80]
            if key and key not in seen_titles:
                seen_titles.add(key)
                all_papers.append(p)
        logger.info(f"[Phase 4] Gap query '{gq[:50]}': {len(result['papers'])} papers")

    # ── Phase 5: Synthesis / Reranking ────────────────────────────────────
    unique_papers = _deduplicate(all_papers)
    logger.info(f"[Phase 5] After dedup: {len(unique_papers)} unique papers")

    open_papers = _filter_paywalled(unique_papers)
    logger.info(f"[Phase 5] After paywall filter: {len(open_papers)} papers")

    ranked_papers = _rerank_papers(query, open_papers)
    final_papers = ranked_papers[:max_papers]

    logger.info(
        f"Deep search complete: {len(final_papers)} papers returned "
        f"from {len(all_sources_hit)} sources. "
        f"Sub-queries: {len(subqueries)}, Gap queries: {len(gap_queries)}"
    )

    return {
        "status": "success",
        "query": query,
        "subqueries_used": subqueries + gap_queries,
        "papers_found": len(final_papers),
        "total_raw": len(all_papers),
        "papers": final_papers,
        "urls": [p.get("url") for p in final_papers],
        "urls_found": len(final_papers),
        "sources_searched": sorted(all_sources_hit),
        "message": (
            f"Deep search: {len(final_papers)} papers from {len(all_sources_hit)} sources "
            f"via {len(subqueries)} sub-queries + {len(gap_queries)} gap queries."
        ),
    }


# ── Sync wrapper for ADK tool registration ────────────────────────────────────

def search_papers(query: str, max_papers: int = DEFAULT_MAX_PAPERS) -> dict:
    """
    ADK Tool: 5-phase deep search across 9 academic and web source tiers.

    Phases: Planning → Iterative Search → Deep Browse → Gap Identification → Synthesis.
    Sources: Semantic Scholar, arXiv, Europe PMC, OpenAlex, Crossref, CORE,
             BASE, DOAJ, Google CSE (if configured).

    Args:
        query:      The research query string.
        max_papers: Maximum number of papers to return (default 20).

    Returns:
        dict with status, papers, sources_searched, subqueries_used.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(
                search_papers_async(query, max_papers), loop
            )
            return future.result(timeout=120)
        else:
            return loop.run_until_complete(search_papers_async(query, max_papers))
    except RuntimeError:
        return asyncio.run(search_papers_async(query, max_papers))
    except Exception as e:
        logger.error(f"search_papers failed: {e}", exc_info=True)
        return {"status": "error", "message": f"Search failed: {str(e)}", "papers": []}


# Emit startup warning if CORE key is absent
if not os.getenv("CORE_API_KEY"):
    logger.warning(
        "CORE_API_KEY not set — CORE search disabled. "
        "Free key at https://core.ac.uk/services/api"
    )
