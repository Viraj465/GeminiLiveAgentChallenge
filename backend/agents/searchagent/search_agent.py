"""
search_agent.py — ADK Tool: Multi-source academic paper search with local semantic reranking.

Flow:
  1. Searches 6 free APIs (Semantic Scholar, arXiv, Europe PMC, OpenAlex, Crossref, CORE) simultaneously.
  2. Deduplicates results by title.
  3. Reranks with local cross-encoder model (zero API cost).
  4. Returns sorted papers with relevance scores.
  NO BROWSER OR PLAYWRIGHT IS USED HERE. CAPTCHA-FREE.
"""

import asyncio
import json
import logging
import httpx
import os
import xml.etree.ElementTree as ET
from typing import List
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# ─── Local reranker (loaded once, zero API cost) ───
_reranker = None

def _get_reranker() -> CrossEncoder:
    """Lazy-load the cross-encoder reranker model."""
    global _reranker
    if _reranker is None:
        logger.info("Loading local reranker model (one-time)...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


async def _search_semantic_scholar(query: str, limit: int = 15) -> list:
    papers = []
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query": query,
                    "limit": limit,
                    "fields": "title,abstract,url,year,authors,citationCount",
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("data", []):
                    papers.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": (item.get("abstract") or "")[:300],
                        "year": item.get("year"),
                        "citation_count": item.get("citationCount", 0),
                        "authors": [a.get("name", "") for a in (item.get("authors") or [])[:5]],
                        "source": "semantic_scholar",
                    })
    except Exception as e:
        logger.warning(f"Semantic Scholar API error: {e}")
    return papers


async def _search_arxiv(query: str, limit: int = 15) -> list:
    papers = []
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "http://export.arxiv.org/api/query",
                params={"search_query": f"all:{query}", "max_results": limit}
            )
            if resp.status_code == 200:
                root = ET.fromstring(resp.text)
                for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
                    title = entry.find("{http://www.w3.org/2005/Atom}title")
                    summary = entry.find("{http://www.w3.org/2005/Atom}summary")
                    url_elem = entry.find("{http://www.w3.org/2005/Atom}id")
                    pub_elem = entry.find("{http://www.w3.org/2005/Atom}published")
                    authors = [a.find("{http://www.w3.org/2005/Atom}name").text for a in entry.findall("{http://www.w3.org/2005/Atom}author") if a.find("{http://www.w3.org/2005/Atom}name") is not None][:5]
                    
                    if title is not None and url_elem is not None:
                        url = url_elem.text.replace("abs", "pdf")
                        year = int(pub_elem.text[:4]) if pub_elem is not None else None
                        
                        papers.append({
                            "title": title.text.replace("\n", " ") if title.text else "",
                            "url": url,
                            "snippet": summary.text.replace("\n", " ")[:300] if summary is not None and summary.text else "",
                            "year": year,
                            "authors": authors,
                            "citation_count": 0,
                            "source": "arxiv"
                        })
    except Exception as e:
        logger.warning(f"arXiv API error: {e}")
    return papers


async def _search_europe_pmc(query: str, limit: int = 15) -> list:
    papers = []
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params={"query": query, "format": "json", "pageSize": limit, "resultType": "core"}
            )
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("resultList", {}).get("result", []):
                    pmcid = item.get("pmcid")
                    url = f"https://europepmc.org/articles/{pmcid}?pdf=render" if pmcid else f"https://europepmc.org/article/MED/{item.get('pmid')}"
                    authors = [a.get("fullName", "") for a in item.get("authorList", {}).get("author", [])][:5]
                    
                    papers.append({
                        "title": item.get("title", ""),
                        "url": url,
                        "snippet": item.get("abstractText", "")[:300] if item.get("abstractText") else "",
                        "year": int(item.get("pubYear", 0)) if item.get("pubYear") else None,
                        "authors": authors,
                        "citation_count": item.get("citedByCount", 0),
                        "source": "europe_pmc"
                    })
    except Exception as e:
        logger.warning(f"Europe PMC API error: {e}")
    return papers


async def _search_openalex(query: str, limit: int = 15) -> list:
    papers = []
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://api.openalex.org/works",
                params={"search": query, "per-page": limit, "filter": "has_fulltext:true"}
            )
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("results", []):
                    # Reconstruct abstract
                    abs_index = item.get("abstract_inverted_index")
                    abstract = ""
                    if abs_index:
                        max_pos = 0
                        for positions in abs_index.values():
                            if positions:
                                max_pos = max(max_pos, max(positions))
                        abs_list = [""] * (max_pos + 1)
                        for word, positions in abs_index.items():
                            for pos in positions:
                                abs_list[pos] = word
                        abstract = " ".join(abs_list)
                    
                    url = item.get("open_access", {}).get("oa_url") or item.get("doi", "")
                    authors = [a.get("author", {}).get("display_name", "") for a in item.get("authorships", [])][:5]

                    if url:
                        papers.append({
                            "title": item.get("title", ""),
                            "url": url,
                            "snippet": abstract[:300],
                            "year": item.get("publication_year"),
                            "authors": authors,
                            "citation_count": item.get("cited_by_count", 0),
                            "source": "openalex"
                        })
    except Exception as e:
        logger.warning(f"OpenAlex API error: {e}")
    return papers


async def _search_crossref(query: str, limit: int = 15) -> list:
    papers = []
    email = os.getenv("CROSSREF_EMAIL", "bot@example.com")
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://api.crossref.org/works",
                params={"query": query, "rows": limit, "filter": "has-full-text:true", "mailto": email}
            )
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("message", {}).get("items", []):
                    title = item.get("title", [""])[0] if item.get("title") else ""
                    url = item.get("URL", "")
                    
                    # try to find PDF link in item.link
                    links = item.get("link", [])
                    for link in links:
                        if link.get("content-type") == "application/pdf":
                            url = link.get("URL")
                            break
                            
                    authors = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in item.get("author", [])][:5]
                    
                    # Extract year
                    year = None
                    try:
                        year = item.get("created", {}).get("date-parts", [[None]])[0][0]
                    except (IndexError, TypeError):
                        pass

                    papers.append({
                        "title": title,
                        "url": url,
                        "snippet": item.get("abstract", "")[:300] if item.get("abstract") else "",
                        "year": year,
                        "authors": authors,
                        "citation_count": item.get("is-referenced-by-count", 0),
                        "source": "crossref"
                    })
    except Exception as e:
        logger.warning(f"Crossref API error: {e}")
    return papers


async def _search_core(query: str, limit: int = 15) -> list:
    papers = []
    api_key = os.getenv("CORE_API_KEY")
    if not api_key:
        return papers
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://api.core.ac.uk/v3/search/works",
                params={"q": query, "limit": limit},
                headers={"Authorization": f"Bearer {api_key}"}
            )
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("results", []):
                    urls = item.get("downloadUrl") or item.get("sourceFulltextUrls", [])
                    url = urls if isinstance(urls, str) else (urls[0] if urls else item.get("abstractUrl", ""))
                    authors = [a.get("name", "") for a in item.get("authors", [])][:5]
                    
                    if url:
                        papers.append({
                            "title": item.get("title", ""),
                            "url": url,
                            "snippet": item.get("abstract", "")[:300] if item.get("abstract") else "",
                            "year": item.get("yearPublished"),
                            "authors": authors,
                            "citation_count": item.get("citationCount", 0),
                            "source": "core"
                        })
    except Exception as e:
        logger.warning(f"CORE API error: {e}")
    return papers


async def _search_all_sources(query: str, max_papers: int) -> dict:
    """
    Query all API sources concurrently.
    """
    logger.info(f"Searching APIs for query: {query}")
    results = await asyncio.gather(
        _search_semantic_scholar(query, max_papers),
        _search_arxiv(query, max_papers),
        _search_europe_pmc(query, max_papers),
        _search_openalex(query, max_papers),
        _search_crossref(query, max_papers),
        _search_core(query, max_papers),
        return_exceptions=True
    )

    all_papers = []
    sources_searched = []
    
    source_names = ["Semantic Scholar", "arXiv", "Europe PMC", "OpenAlex", "Crossref", "CORE"]
    
    for i, res in enumerate(results):
        if isinstance(res, list):
            all_papers.extend(res)
            if len(res) > 0:
                sources_searched.append(source_names[i])
        elif isinstance(res, Exception):
            logger.warning(f"Source {source_names[i]} failed completely: {res}")

    return {
        "papers": all_papers,
        "sources_searched": sources_searched
    }


def _deduplicate(papers: list) -> list:
    """Deduplicate papers by normalized title."""
    seen = set()
    unique = []
    for p in papers:
        key = p.get("title", "").strip().lower()[:80]
        if key and key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _rerank_papers(query: str, papers: list) -> list:
    """
    Rerank papers by relevance using a local cross-encoder model.
    Zero API cost — runs entirely on CPU.
    """
    if not papers:
        return papers

    reranker = _get_reranker()

    # Build query-document pairs for scoring
    pairs = []
    for p in papers:
        doc_text = f"{p.get('title', '')}. {p.get('snippet', '')}"
        pairs.append((query, doc_text))

    # Score all pairs in one batch (fast)
    scores = reranker.predict(pairs)

    # Attach scores and sort descending
    for i, p in enumerate(papers):
        p["relevance_score"] = round(float(scores[i]), 4)

    papers.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    return papers


async def search_papers_async(query: str, max_papers: int = 15) -> dict:
    """
    ADK Tool: Searches multiple academic databases via API concurrently,
    deduplicates results, and reranks by relevance.
    """
    # ── Source 1-6: API Fast Search (zero browser) ──
    search_result = await _search_all_sources(query, max_papers * 2)
    all_papers = search_result["papers"]
    sources = search_result["sources_searched"]
    
    logger.info(f"Raw API papers found: {len(all_papers)}")

    # ── Deduplicate ──
    unique_papers = _deduplicate(all_papers)
    logger.info(f"After dedup: {len(unique_papers)} unique papers")

    # ── Rerank locally (zero API cost) ──
    ranked_papers = _rerank_papers(query, unique_papers)

    # ── Trim to max ──
    final_papers = ranked_papers[:max_papers]

    # Return standard ADK format
    return {
        "status": "success",
        "query": query,
        "papers_found": len(final_papers),
        "total_raw": len(all_papers),
        "papers": final_papers,
        "urls": [p.get("url") for p in final_papers],
        "urls_found": len(final_papers),
        "sources_searched": sources,
        "message": f"Found {len(final_papers)} papers across {len(sources)} API sources (zero CAPTCHA risk).",
    }


# Sync wrapper for ADK (ADK calls sync functions)
def search_papers_sync(query: str, max_papers: int = 15) -> dict:
    """ADK-compatible sync wrapper.
    ADK Tool: One sentence description of what this does.
    Required query parameter for topics to search.
    Returns dict with status key always present.
    """
    try:
        result = asyncio.run(search_papers_async(query, max_papers))
        return result
    except Exception as e:
        return {"status": "error", "message": f"Search failed: {str(e)}"}


# Override for ADK tool registration
search_papers = search_papers_sync