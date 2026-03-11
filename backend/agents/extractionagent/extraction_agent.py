"""
extraction_agent.py — ADK Tool: In-memory PDF text extraction.

Flow:
  1. For each URL, try httpx PDF download.
  2. If success: PyMuPDF extracts text from bytes in RAM (zero disk writes).
  3. If fail: Visual fallback using BrowserController + Gemini purely on CAPTCHA-safe domains.
  4. Returns text for synthesis.
"""

import asyncio
import logging
import os
import httpx
import fitz

from core.browser import BrowserController
from core.vision_loop import run_vision_loop

logger = logging.getLogger(__name__)

SAFE_DOMAINS = ["arxiv.org", "europepmc.org", "semanticscholar.org", "pubmed.ncbi.nlm.nih.gov", "core.ac.uk"]

async def _download_pdf_bytes(url: str) -> bytes | None:
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            resp = await client.get(url)
            # Basic validation that it's a PDF
            if resp.status_code == 200 and b"%PDF" in resp.content[:1024]:
                return resp.content
    except Exception as e:
        logger.warning(f"Failed to download PDF from {url}: {e}")
    return None

async def _visual_fallback(url: str) -> str | None:
    # Check if domain is safe to avoid CAPTCHAs
    is_safe = any(domain in url for domain in SAFE_DOMAINS)
    if not is_safe or os.getenv("USE_VISUAL_FALLBACK", "true").lower() != "true":
        logger.info(f"Skipping visual fallback for {url} (not safe or disabled)")
        return None

    logger.info(f"Using visual fallback for {url}")
    browser = BrowserController()
    await browser.start()
    text = ""
    try:
        task = f"Navigate to {url}\nRead the abstract, introduction, and key findings of this paper visually. Return 'done' when finished and include the text you read in the 'reason' field."
        async for action in run_vision_loop(browser, task, max_steps=10):
            if action.get("action") == "done":
                text = action.get("reason", "")
                break
    except Exception as e:
        logger.error(f"Visual fallback error: {e}")
    finally:
        await browser.close()
    
    return text if len(text) > 50 else None

async def _extract_single(paper: dict) -> dict:
    url = paper.get("url")
    title = paper.get("title", "Unknown")
    abstract = paper.get("snippet", "")
    
    result = {
        "url": url,
        "title": title,
        "authors": paper.get("authors", []),
        "year": paper.get("year"),
        "abstract": abstract,
        "text": "",
        "char_count": 0,
        "status": "error",
        "extraction_method": "abstract_only"
    }

    if not url:
        if abstract:
            result["text"] = abstract
            result["char_count"] = len(abstract)
            result["status"] = "success"
        return result

    # Step 1: Try PDF download
    pdf_bytes = await _download_pdf_bytes(url)
    if pdf_bytes:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            del pdf_bytes
            
            if len(text.strip()) > 200:
                result["text"] = text[:40000] # Cap at 40k chars
                result["char_count"] = len(result["text"])
                result["status"] = "success"
                result["extraction_method"] = "pdf"
                return result
        except Exception as e:
            logger.warning(f"PyMuPDF error for {url}: {e}")

    # Step 2: Visual Fallback
    text = await _visual_fallback(url)
    if text:
        result["text"] = text[:40000]
        result["char_count"] = len(result["text"])
        result["status"] = "success"
        result["extraction_method"] = "visual"
        return result

    # Step 3: Abstract Only Fallback
    if abstract:
        result["text"] = abstract
        result["char_count"] = len(abstract)
        result["status"] = "success"
        result["extraction_method"] = "abstract_only"
        
    return result

async def _extract_in_batches(papers: list) -> dict:
    batch_size = int(os.getenv("BATCH_SIZE", 3))
    all_extractions = []
    
    # Process papers in batches safely
    for i in range(0, len(papers), batch_size):
        batch = papers[i : i + batch_size]
        logger.info(f"Extracting batch {i//batch_size + 1}")
        
        tasks = [_extract_single(p) for p in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for r in results:
            if isinstance(r, dict):
                all_extractions.append(r)
            else:
                logger.error(f"Task failed in batch: {r}")
                
        await asyncio.sleep(1) # Delay between batches
        
    success_count = sum(1 for e in all_extractions if e.get("status") == "success")
    return {
        "status": "success",
        "papers_extracted": success_count,
        "extractions": all_extractions
    }

def extract_papers(papers: list = None, urls: list = None) -> dict:
    """
    ADK Tool: Download PDFs into RAM. Extract text. Zero disk. Zero CAPTCHA.
    If PDF fails, fallback to Visual browser reading for CAPTCHA-safe sites only.
    """
    if papers is None:
        papers = [{"url": u} for u in (urls or [])]
        
    if not papers:
        return {"status": "error", "message": "No papers provided"}
        
    try:
        return asyncio.run(_extract_in_batches(papers))
    except Exception as e:
        return {"status": "error", "message": f"Extraction failed: {str(e)}"}
