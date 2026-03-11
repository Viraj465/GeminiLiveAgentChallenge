"""
pdf_processor.py — In-memory PDF ingestion pipeline.

Features:
  - Takes a list of dicts: {"title": str, "pdf_url": str}
  - Streams PDFs strictly into RAM (BytesIO) using httpx
  - Extracts text from all pages using PyMuPDF (fitz)
  - Applies essential text deduplication and cleaning
"""

import httpx
import fitz
import asyncio
import logging
from io import BytesIO

logger = logging.getLogger(__name__)


async def fetch_pdf_bytes(client: httpx.AsyncClient, url: str) -> bytes:
    """Download a PDF directly into memory."""
    try:
        response = await client.get(url, timeout=30.0, follow_redirects=True)
        response.raise_for_status()
        
        # Verify it actually is a PDF (some servers return HTML paywalls)
        content_type = response.headers.get("Content-Type", "").lower()
        if "application/pdf" not in content_type and not url.endswith('.pdf'):
            logger.warning(f"URL {url} returned content-type: {content_type}, might not be a PDF.")
            
        return response.content
    except Exception as e:
        logger.error(f"Failed to fetch PDF from {url}: {e}")
        return None


def extract_text_from_bytes(pdf_bytes: bytes) -> str:
    """
    Open the bytes stream with PyMuPDF and extract raw text.
    Strictly adheres to: Never Write Files to Disk Rule.
    """
    text = ""
    try:
        # Load PDF from memory stream
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page_num in range(len(doc)):
                # extract_text("text") pulls standard textual data
                text += doc[page_num].get_text("text") + "\n\n"
        
        # Clean excessive newlines/whitespace
        cleaned_text = " ".join(text.split())
        return cleaned_text
        
    except Exception as e:
        logger.error(f"PyMuPDF failed to parse byte stream: {e}")
        return ""


async def process_scraped_papers(papers: list[dict]) -> dict:
    """
    ADK Tool: Given a list of paper dicts containing 'pdf_url' and 'title', 
    downloads them into RAM and extracts text.
    
    Returns a dict mapping: { "Paper Title": "Extracted cleaned text..." }
    """
    logger.info(f"Starting PDF processing for {len(papers)} papers...")
    
    results = {}
    
    # We use httpx AsyncClient for concurrent downloads
    async with httpx.AsyncClient(verify=False) as client: # verify=False helps with strict academic proxies
        
        tasks = []
        for paper in papers:
            title = paper.get("title", "Untitled")
            pdf_url = paper.get("pdf_url")
            
            if not pdf_url:
                logger.warning(f"No PDF URL for '{title}'")
                continue
                
            # Create download tasks
            tasks.append((title, fetch_pdf_bytes(client, pdf_url)))
            
        # Await all downloads concurrently
        for title, dl_task in tasks:
            pdf_bytes = await dl_task
            
            if pdf_bytes:
                logger.info(f"Extracting text: {title} ({len(pdf_bytes)} bytes)")
                
                # Execute CPU-bound PyMuPDF text extraction in thread pool to avoid blocking async loop
                extracted_text = await asyncio.to_thread(extract_text_from_bytes, pdf_bytes)
                
                if extracted_text:
                    results[title] = extracted_text
                    logger.info(f"Successfully extracted {len(extracted_text)} characters from '{title}'")
                else:
                    results[title] = "[Extraction Failed] No readable text found in PDF."
            else:
                results[title] = "[Download Failed] Could not fetch PDF from URL."
                
    logger.info(f"PDF processing complete. Successfully ingested {len([k for k in results.keys() if 'Failed' not in results[k]])} papers.")
                
    return {
        "status": "success",
        "data": results
    }
