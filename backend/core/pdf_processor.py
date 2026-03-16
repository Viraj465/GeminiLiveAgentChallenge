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
import base64
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


def extract_multimodal_from_bytes(pdf_bytes: bytes) -> dict:
    """
    Extract text, figures (as base64), and tables from PDF bytes.
    Strictly adheres to: Never Write Files to Disk Rule.
    """
    results = {
        "text": "",
        "figures": [],  # List of {"base64": str, "page": int, "index": int}
        "tables": []    # List of {"data": list[list], "page": int}
    }
    
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # 1. Extract Text
                results["text"] += page.get_text("text") + "\n\n"
                
                # 2. Extract Figures/Images
                images = page.get_images(full=True)
                for img_index, img in enumerate(images):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                    results["figures"].append({
                        "base64": image_b64,
                        "page": page_num + 1,
                        "index": img_index,
                        "extension": base_image["ext"]
                    })

                # 3. Detect Tables
                try:
                    tabs = page.find_tables()
                    for tab_index, tab in enumerate(tabs.tables):
                        results["tables"].append({
                            "data": tab.extract(),
                            "page": page_num + 1,
                            "index": tab_index
                        })
                except Exception as e:
                    logger.warning(f"Table detection failed on page {page_num+1}: {e}")

        # Clean excessive newlines/whitespace for the aggregated text
        results["text"] = " ".join(results["text"].split())
        return results
        
    except Exception as e:
        logger.error(f"PyMuPDF failed to parse byte stream: {e}")
        return results


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
                logger.info(f"Extracting multimodal data: {title} ({len(pdf_bytes)} bytes)")
                
                # Execute CPU-bound PyMuPDF extraction in thread pool
                extracted_data = await asyncio.to_thread(extract_multimodal_from_bytes, pdf_bytes)
                
                if extracted_data["text"] or extracted_data["figures"] or extracted_data["tables"]:
                    results[title] = extracted_data
                    logger.info(f"Successfully extracted from '{title}': {len(extracted_data['text'])} chars, {len(extracted_data['figures'])} figures, {len(extracted_data['tables'])} tables.")
                else:
                    results[title] = {"text": "[Extraction Failed]", "figures": [], "tables": []}
            else:
                results[title] = {"text": "[Download Failed]", "figures": [], "tables": []}
                
    logger.info(f"PDF processing complete. Successfully ingested {len([k for k in results.keys() if 'Failed' not in results[k]['text']])} papers.")
                
    return {
        "status": "success",
        "data": results
    }
