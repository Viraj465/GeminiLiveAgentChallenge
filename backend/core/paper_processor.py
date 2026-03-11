"""
paper_processor.py — Two-tier PDF extraction pipeline.

Tier 1: PyMuPDF (fast, free, local) — extracts text, metadata, TOC, images, tables.
Tier 2: Document AI (GCP, OCR) — fallback for scanned or complex-layout PDFs.

All processing is in-memory. No disk writes. Papers are downloaded in async batches 
and released from memory immediately after extraction.
"""

import httpx
import asyncio
import logging
import re
import fitz  # PyMuPDF
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
#  Data Models
# ═══════════════════════════════════════════════

@dataclass
class PaperPayload:
    """Raw downloaded paper — lives only in memory."""
    url: str
    content: bytes
    content_type: str
    size_bytes: int

    def release(self):
        """Explicitly free memory."""
        self.content = b""
        self.size_bytes = 0


@dataclass
class ExtractedSection:
    """A single section from the paper."""
    heading: str
    level: int          # 1 = H1, 2 = H2, etc.
    text: str
    page_start: int
    page_end: int


@dataclass
class ExtractedTable:
    """A table found in the paper."""
    page: int
    rows: list[list[str]]
    caption: str = ""


@dataclass
class ExtractedPaper:
    """Fully extracted paper with all data."""
    url: str
    method: str                                     # "pymupdf" or "document_ai"
    status: str                                     # "success" or "error"
    size_kb: float = 0.0
    error: str = ""

    # ── Metadata ──
    title: str = ""
    authors: list[str] = field(default_factory=list)
    subject: str = ""
    keywords: list[str] = field(default_factory=list)
    creation_date: str = ""
    page_count: int = 0

    # ── Content ──
    full_text: str = ""
    abstract: str = ""
    sections: list[ExtractedSection] = field(default_factory=list)
    tables: list[ExtractedTable] = field(default_factory=list)
    references: list[str] = field(default_factory=list)

    # ── Table of Contents ──
    toc: list[dict] = field(default_factory=list)   # [{"level": 1, "title": "...", "page": 3}]

    def to_dict(self) -> dict:
        """Serialize for JSON response / downstream agents."""
        return {
            "url": self.url,
            "method": self.method,
            "status": self.status,
            "size_kb": round(self.size_kb, 2),
            "error": self.error,
            "metadata": {
                "title": self.title,
                "authors": self.authors,
                "subject": self.subject,
                "keywords": self.keywords,
                "creation_date": self.creation_date,
                "page_count": self.page_count,
            },
            "abstract": self.abstract,
            "full_text": self.full_text,
            "sections": [
                {
                    "heading": s.heading,
                    "level": s.level,
                    "text": s.text,
                    "page_start": s.page_start,
                    "page_end": s.page_end,
                }
                for s in self.sections
            ],
            "tables": [
                {"page": t.page, "rows": t.rows, "caption": t.caption}
                for t in self.tables
            ],
            "references": self.references,
            "toc": self.toc,
        }


# ═══════════════════════════════════════════════
#  Tier 1 — PyMuPDF (Local, Fast, Free)
# ═══════════════════════════════════════════════

def _parse_pdf_date(date_str: str) -> str:
    """Convert PDF date format (D:20240115...) to ISO string."""
    if not date_str:
        return ""
    try:
        clean = date_str.replace("D:", "").split("+")[0].split("-")[0]
        dt = datetime.strptime(clean[:14], "%Y%m%d%H%M%S")
        return dt.isoformat()
    except Exception:
        return date_str


def _extract_abstract(full_text: str) -> str:
    """Pull the abstract from the full text using common patterns."""
    patterns = [
        r"(?i)abstract\s*\n+(.*?)(?=\n\s*(?:1[\.\s]|introduction|keywords|I\.\s))",
        r"(?i)abstract[:\.\-—]\s*(.*?)(?=\n\s*(?:1[\.\s]|introduction|keywords))",
        r"(?i)abstract\s+(.*?)(?=\n\n)",
    ]
    for pattern in patterns:
        match = re.search(pattern, full_text, re.DOTALL)
        if match:
            abstract = match.group(1).strip()
            if len(abstract) > 50:  # sanity check
                return abstract
    return ""


def _extract_references(full_text: str) -> list[str]:
    """Extract the references section and split into individual citations."""
    # Find the references section
    ref_patterns = [
        r"(?i)(?:references|bibliography|works cited)\s*\n+(.*)",
        r"(?i)\n\s*references\s*\n(.*)",
    ]
    ref_text = ""
    for pattern in ref_patterns:
        match = re.search(pattern, full_text, re.DOTALL)
        if match:
            ref_text = match.group(1).strip()
            break

    if not ref_text:
        return []

    # Split references by numbered pattern [1], [2] etc. or 1. 2. etc.
    refs = re.split(r"\n\s*\[?\d+[\]\.]\s+", ref_text)
    # Clean and filter
    cleaned = []
    for ref in refs:
        ref = ref.strip().replace("\n", " ")
        ref = re.sub(r"\s+", " ", ref)
        if len(ref) > 20:  # skip empty/noise
            cleaned.append(ref)
    return cleaned


def _extract_sections_from_text(full_text: str, toc: list[dict]) -> list[ExtractedSection]:
    """Split full text into sections using the TOC or heading patterns."""
    sections = []

    if toc:
        # Use TOC entries to find section boundaries
        headings = [entry["title"] for entry in toc]
        for idx, heading in enumerate(headings):
            # Find the heading in the text
            pos = full_text.find(heading)
            if pos == -1:
                continue
            # Find the next heading to determine section boundary
            next_pos = len(full_text)
            if idx + 1 < len(headings):
                np = full_text.find(headings[idx + 1], pos + len(heading))
                if np != -1:
                    next_pos = np

            section_text = full_text[pos + len(heading):next_pos].strip()
            sections.append(ExtractedSection(
                heading=heading,
                level=toc[idx].get("level", 1),
                text=section_text,
                page_start=toc[idx].get("page", 0),
                page_end=toc[idx + 1].get("page", 0) if idx + 1 < len(toc) else 0,
            ))
    else:
        # Fallback: detect headings via common patterns
        heading_pattern = r"\n\s*((?:\d+\.?\s+)?[A-Z][A-Za-z\s&:,\-]{3,60})\s*\n"
        matches = list(re.finditer(heading_pattern, full_text))

        for idx, match in enumerate(matches):
            heading = match.group(1).strip()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)
            section_text = full_text[start:end].strip()

            if len(section_text) > 30:  # skip trivially small sections
                sections.append(ExtractedSection(
                    heading=heading,
                    level=1,
                    text=section_text,
                    page_start=0,
                    page_end=0,
                ))

    return sections


def extract_with_pymupdf(paper: PaperPayload) -> ExtractedPaper:
    """
    Tier 1: Full extraction using PyMuPDF.
    Extracts metadata, text, TOC, tables, abstract, sections, and references.
    """
    result = ExtractedPaper(
        url=paper.url,
        method="pymupdf",
        status="error",
        size_kb=paper.size_bytes / 1024,
    )

    try:
        doc = fitz.open(stream=paper.content, filetype="pdf")
    except Exception as e:
        result.error = f"Could not open PDF: {e}"
        logger.error(result.error)
        return result

    try:
        result.page_count = len(doc)

        # ── Metadata ──
        meta = doc.metadata or {}
        result.title = (meta.get("title") or "").strip()
        result.subject = (meta.get("subject") or "").strip()
        result.creation_date = _parse_pdf_date(meta.get("creationDate", ""))

        author_raw = (meta.get("author") or "").strip()
        if author_raw:
            # Split by common delimiters: comma, semicolon, " and "
            result.authors = [
                a.strip()
                for a in re.split(r"[;,]|\band\b", author_raw)
                if a.strip()
            ]

        keywords_raw = (meta.get("keywords") or "").strip()
        if keywords_raw:
            result.keywords = [k.strip() for k in re.split(r"[;,]", keywords_raw) if k.strip()]

        # ── Table of Contents ──
        raw_toc = doc.get_toc()  # [[level, title, page], ...]
        result.toc = [
            {"level": entry[0], "title": entry[1], "page": entry[2]}
            for entry in raw_toc
        ]

        # ── Full Text (page by page) ──
        page_texts = []
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            page_texts.append(text)
        result.full_text = "\n\n".join(page_texts).strip()

        # ── Tables (per page) ──
        for page_num, page in enumerate(doc):
            try:
                tabs = page.find_tables()
                for tab in tabs:
                    table_data = tab.extract()
                    if table_data and len(table_data) > 1:
                        result.tables.append(ExtractedTable(
                            page=page_num + 1,
                            rows=table_data,
                            caption="",
                        ))
            except Exception:
                pass  # find_tables() can fail on some page layouts

        # ── Abstract ──
        result.abstract = _extract_abstract(result.full_text)

        # ── References ──
        result.references = _extract_references(result.full_text)

        # ── Sections ──
        result.sections = _extract_sections_from_text(result.full_text, result.toc)

        # ── Infer title from first line if metadata had none ──
        if not result.title and result.full_text:
            first_line = result.full_text.split("\n")[0].strip()
            if 10 < len(first_line) < 200:
                result.title = first_line

        # ── Quality check ──
        if len(result.full_text) < 100:
            result.status = "low_quality"
            logger.warning(f"PyMuPDF extracted very little text from {paper.url} — flagging for Document AI")
        else:
            result.status = "success"
            logger.info(
                f"PyMuPDF extracted {len(result.full_text)} chars, "
                f"{len(result.sections)} sections, {len(result.tables)} tables "
                f"from {paper.url}"
            )

    except Exception as e:
        result.error = f"PyMuPDF extraction error: {e}"
        logger.error(result.error)
    finally:
        doc.close()

    return result


# ═══════════════════════════════════════════════
#  Tier 2 — Document AI (GCP, OCR Fallback)
# ═══════════════════════════════════════════════

async def extract_with_document_ai(paper: PaperPayload) -> ExtractedPaper:
    """
    Tier 2: Full extraction using Google Document AI.
    Used as fallback when PyMuPDF returns low/no text (scanned PDFs, complex layouts).
    """
    from google.cloud import documentai_v1 as documentai
    from config import settings

    result = ExtractedPaper(
        url=paper.url,
        method="document_ai",
        status="error",
        size_kb=paper.size_bytes / 1024,
    )

    try:
        client = documentai.DocumentProcessorServiceClient()
        resource_name = client.processor_path(
            settings.PROJECT_ID,
            settings.DOC_AI_LOCATION,
            settings.DOC_AI_PROCESSOR_ID,
        )

        raw_document = documentai.RawDocument(
            content=paper.content,
            mime_type="application/pdf",
        )
        request = documentai.ProcessRequest(
            name=resource_name,
            raw_document=raw_document,
        )

        # Document AI client is synchronous — run in thread to not block event loop
        response = await asyncio.to_thread(client.process_document, request=request)
        document = response.document

        # ── Full text ──
        result.full_text = document.text or ""
        result.page_count = len(document.pages)

        # ── Extract structured data from Document AI entities ──
        for entity in document.entities:
            etype = entity.type_.lower()
            value = entity.mention_text.strip()

            if "title" in etype:
                result.title = value
            elif "author" in etype:
                result.authors.append(value)
            elif "abstract" in etype:
                result.abstract = value
            elif "date" in etype:
                result.creation_date = value
            elif "keyword" in etype:
                result.keywords.append(value)

        # ── Extract tables from Document AI pages ──
        for page_idx, page in enumerate(document.pages):
            for table in page.tables:
                rows = []
                # Header rows
                for header_row in table.header_rows:
                    row_cells = []
                    for cell in header_row.cells:
                        cell_text = _extract_docai_text(cell.layout, document.text)
                        row_cells.append(cell_text)
                    rows.append(row_cells)
                # Body rows
                for body_row in table.body_rows:
                    row_cells = []
                    for cell in body_row.cells:
                        cell_text = _extract_docai_text(cell.layout, document.text)
                        row_cells.append(cell_text)
                    rows.append(row_cells)

                if rows:
                    result.tables.append(ExtractedTable(
                        page=page_idx + 1,
                        rows=rows,
                        caption="",
                    ))

        # ── Fallback: use regex extraction if entities didn't provide them ──
        if not result.abstract:
            result.abstract = _extract_abstract(result.full_text)
        if not result.references:
            result.references = _extract_references(result.full_text)
        if not result.sections:
            result.sections = _extract_sections_from_text(result.full_text, [])

        # ── Infer title ──
        if not result.title and result.full_text:
            first_line = result.full_text.split("\n")[0].strip()
            if 10 < len(first_line) < 200:
                result.title = first_line

        result.status = "success"
        logger.info(
            f"Document AI extracted {len(result.full_text)} chars, "
            f"{len(result.tables)} tables from {paper.url}"
        )

    except Exception as e:
        result.error = f"Document AI error: {e}"
        result.status = "error"
        logger.error(result.error)

    return result


def _extract_docai_text(layout, full_text: str) -> str:
    """Helper: extract text from a Document AI layout segment."""
    text = ""
    for segment in layout.text_anchor.text_segments:
        start = int(segment.start_index)
        end = int(segment.end_index)
        text += full_text[start:end]
    return text.strip()


# ═══════════════════════════════════════════════
#  Two-Tier Orchestrator
# ═══════════════════════════════════════════════

async def extract_paper_content(paper: PaperPayload) -> ExtractedPaper:
    """
    Two-tier extraction:
      1. Try PyMuPDF (fast, free, local)
      2. Fallback to Document AI if PyMuPDF returns low quality text
    """
    # ── Tier 1: PyMuPDF ──
    result = extract_with_pymupdf(paper)

    if result.status == "success":
        return result

    # ── Tier 2: Document AI fallback ──
    logger.info(f"Falling back to Document AI for {paper.url} (PyMuPDF status: {result.status})")
    try:
        docai_result = await extract_with_document_ai(paper)
        return docai_result
    except Exception as e:
        logger.error(f"Both extraction tiers failed for {paper.url}: {e}")
        # Return the PyMuPDF result even if low quality — better than nothing
        result.error = f"Both tiers failed. PyMuPDF: {result.error}. DocAI: {e}"
        return result


# ═══════════════════════════════════════════════
#  Download + Batch Processing
# ═══════════════════════════════════════════════

async def download_paper(
    client: httpx.AsyncClient,
    url: str,
    max_size_mb: float = 10.0,
) -> Optional[PaperPayload]:
    """
    Stream-download a PDF into memory.
    Aborts mid-stream if file exceeds max_size_mb.
    """
    try:
        async with client.stream("GET", url, follow_redirects=True) as resp:
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")

            chunks = []
            total = 0
            max_bytes = int(max_size_mb * 1024 * 1024)

            async for chunk in resp.aiter_bytes(chunk_size=65_536):
                total += len(chunk)
                if total > max_bytes:
                    logger.warning(f"Skipping {url} — exceeds {max_size_mb}MB limit")
                    return None
                chunks.append(chunk)

            content = b"".join(chunks)
            logger.info(f"Downloaded {url} — {total / 1024:.1f}KB")
            return PaperPayload(
                url=url,
                content=content,
                content_type=content_type,
                size_bytes=total,
            )
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP {e.response.status_code} downloading {url}")
        return None
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return None


async def process_papers_batch(
    urls: list[str],
    batch_size: int = 3,
    max_size_mb: float = 10.0,
    timeout: float = 30.0,
) -> list[dict]:
    """
    Download and extract papers in batches of `batch_size`.
    Each paper is held in memory only during its batch — then released.
    Returns a list of ExtractedPaper dicts ready for downstream agents.
    """
    results = []

    async with httpx.AsyncClient(timeout=timeout) as client:
        for i in range(0, len(urls), batch_size):
            batch_urls = urls[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            logger.info(f"Batch {batch_num}: downloading {len(batch_urls)} papers")

            # Download batch concurrently
            papers = await asyncio.gather(
                *[download_paper(client, url, max_size_mb) for url in batch_urls],
                return_exceptions=True,
            )

            # Extract each paper in this batch
            for paper in papers:
                if isinstance(paper, Exception):
                    logger.error(f"Download error: {paper}")
                    continue
                if paper is None:
                    continue

                try:
                    extracted = await extract_paper_content(paper)
                    results.append(extracted.to_dict())
                finally:
                    paper.release()

            logger.info(f"Batch {batch_num} complete — memory released")

    logger.info(f"All done — {len(results)} papers extracted from {len(urls)} URLs")
    return results
