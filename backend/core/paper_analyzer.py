import asyncio
import base64
import logging
import os
import re
from io import BytesIO
from typing import Any

import fitz
import httpx

from config import settings
from prompts import PAPER_CHUNK_SUMMARIZATION_PROMPT, PAPER_REDUCE_SUMMARIZATION_PROMPT, VISION_SCREENSHOT_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)

# ─── Configuration ───
MAX_CHARS_PER_PAPER = int(os.getenv("MAX_CHARS_PER_PAPER", "80000"))
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "12000"))
MAX_VISION_SCREENSHOTS = int(os.getenv("MAX_VISION_SCREENSHOTS_ANALYSIS", "4"))
PRIORITY_SECTIONS = [
    "abstract", "introduction", "related work", "background",
    "method", "methodology", "approach", "model", "architecture",
    "experiment", "results", "evaluation", "discussion",
    "conclusion", "conclusions", "summary",
    "references", "bibliography",
]
LOW_PRIORITY_SECTIONS = [
    "acknowledgment", "acknowledgement", "acknowledgments",
    "appendix", "supplementary", "supplemental",
    "author contributions", "funding", "data availability",
]



# PDF Download + Text Extraction


async def download_pdf_bytes(url: str) -> bytes | None:
    """Download PDF bytes into RAM. Returns None if not a valid PDF."""
    try:
        async with httpx.AsyncClient(timeout=25.0, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code == 200 and b"%PDF" in resp.content[:1024]:
                logger.info(f"Downloaded PDF: {url} ({len(resp.content):,} bytes)")
                return resp.content
    except Exception as e:
        logger.warning(f"PDF download failed for {url}: {e}")
    return None


def extract_sections_from_pdf(pdf_bytes: bytes) -> dict:
    """
    Extract structured content from PDF bytes using PyMuPDF.

    Returns:
        {
            "full_text": str,
            "sections": [{"heading": str, "text": str, "page": int, "priority": str}],
            "figures": [{"base64": str, "page": int, "index": int, "ext": str}],
            "tables": [{"data": list[list], "page": int, "index": int}],
            "page_count": int,
            "char_count": int,
            "citation_strings": [str],  # raw reference entries
        }
    """
    result = {
        "full_text": "",
        "sections": [],
        "figures": [],
        "tables": [],
        "page_count": 0,
        "char_count": 0,
        "citation_strings": [],
    }

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            result["page_count"] = len(doc)
            all_text_parts = []

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Extract text
                page_text = page.get_text("text")
                all_text_parts.append(page_text)

                # Extract images (limit to first 10 per page to avoid memory issues)
                try:
                    images = page.get_images(full=True)
                    for img_idx, img in enumerate(images[:10]):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        if base_image and base_image.get("image"):
                            img_b64 = base64.b64encode(base_image["image"]).decode("utf-8")
                            result["figures"].append({
                                "base64": img_b64,
                                "page": page_num + 1,
                                "index": img_idx,
                                "ext": base_image.get("ext", "png"),
                            })
                except Exception as e:
                    logger.debug(f"Image extraction failed on page {page_num + 1}: {e}")

                # Extract tables
                try:
                    tabs = page.find_tables()
                    for tab_idx, tab in enumerate(tabs.tables):
                        result["tables"].append({
                            "data": tab.extract(),
                            "page": page_num + 1,
                            "index": tab_idx,
                        })
                except Exception:
                    pass

            full_text = "\n\n".join(all_text_parts)
            result["full_text"] = full_text
            result["char_count"] = len(full_text)

            # Parse sections from the full text
            result["sections"] = _parse_sections(full_text)

            # Extract citation strings from the references section
            result["citation_strings"] = _extract_citation_strings(full_text)

    except Exception as e:
        logger.error(f"PyMuPDF extraction failed: {e}")

    return result


def _parse_sections(text: str) -> list[dict]:
    """
    Parse section headings and their content from extracted text.
    Uses heuristic heading detection (all-caps lines, numbered sections, etc.)
    """
    sections = []
    lines = text.split("\n")

    # Heading patterns
    heading_patterns = [
        # Numbered: "1. Introduction", "2.1 Related Work"
        re.compile(r"^\s*(\d+\.?\d*\.?\d*)\s+([A-Z][A-Za-z\s&:,\-]+)$"),
        # All-caps: "ABSTRACT", "INTRODUCTION"
        re.compile(r"^\s*([A-Z][A-Z\s&:,\-]{3,})$"),
        # Title case with colon: "Results:", "Discussion:"
        re.compile(r"^\s*([A-Z][a-z]+(?:\s+[A-Za-z]+)*)\s*:?\s*$"),
    ]

    current_heading = "Preamble"
    current_text_parts = []
    current_page_estimate = 1

    for line_idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            current_text_parts.append("")
            continue

        is_heading = False
        for pattern in heading_patterns:
            match = pattern.match(stripped)
            if match:
                # Save previous section
                if current_text_parts:
                    section_text = "\n".join(current_text_parts).strip()
                    if section_text:
                        priority = _classify_section_priority(current_heading)
                        sections.append({
                            "heading": current_heading,
                            "text": section_text,
                            "page": current_page_estimate,
                            "priority": priority,
                        })

                # Start new section
                groups = match.groups()
                current_heading = groups[-1].strip() if groups else stripped
                current_text_parts = []
                is_heading = True
                # Rough page estimate
                current_page_estimate = max(1, (line_idx * 50) // max(len(lines), 1) + 1)
                break

        if not is_heading:
            current_text_parts.append(stripped)

    # Save last section
    if current_text_parts:
        section_text = "\n".join(current_text_parts).strip()
        if section_text:
            priority = _classify_section_priority(current_heading)
            sections.append({
                "heading": current_heading,
                "text": section_text,
                "page": current_page_estimate,
                "priority": priority,
            })

    return sections


def _classify_section_priority(heading: str) -> str:
    """Classify a section heading as 'high', 'medium', or 'low' priority."""
    heading_lower = heading.lower().strip()
    for kw in PRIORITY_SECTIONS:
        if kw in heading_lower:
            return "high"
    for kw in LOW_PRIORITY_SECTIONS:
        if kw in heading_lower:
            return "low"
    return "medium"


def _extract_citation_strings(text: str) -> list[str]:
    """Extract individual reference entries from the references section."""
    # Find references section
    ref_patterns = [
        r"(?:^|\n)\s*(?:References?|REFERENCES?|Bibliography|BIBLIOGRAPHY)\s*\n",
    ]
    ref_start = -1
    for pat in ref_patterns:
        m = re.search(pat, text)
        if m and m.start() > len(text) * 0.5:
            ref_start = m.end()
            break

    if ref_start < 0:
        # Fallback: last 15% of text
        ref_start = int(len(text) * 0.85)

    ref_block = text[ref_start:]

    # Split into individual entries
    entry_pattern = re.compile(
        r"(?:^\s*\[?\d+\]?\.?\s+|\n\s*\[?\d+\]?\.?\s+)"
        r"(.+?)(?=\n\s*\[?\d+\]?\.?\s+|\Z)",
        re.DOTALL,
    )
    entries = entry_pattern.findall(ref_block[:15000])
    cleaned = [" ".join(e.split()) for e in entries if e.strip() and len(e.strip()) > 15]

    if not cleaned:
        # Fallback: split by newlines
        cleaned = [l.strip() for l in ref_block.split("\n") if len(l.strip()) > 20][:80]

    return cleaned



# Chunked Summarization (Map → Reduce → Final)


def chunk_text_by_sections(
    sections: list[dict],
    max_chunk_chars: int = MAX_CHUNK_CHARS,
) -> list[dict]:
    """
    Chunk paper sections into groups that fit within the token budget.
    Priority sections are always included; low-priority sections are deferred.

    Returns list of chunks: [{"text": str, "sections": [str], "priority": str}]
    """
    # Sort: high priority first, then medium, then low
    priority_order = {"high": 0, "medium": 1, "low": 2}
    sorted_sections = sorted(sections, key=lambda s: priority_order.get(s.get("priority", "medium"), 1))

    chunks = []
    current_chunk_text = ""
    current_chunk_sections = []
    current_priority = "high"
    total_chars = 0

    for section in sorted_sections:
        section_text = section.get("text", "")
        section_heading = section.get("heading", "Unknown")
        section_priority = section.get("priority", "medium")

        # Enforce per-paper character ceiling
        if total_chars + len(section_text) > MAX_CHARS_PER_PAPER:
            # Truncate this section to fit
            remaining = MAX_CHARS_PER_PAPER - total_chars
            if remaining > 200:
                section_text = section_text[:remaining] + "\n[... truncated ...]"
            else:
                break

        # Check if adding this section exceeds chunk limit
        if len(current_chunk_text) + len(section_text) > max_chunk_chars:
            # Save current chunk
            if current_chunk_text:
                chunks.append({
                    "text": current_chunk_text.strip(),
                    "sections": current_chunk_sections,
                    "priority": current_priority,
                })
            current_chunk_text = f"## {section_heading}\n{section_text}\n\n"
            current_chunk_sections = [section_heading]
            current_priority = section_priority
        else:
            current_chunk_text += f"## {section_heading}\n{section_text}\n\n"
            current_chunk_sections.append(section_heading)

        total_chars += len(section_text)

    # Save last chunk
    if current_chunk_text.strip():
        chunks.append({
            "text": current_chunk_text.strip(),
            "sections": current_chunk_sections,
            "priority": current_priority,
        })

    logger.info(f"Chunked paper into {len(chunks)} chunks ({total_chars:,} chars total)")
    return chunks


async def summarize_chunk(chunk_text: str, paper_title: str, query: str = "") -> str:
    """
    Summarize a single chunk of paper text using Gemini.
    This is the MAP step of map-reduce summarization.
    """
    from google import genai
    from google.genai import types

    client = _get_gemini_client()
    model_name = settings.GOOGLE_REASONING_MODEL

    prompt = PAPER_CHUNK_SUMMARIZATION_PROMPT.format(paper_title=paper_title, query=query) + f"\n\nTEXT:\n{chunk_text[:10000]}"

    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=2048,
                ),
            ),
            timeout=60.0,
        )
        return response.text.strip() if response.text else ""
    except Exception as e:
        logger.warning(f"Chunk summarization failed: {e}")
        # Fallback: return first 500 chars of the chunk
        return chunk_text[:500]


async def reduce_summaries(
    chunk_summaries: list[str],
    paper_title: str,
    query: str = "",
) -> str:
    """
    Reduce multiple chunk summaries into a single coherent paper summary.
    This is the REDUCE step of map-reduce summarization.
    """
    from google import genai
    from google.genai import types

    client = _get_gemini_client()
    model_name = settings.GOOGLE_REASONING_MODEL

    combined = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{s}" for i, s in enumerate(chunk_summaries) if s
    )

    prompt = PAPER_REDUCE_SUMMARIZATION_PROMPT.format(paper_title=paper_title, query=query) + f"\n\nSECTION SUMMARIES:\n{combined[:15000]}"

    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=4096,
                ),
            ),
            timeout=90.0,
        )
        return response.text.strip() if response.text else ""
    except Exception as e:
        logger.warning(f"Summary reduction failed: {e}")
        return "\n\n".join(chunk_summaries)


async def hierarchical_summarize(
    sections: list[dict],
    paper_title: str,
    query: str = "",
) -> str:
    """
    Full map-reduce summarization pipeline for a paper.

    1. Chunk sections by token budget
    2. MAP: Summarize each chunk independently
    3. REDUCE: Combine chunk summaries into final synthesis
    """
    chunks = chunk_text_by_sections(sections)

    if not chunks:
        return ""

    # If only 1-2 chunks, skip the map step and go straight to reduce
    if len(chunks) <= 2:
        combined_text = "\n\n".join(c["text"] for c in chunks)
        return await summarize_chunk(combined_text, paper_title, query)

    # MAP: Summarize each chunk concurrently (limit concurrency to 3)
    semaphore = asyncio.Semaphore(3)

    async def _bounded_summarize(chunk):
        async with semaphore:
            return await summarize_chunk(chunk["text"], paper_title, query)

    chunk_summaries = await asyncio.gather(
        *[_bounded_summarize(c) for c in chunks],
        return_exceptions=True,
    )

    # Filter out exceptions
    valid_summaries = [
        s for s in chunk_summaries
        if isinstance(s, str) and s.strip()
    ]

    if not valid_summaries:
        # Fallback: concatenate raw section text
        return "\n\n".join(c["text"][:2000] for c in chunks[:3])

    # REDUCE: Combine into final summary
    return await reduce_summaries(valid_summaries, paper_title, query)



# Multimodal Screenshot Analysis


async def analyze_vision_screenshots(
    screenshots: list[dict],
    paper_title: str,
) -> list[dict]:
    """
    Analyze selected vision screenshots captured during PAPER_CAPTURE mode.

    Each screenshot dict has: {"label": str, "jpeg_bytes": bytes}

    Returns list of analysis results:
    [{"label": str, "analysis": str, "key_findings": list[str]}]
    """
    if not screenshots:
        return []

    from google import genai
    from google.genai import types

    client = _get_gemini_client()
    model_name = settings.GOOGLE_VISION_MODEL

    selected = screenshots[:MAX_VISION_SCREENSHOTS]
    results = []

    for ss in selected:
        label = ss.get("label", "unknown")
        jpeg_bytes = ss.get("jpeg_bytes", b"")

        if not jpeg_bytes:
            continue

        prompt = VISION_SCREENSHOT_ANALYSIS_PROMPT.format(paper_title=paper_title, label=label)

        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Content(role="user", parts=[
                            types.Part.from_text(text=prompt),
                            types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                        ])
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=1024,
                    ),
                ),
                timeout=30.0,
            )

            analysis_text = response.text.strip() if response.text else ""
            results.append({
                "label": label,
                "analysis": analysis_text,
                "key_findings": _extract_key_findings(analysis_text),
            })

        except Exception as e:
            logger.warning(f"Screenshot analysis failed for '{label}': {e}")
            results.append({
                "label": label,
                "analysis": f"[Analysis failed: {e}]",
                "key_findings": [],
            })

    logger.info(f"Analyzed {len(results)} screenshots for '{paper_title[:40]}'")
    return results


def _extract_key_findings(analysis_text: str) -> list[str]:
    """Extract bullet-point key findings from analysis text."""
    findings = []
    # Look for lines with numbers (likely quantitative findings)
    for line in analysis_text.split("\n"):
        line = line.strip().lstrip("- •*")
        if line and re.search(r"\d+\.?\d*%?", line):
            findings.append(line.strip())
    return findings[:5]



# Full Paper Analysis Pipeline


async def analyze_paper_hybrid(
    paper_data: dict,
    query: str = "",
) -> dict:
    """
    Full hybrid analysis pipeline for a single paper.

    Combines:
      1. In-memory PDF extraction (text + sections + figures + tables + citations)
      2. Hierarchical summarization (map → reduce)
      3. Multimodal screenshot analysis (from PAPER_CAPTURE mode)

    Args:
        paper_data: Paper dict with at minimum 'url' and optionally
                    'vision_screenshots' from the capture phase.
        query: Research query for context.

    Returns:
        Enriched paper dict ready for the ADK pipeline.
    """
    url = paper_data.get("url", "")
    title = paper_data.get("title", "Unknown")
    vision_screenshots = paper_data.get("vision_screenshots", [])

    result = {
        "url": url,
        "title": title,
        "name": title,
        "authors": paper_data.get("authors", []),
        "year": paper_data.get("year"),
        "abstract": paper_data.get("snippet", paper_data.get("abstract", "")),
        "text": "",
        "char_count": 0,
        "status": "error",
        "extraction_method": "hybrid_analysis",
        # Provenance fields
        "methodology": {},
        "key_claims": [],
        "limitations": [],
        "figures_tables": [],
        "citations_in_text": [],
        "citation_strings": [],
        # Hybrid-specific fields
        "section_map": [],
        "hierarchical_summary": "",
        "vision_insights": [],
        "capture_metadata": paper_data.get("capture_metadata", {}),
        "cache_name": None,
        "gcs_uri": None,
    }

    # ── Step 1: Download and extract PDF ──
    pdf_bytes = None
    if url:
        pdf_bytes = await download_pdf_bytes(url)

    extracted = None
    if pdf_bytes:
        extracted = await asyncio.to_thread(extract_sections_from_pdf, pdf_bytes)
        del pdf_bytes  # Free RAM

    if extracted and extracted.get("char_count", 0) > 200:
        result["text"] = extracted["full_text"][:MAX_CHARS_PER_PAPER]
        result["char_count"] = len(result["text"])
        result["section_map"] = [
            {"heading": s["heading"], "priority": s["priority"], "page": s["page"]}
            for s in extracted.get("sections", [])
        ]
        result["citation_strings"] = extracted.get("citation_strings", [])
        result["figures_tables"] = [
            {
                "label": f"Figure p{f['page']}.{f['index']}",
                "type": "figure",
                "page": f["page"],
                "caption": "",
                "key_finding": "",
            }
            for f in extracted.get("figures", [])[:10]
        ] + [
            {
                "label": f"Table p{t['page']}.{t['index']}",
                "type": "table",
                "page": t["page"],
                "caption": "",
                "key_finding": "",
            }
            for t in extracted.get("tables", [])[:10]
        ]

        # ── Step 2: Hierarchical summarization ──
        sections = extracted.get("sections", [])
        if sections:
            try:
                summary = await hierarchical_summarize(sections, title, query)
                result["hierarchical_summary"] = summary
                # Append summary to text for downstream synthesis
                if summary:
                    result["text"] += f"\n\n### HIERARCHICAL SUMMARY ###\n{summary}"
                    result["char_count"] = len(result["text"])
            except Exception as e:
                logger.warning(f"Hierarchical summarization failed for '{title}': {e}")

        result["status"] = "success"
        result["extraction_method"] = "hybrid_analysis"

    elif result.get("abstract"):
        # Fallback: use abstract only
        result["text"] = result["abstract"]
        result["char_count"] = len(result["text"])
        result["status"] = "success"
        result["extraction_method"] = "abstract_only"

    # ── Step 3: Multimodal screenshot analysis ──
    if vision_screenshots:
        try:
            vision_insights = await analyze_vision_screenshots(vision_screenshots, title)
            result["vision_insights"] = vision_insights

            # Append vision insights to text
            if vision_insights:
                insights_text = "\n\n### VISUAL EVIDENCE (from browser screenshots) ###\n"
                for vi in vision_insights:
                    insights_text += f"\n**{vi['label']}**:\n{vi['analysis']}\n"
                    if vi.get("key_findings"):
                        for kf in vi["key_findings"]:
                            insights_text += f"  - {kf}\n"
                result["text"] += insights_text
                result["char_count"] = len(result["text"])

        except Exception as e:
            logger.warning(f"Vision screenshot analysis failed for '{title}': {e}")

    logger.info(
        f"Hybrid analysis complete for '{title[:40]}': "
        f"status={result['status']}, method={result['extraction_method']}, "
        f"{result['char_count']:,} chars, "
        f"{len(result.get('vision_insights', []))} vision insights"
    )

    return result


async def analyze_papers_batch(
    papers: list[dict],
    query: str = "",
    batch_size: int = 3,
) -> list[dict]:
    """
    Analyze multiple papers using the hybrid pipeline in batches.

    Args:
        papers: List of paper dicts (with optional vision_screenshots).
        query: Research query for context.
        batch_size: Number of papers to process concurrently.

    Returns:
        List of enriched paper dicts.
    """
    all_results = []

    for i in range(0, len(papers), batch_size):
        batch = papers[i:i + batch_size]
        logger.info(
            f"Analyzing batch {i // batch_size + 1} "
            f"({len(batch)} papers, {i + len(batch)}/{len(papers)} total)"
        )

        tasks = [analyze_paper_hybrid(p, query) for p in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, dict):
                all_results.append(r)
            else:
                logger.error(f"Paper analysis failed in batch: {r}")

        # Brief pause between batches
        if i + batch_size < len(papers):
            await asyncio.sleep(1)

    success_count = sum(1 for r in all_results if r.get("status") == "success")
    logger.info(f"Batch analysis complete: {success_count}/{len(papers)} papers succeeded")

    return all_results



# Gemini Client Helper


_gemini_client = None


def _get_gemini_client():
    """Get a Gemini client (lazy singleton)."""
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client

    from google import genai
    from dotenv import load_dotenv
    load_dotenv()

    project_id = settings.VERTEX_AI_PROJECT or settings.PROJECT_ID
    if not project_id:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    location = settings.VERTEX_AI_LOCATION or "global"

    if project_id:
        _gemini_client = genai.Client(vertexai=True, project=project_id, location=location)
        return _gemini_client

    api_key = settings.GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        _gemini_client = genai.Client(api_key=api_key)
        return _gemini_client

    raise EnvironmentError("No Gemini API credentials available for paper analysis")
