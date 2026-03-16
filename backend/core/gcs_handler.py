"""
gcs_handler.py — In-memory PDF → Google Cloud Storage transfer.

Zero disk I/O: PDF bytes flow from RAM directly into GCS.
GCS acts as a virtual memory buffer so Gemini Context Caching API
can reference the document via a gs:// URI.

Flow:
  fetch_pdf_to_gcs(url)  →  gs://bucket/papers/<uuid>.pdf
  delete_gcs_object(uri) →  cleanup after cache is deleted
"""

import asyncio
import logging
import uuid
from io import BytesIO

import httpx
from google.cloud import storage

from config import settings

logger = logging.getLogger(__name__)

# ─── GCS client (lazy singleton) ───
_gcs_client: storage.Client | None = None


def _get_gcs_client() -> storage.Client:
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = storage.Client(project=settings.PROJECT_ID or settings.VERTEX_AI_PROJECT)
    return _gcs_client


async def fetch_pdf_to_gcs(url: str, bucket_name: str | None = None) -> str | None:
    """
    Download a PDF from `url` into RAM, then stream it directly to GCS.

    Args:
        url:         Direct URL to the PDF (must return application/pdf or raw PDF bytes).
        bucket_name: GCS bucket name. Defaults to settings.CLOUD_STORAGE_BUCKET.

    Returns:
        GCS URI string  e.g. "gs://my-bucket/papers/abc123.pdf"
        None if download or upload fails.
    """
    bucket_name = bucket_name or settings.CLOUD_STORAGE_BUCKET
    if not bucket_name:
        logger.error("CLOUD_STORAGE_BUCKET is not configured — cannot upload PDF to GCS.")
        return None

    # ── Step 1: Download PDF bytes into RAM ──
    pdf_bytes: bytes | None = None
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code == 200 and (
                b"%PDF" in resp.content[:1024]
                or "application/pdf" in resp.headers.get("content-type", "")
            ):
                pdf_bytes = resp.content
                logger.info(f"Downloaded PDF from {url} ({len(pdf_bytes):,} bytes)")
            else:
                logger.warning(
                    f"URL {url} returned status {resp.status_code} "
                    f"or non-PDF content-type: {resp.headers.get('content-type', 'unknown')}"
                )
    except Exception as e:
        logger.warning(f"PDF download failed for {url}: {e}")
        return None

    if not pdf_bytes:
        return None

    # ── Step 2: Upload bytes from RAM directly to GCS (no disk write) ──
    object_name = f"papers/{uuid.uuid4().hex}.pdf"
    try:
        gcs = _get_gcs_client()
        bucket = gcs.bucket(bucket_name)
        blob = bucket.blob(object_name)

        # upload_from_file with BytesIO — zero disk I/O
        pdf_stream = BytesIO(pdf_bytes)
        await asyncio.to_thread(
            blob.upload_from_file,
            pdf_stream,
            content_type="application/pdf",
        )

        gcs_uri = f"gs://{bucket_name}/{object_name}"
        logger.info(f"Uploaded PDF to GCS: {gcs_uri}")
        return gcs_uri

    except Exception as e:
        logger.error(f"GCS upload failed for {url}: {e}")
        return None
    finally:
        del pdf_bytes  # Free RAM immediately


async def delete_gcs_object(gcs_uri: str) -> bool:
    """
    Delete a GCS object by its gs:// URI.

    Args:
        gcs_uri: e.g. "gs://my-bucket/papers/abc123.pdf"

    Returns:
        True if deleted successfully, False otherwise.
    """
    if not gcs_uri or not gcs_uri.startswith("gs://"):
        return False

    try:
        # Parse bucket and object name from URI
        without_prefix = gcs_uri[len("gs://"):]
        bucket_name, _, object_name = without_prefix.partition("/")

        gcs = _get_gcs_client()
        bucket = gcs.bucket(bucket_name)
        blob = bucket.blob(object_name)

        await asyncio.to_thread(blob.delete)
        logger.info(f"Deleted GCS object: {gcs_uri}")
        return True

    except Exception as e:
        logger.warning(f"Failed to delete GCS object {gcs_uri}: {e}")
        return False
