"""
db.py — Google Cloud Firestore Database Wrapper for Session Persistence.

Provides async/sync hybrid wrappers to save and retrieve generated research
outputs (like citation graphs and literature reviews) so they persist
across browser refreshes.
"""

import logging
from google.cloud import firestore

logger = logging.getLogger(__name__)

# Initialize the Firestore client
# Uses the active GOOGLE_APPLICATION_CREDENTIALS or gcloud auth application-default
try:
    db = firestore.AsyncClient()
    logger.info("Firestore AsyncClient successfully initialized.")
except Exception as e:
    logger.error(f"Failed to initialize Firestore: {e}")
    db = None

COLLECTION_NAME = "ResearchSessions"


async def save_session_data(session_id: str, data_key: str, data_value: any) -> bool:
    """
    ADK Tool: Saves generated data (e.g., 'graph' or 'report') to the session document.
    Must provide the active session_id, the name of the key to save, and the dictionary/string.
    Returns True if successful. 
    """
    if not db:
        logger.warning(f"Firestore not initialized. Cannot save {data_key} for {session_id}")
        return False
        
    try:
        doc_ref = db.collection(COLLECTION_NAME).document(session_id)
        
        # We use merge=True so we don't overwrite the existing document (like the graph)
        # when saving a new field (like the report).
        await doc_ref.set({
            data_key: data_value,
            "last_updated": firestore.SERVER_TIMESTAMP
        }, merge=True)
        
        logger.info(f"Successfully saved '{data_key}' for session {session_id} to Firestore.")
        return True
    
    except Exception as e:
        logger.error(f"Failed to save to Firestore for session {session_id}: {e}")
        return False


async def get_session(session_id: str) -> dict:
    """
    Retrieves the entire stored context for a session if the user refreshes the page.
    Returns the document dictionary or an empty dict if not found.
    """
    if not db:
        return {}
        
    try:
        doc_ref = db.collection(COLLECTION_NAME).document(session_id)
        doc = await doc_ref.get()
        
        if doc.exists:
            return doc.to_dict()
        return {}
        
    except Exception as e:
        logger.error(f"Failed to retrieve Firestore session {session_id}: {e}")
        return {}
