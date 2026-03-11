"""
Configuration file that loads environment variables and sets up constants for the ResearchAgent backend.
"""

from pydantic import BaseModel
from dotenv import load_dotenv
from constants import GOOGLE_MODELS
import os

load_dotenv()

class Settings(BaseModel):
    PROJECT_ID: str = os.getenv("GOOGLE_CLOUD_PROJECT_ID","")
    GEMINI_API_KEY: str = os.getenv("GOOGLE_API_KEY","")
    CLOUD_STORAGE_BUCKET: str = os.getenv("CLOUD_STORAGE_BUCKET","")
    FIRESTORE_COLLECTION: str = os.getenv("FIRESTORE_COLLECTION","")

    # Document AI
    DOC_AI_PROCESSOR_ID: str = os.getenv("DOC_AI_PROCESSOR_ID", "")
    DOC_AI_LOCATION: str = os.getenv("DOC_AI_LOCATION", "us")

    GOOGLE_VISION_MODEL: str = GOOGLE_MODELS[2]
    GOOGLE_REASONING_MODEL: str = GOOGLE_MODELS[1]
    GOOGLE_REPORT_MODEL: str = GOOGLE_MODELS[0]

    # Vertex AI
    VERTEX_AI_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    VERTEX_AI_LOCATION: str = os.getenv("VERTEX_AI_LOCATION", "us-central1")

    # websocket
    WS_MAX_FRAME_SIZE: int = 1_000_000  # 1MB max per frame
    FRAME_SAMPLE_RATE: float = 1.0       # 1 FPS

settings = Settings()
