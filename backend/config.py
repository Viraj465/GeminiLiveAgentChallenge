"""
Configuration file that loads environment variables and sets up constants for the ResearchAgent backend.
"""

from pydantic import BaseModel
from dotenv import load_dotenv
from constants import GOOGLE_MODELS, GOOGLE_COMPUTER_USE_MODEL
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
    GOOGLE_REASONING_MODEL: str = GOOGLE_MODELS[0]
    GOOGLE_REPORT_MODEL: str = GOOGLE_MODELS[5]
    GOOGLE_COMPUTER_USE_MODEL: str = GOOGLE_COMPUTER_USE_MODEL

    # Vertex AI
    VERTEX_AI_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    VERTEX_AI_LOCATION: str = os.getenv("VERTEX_AI_LOCATION", "global")

    # websocket
    WS_MAX_FRAME_SIZE: int = 1_000_000  # 1MB max per frame
    FRAME_SAMPLE_RATE: float = 1.0       # 1 FPS

    # Browser Configuration
    USE_STEALTH_BROWSER: bool = os.getenv("USE_STEALTH_BROWSER", "true").lower() == "true"
    BROWSER_HEADLESS: bool = os.getenv("BROWSER_HEADLESS", "false").lower() == "true"
    
    # Vision Loop Configuration
    USE_OPTIMIZED_VISION_LOOP: bool = os.getenv("USE_OPTIMIZED_VISION_LOOP", "true").lower() == "true"
    USE_COMPUTER_USE: bool = os.getenv("USE_COMPUTER_USE", "true").lower() == "true"
    MAX_ACTION_RETRIES: int = int(os.getenv("MAX_ACTION_RETRIES", "2"))  # 2-try strategy

    # Context Caching Configuration
    ENABLE_CONTEXT_CACHING: bool = os.getenv("ENABLE_CONTEXT_CACHING", "true").lower() == "true"
    CONTEXT_CACHE_TTL_SECONDS: int = int(os.getenv("CONTEXT_CACHE_TTL", "3600"))  # 1 hour default

    # Hybrid Vision + In-Memory Analysis Configuration
    USE_HYBRID_ANALYSIS: bool = os.getenv("USE_HYBRID_ANALYSIS", "true").lower() == "true"
    MAX_SCREENSHOTS_PER_PAPER: int = int(os.getenv("MAX_SCREENSHOTS_PER_PAPER", "6"))
    MAX_PAPERS_PER_RUN: int = int(os.getenv("MAX_PAPERS_PER_RUN", "15"))
    MAX_GEMINI_CALLS_PER_RUN: int = int(os.getenv("MAX_GEMINI_CALLS_PER_RUN", "200"))
    MAX_CHARS_PER_PAPER: int = int(os.getenv("MAX_CHARS_PER_PAPER", "80000"))

    # Authentication
    USE_AUTH: bool = os.getenv("USE_AUTH", "false").lower() == "true"

settings = Settings()
