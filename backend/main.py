import json
import logging
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from ws_handlers.handler import websocket_endpoint
from core.browser import BrowserController
from core.stealth_browser import StealthBrowserController
from core.vision_loop import run_vision_loop
from core.vision_loop_optimized import run_vision_loop_optimized
from core.vision_loop_computer_use import run_vision_loop_computer_use
from config import settings

from agents.coordinator import coordinator
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
import sys
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


# ── Logging setup ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="ResearchAgent Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Lock down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session_service = InMemorySessionService()

# Health check
@app.get("/health")
async def health():
    return {"status": "ok", "project": settings.PROJECT_ID}

# Authentication Dependency
from fastapi import Depends, HTTPException, Security, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import firebase_admin
from firebase_admin import auth as firebase_auth

security = HTTPBearer()

if settings.USE_AUTH:
    firebase_admin.initialize_app()

def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    if not settings.USE_AUTH:
        return "anonymous_user"
    try:
        token = credentials.credentials
        decoded_token = firebase_auth.verify_id_token(token)
        return decoded_token['uid']
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

def get_ws_current_user(token: str = Query(None)):
    if not settings.USE_AUTH:
        return "anonymous_user"
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")
    try:
        decoded_token = firebase_auth.verify_id_token(token)
        return decoded_token['uid']
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

# Session History Routes
from core.db import get_all_sessions, get_session

@app.get("/api/sessions")
async def list_sessions(user_id: str = Depends(get_current_user)):
    """Returns a list of all research sessions."""
    sessions = await get_all_sessions(user_id)
    return {"sessions": sessions}

@app.get("/api/sessions/{session_id}")
async def fetch_session(session_id: str, user_id: str = Depends(get_current_user)):
    """Returns the full data for a specific session."""
    data = await get_session(session_id, user_id)
    return {"session": data}

# WebSocket route — bidirectional handler (copilot + autopilot via modes)
@app.websocket("/ws/{session_id}")
async def ws_route(websocket: WebSocket, session_id: str, user_id: str = Depends(get_ws_current_user)):
    # Store user_id on the websocket object for downstream use if needed
    websocket.scope["user_id"] = user_id
    await websocket_endpoint(websocket, session_id)

# WebSocket route — direct autopilot agent endpoint
@app.websocket("/ws/agent")
async def agent_websocket(websocket: WebSocket):
    await websocket.accept()
    
    # Choose browser based on configuration
    if settings.USE_STEALTH_BROWSER:
        logger.info("🔒 Using Stealth Browser (SeleniumBase + Playwright)")
        browser = StealthBrowserController()
        await browser.start(headless=settings.BROWSER_HEADLESS)
        # await browser.start()
    else:
        logger.info("Using Standard Browser (Playwright only)")
        browser = BrowserController()
        await browser.start()

    try:
        # Receive task from frontend
        data = await websocket.receive_text()
        payload = json.loads(data)
        task = payload.get("task", "")

        if not task:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Missing 'task' field in payload"
            }))
            return

        browser_mode = "Stealth Mode (CAPTCHA bypass enabled)" if settings.USE_STEALTH_BROWSER else "Standard Mode"
        await websocket.send_text(json.dumps({
            "type": "status",
            "message": f"Starting: {task} [{browser_mode}]"
        }))

        # Pre-navigate to Google to avoid blank screen loops
        await browser.page.goto("https://google.com", wait_until="networkidle")
        
        # Choose vision loop based on configuration
        if settings.USE_COMPUTER_USE:
            vision_loop = run_vision_loop_computer_use
            loop_type = "Computer Use (native tool calls)"
        elif settings.USE_OPTIMIZED_VISION_LOOP:
            vision_loop = run_vision_loop_optimized
            loop_type = "Optimized (2-try strategy)"
        else:
            vision_loop = run_vision_loop
            loop_type = "Standard"
        
        logger.info(f"Using {loop_type} vision loop")
        
        # Run vision loop, stream each action back
        async for action in vision_loop(browser, task):
            await websocket.send_text(json.dumps({
                "type": "action",
                "data": action
            }))

        await websocket.send_text(json.dumps({
            "type": "complete",
            "message": "Task finished"
        }))

    except Exception as e:
        logger.error(f"Agent WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Agent error: {str(e)}"
            }))
        except Exception:
            pass  # WebSocket may already be closed

    finally:
        await browser.close()
        try:
            await websocket.close()
        except Exception:
            pass

@app.websocket("/ws/research")
async def research_websocket(websocket: WebSocket):
    await websocket.accept()
    data = await websocket.receive_text()
    payload = json.loads(data)
    query = payload.get("query", "")
    
    await websocket.send_text(json.dumps({
        "type": "status", 
        "message": f"CoordinatorAgent starting research: {query}"
    }))
    
    # Run ADK agent
    runner = Runner(
        agent=coordinator,
        session_service=session_service,
        app_name="research-agent"
    )
    
    session = session_service.create_session(app_name="research-agent", user_id="user")
    
    total_prompt_tokens = 0
    total_candidates_tokens = 0
    total_tokens = 0

    async for event in runner.run_async(
        user_id="user",
        session_id=session.id,
        new_message=query
    ):
        event_str = str(event)
        
        # Heuristic to grab token usage from ADK tool returns safely without modifying coordinator
        # ADK tools (synthesis_agent, report_agent) return token_usage in their dict which becomes part of the event string
        try:
            if hasattr(event, 'model_dump'):
                event_dict = event.model_dump()
            elif hasattr(event, 'dict'):
                event_dict = event.dict()
            else:
                event_dict = {}
                
            # If the event contains our token_usage payload, accumulate it
            # Since the tools return a dict, and ADK wraps it, let's also send it as a separate websocket message
            # But the easiest way is to push the string event, and if token_usage is found, push a token_update
            
        except Exception:
            pass
            
        # Parse tokens from the string representation using regex as fallback
        import re
        usage_match = re.search(r"'token_usage':\s*\{'prompt_tokens':\s*(\d+),\s*'candidates_tokens':\s*(\d+),\s*'total_tokens':\s*(\d+)\}", event_str)
        if usage_match:
            total_prompt_tokens += int(usage_match.group(1))
            total_candidates_tokens += int(usage_match.group(2))
            total_tokens += int(usage_match.group(3))
            
            await websocket.send_text(json.dumps({
                "type": "token_update",
                "payload": {
                    "prompt_tokens": total_prompt_tokens,
                    "candidates_tokens": total_candidates_tokens,
                    "total_tokens": total_tokens
                }
            }))
            
        # Safely extract deeply nested tool returns (graph_data, report) from ADK events
        def find_key(d, target_key):
            if isinstance(d, dict):
                if target_key in d: return d[target_key]
                for k, v in d.items():
                    res = find_key(v, target_key)
                    if res is not None: return res
            elif isinstance(d, list):
                for item in d:
                    res = find_key(item, target_key)
                    if res is not None: return res
            return None

        try:
            e_dict = {}
            if hasattr(event, 'model_dump'): e_dict = event.model_dump()
            elif hasattr(event, 'dict'): e_dict = event.dict()
            
            graph_data = find_key(e_dict, 'graph_data')
            if graph_data:
                await websocket.send_text(json.dumps({
                    "type": "graph_update",
                    "payload": { "graph_data": graph_data }
                }))
                
            report_data = find_key(e_dict, 'report')
            if report_data:
                await websocket.send_text(json.dumps({
                    "type": "report_update",
                    "payload": { "report": report_data }
                }))
        except Exception as e:
            pass

        await websocket.send_text(json.dumps({
            "type": "agent_event",
            "data": event_str
        }))
    
    await websocket.send_text(json.dumps({
        "type": "complete",
        "payload": {
            "message": "Research complete.",
            "total_tokens": total_tokens
        }
    }))
    
    await websocket.close()
