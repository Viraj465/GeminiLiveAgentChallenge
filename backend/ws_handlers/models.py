from pydantic import BaseModel
from typing import Optional, Literal, Any
from enum import Enum

class WSMessageType(str, Enum):
    # Upstream (Client → Server)
    SCREEN_FRAME = "screen_frame"        # base64 screenshot
    USER_COMMAND = "user_command"         # text command from user
    MODE_SWITCH = "mode_switch"          # copilot ↔ autopilot
    PONG = "pong"                        # heartbeat response from client
    USER_ACTION = "user_action"          # frontend browser preview user action (e.g., click)

    # Downstream (Server → Client)
    AGENT_ACTION = "agent_action"        # what agent is doing
    GUIDANCE = "guidance"                # copilot guidance text
    LOG_UPDATE = "log_update"            # real-time logs
    GRAPH_UPDATE = "graph_update"        # citation graph data
    REPORT_CHUNK = "report_chunk"        # streaming report output
    BROWSER_FRAME = "browser_frame"      # live Playwright screenshot (base64)
    PING = "ping"                        # heartbeat ping from server
    ERROR = "error"
    CACHE_STATUS = "cache_status"        # context cache lifecycle updates
    GCS_STATUS = "gcs_status"           # GCS upload/delete updates

class WSMessage(BaseModel):
    type: WSMessageType
    payload: dict = {}
    session_id: Optional[str] = None
    timestamp: Optional[float] = None