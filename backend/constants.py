"""
constants.py — Status codes, messages, and enumerations for the ResearchAgent backend.
"""

# ──────────────────────────────────────────────
# Google Models
# ──────────────────────────────────────────────
GOOGLE_MODELS = [
    "gemini-3-pro",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-flash",
    "gemini-3.1-flash-lite",
]


# ──────────────────────────────────────────────
# HTTP Status Codes & Messages
# ──────────────────────────────────────────────
class HTTPStatus:
    # 2xx — Success
    OK                    = (200, "Request successful.")
    CREATED               = (201, "Resource created successfully.")
    ACCEPTED              = (202, "Request accepted for processing.")
    NO_CONTENT            = (204, "No content to return.")

    # 4xx — Client Errors
    BAD_REQUEST           = (400, "Invalid request payload.")
    UNAUTHORIZED          = (401, "Authentication required.")
    FORBIDDEN             = (403, "You do not have permission to perform this action.")
    NOT_FOUND             = (404, "Requested resource not found.")
    METHOD_NOT_ALLOWED    = (405, "HTTP method not allowed on this endpoint.")
    CONFLICT              = (409, "Resource conflict — duplicate or version mismatch.")
    PAYLOAD_TOO_LARGE     = (413, "Payload exceeds the maximum allowed size.")
    UNPROCESSABLE_ENTITY  = (422, "Request is well-formed but contains semantic errors.")
    RATE_LIMITED           = (429, "Too many requests — rate limit exceeded.")

    # 5xx — Server Errors
    INTERNAL_SERVER_ERROR = (500, "An unexpected server error occurred.")
    NOT_IMPLEMENTED       = (501, "This feature is not yet implemented.")
    BAD_GATEWAY           = (502, "Bad gateway — upstream service unreachable.")
    SERVICE_UNAVAILABLE   = (503, "Service temporarily unavailable.")
    GATEWAY_TIMEOUT       = (504, "Upstream service timed out.")


# ──────────────────────────────────────────────
# WebSocket Close Codes & Messages
# ──────────────────────────────────────────────
class WSCloseCode:
    NORMAL_CLOSURE        = (1000, "Connection closed normally.")
    GOING_AWAY            = (1001, "Server is shutting down or client navigated away.")
    PROTOCOL_ERROR        = (1002, "WebSocket protocol violation.")
    UNSUPPORTED_DATA      = (1003, "Received unsupported data type.")
    INVALID_PAYLOAD       = (1007, "Malformed message payload.")
    POLICY_VIOLATION      = (1008, "Message violates server policy.")
    MESSAGE_TOO_BIG       = (1009, "Message exceeds maximum frame size.")
    INTERNAL_ERROR        = (1011, "Unexpected server-side WebSocket error.")
    SESSION_EXPIRED       = (4000, "Session has expired — please reconnect.")
    AUTH_FAILED           = (4001, "WebSocket authentication failed.")
    SESSION_NOT_FOUND     = (4004, "Session ID not found on server.")
    DUPLICATE_CONNECTION  = (4009, "Another connection already exists for this session.")
    IDLE_TIMEOUT          = (4010, "Connection closed due to inactivity.")


# ──────────────────────────────────────────────
# WebSocket Message Types
# ──────────────────────────────────────────────
class WSMessageType:
    # Upstream  (Client → Server)
    SCREEN_FRAME   = "screen_frame"
    USER_COMMAND   = "user_command"
    MODE_SWITCH    = "mode_switch"
    PING           = "ping"

    # Downstream (Server → Client)
    AGENT_ACTION   = "agent_action"
    GUIDANCE       = "guidance"
    LOG_UPDATE     = "log_update"
    GRAPH_UPDATE   = "graph_update"
    REPORT_CHUNK   = "report_chunk"
    ERROR          = "error"
    PONG           = "pong"


# ──────────────────────────────────────────────
# Agent Mode
# ──────────────────────────────────────────────
class AgentMode:
    COPILOT   = "copilot"
    AUTOPILOT = "autopilot"
    IDLE      = "idle"


# ──────────────────────────────────────────────
# Agent Pipeline Status (ADK Agents)
# ──────────────────────────────────────────────
class AgentStatus:
    IDLE           = ("idle",        "Agent is idle.")
    QUEUED         = ("queued",      "Task is queued for processing.")
    RUNNING        = ("running",     "Agent is actively processing.")
    WAITING        = ("waiting",     "Waiting for upstream input.")
    SUCCESS        = ("success",     "Agent completed successfully.")
    PARTIAL        = ("partial",     "Agent returned partial results.")
    FAILED         = ("failed",      "Agent encountered an error.")
    CANCELLED      = ("cancelled",   "Task was cancelled by the user.")
    TIMEOUT        = ("timeout",     "Agent timed out.")


# ──────────────────────────────────────────────
# Individual ADK Agent Identifiers
# ──────────────────────────────────────────────
class AgentID:
    COORDINATOR = "coordinator_agent"
    SEARCH      = "search_agent"
    EXTRACTION  = "extraction_agent"
    SYNTHESIS   = "synthesis_agent"
    CITATION    = "citation_agent"
    REPORT      = "report_agent"


# ──────────────────────────────────────────────
# Copilot / Vision Statuses
# ──────────────────────────────────────────────
class CopilotStatus:
    FRAME_RECEIVED       = ("frame_received",       "Screen frame received.")
    ANALYZING            = ("analyzing",            "Analyzing screen with Gemini Vision.")
    GUIDANCE_READY       = ("guidance_ready",       "Guidance generated — sending to client.")
    VISION_ERROR         = ("vision_error",         "Vision model returned an error.")
    FRAME_DECODE_ERROR   = ("frame_decode_error",   "Could not decode the incoming frame.")
    NO_ACTIONABLE_CONTENT = ("no_actionable_content","No actionable content detected on screen.")


# ──────────────────────────────────────────────
# Autopilot / Playwright Statuses
# ──────────────────────────────────────────────
class AutopilotStatus:
    BROWSER_LAUNCHING = ("browser_launching", "Launching headless browser.")
    BROWSER_READY     = ("browser_ready",     "Browser is ready for actions.")
    NAVIGATING        = ("navigating",        "Navigating to URL.")
    CLICKING          = ("clicking",          "Clicking on element.")
    TYPING            = ("typing",            "Typing into input field.")
    SCROLLING         = ("scrolling",         "Scrolling the page.")
    SCREENSHOTTING    = ("screenshotting",     "Capturing internal screenshot.")
    BROWSER_ERROR     = ("browser_error",     "Browser action failed.")
    BROWSER_CLOSED    = ("browser_closed",    "Browser session ended.")


# ──────────────────────────────────────────────
# Document Processing (Document AI / PDF)
# ──────────────────────────────────────────────
class DocumentStatus:
    UPLOADING          = ("uploading",          "Uploading document to Cloud Storage.")
    UPLOAD_COMPLETE    = ("upload_complete",     "Document uploaded successfully.")
    UPLOAD_FAILED      = ("upload_failed",       "Document upload failed.")
    PARSING            = ("parsing",            "Parsing document with Document AI.")
    PARSE_COMPLETE     = ("parse_complete",      "Document parsed successfully.")
    PARSE_FAILED       = ("parse_failed",        "Document parsing failed.")
    UNSUPPORTED_FORMAT = ("unsupported_format",  "Document format is not supported.")
    FILE_TOO_LARGE     = ("file_too_large",      "Document exceeds maximum file size.")


# ──────────────────────────────────────────────
# Firestore / Citation Graph
# ──────────────────────────────────────────────
class GraphStatus:
    NODE_ADDED         = ("node_added",          "Citation node added to graph.")
    EDGE_ADDED         = ("edge_added",          "Citation edge added to graph.")
    GRAPH_UPDATED      = ("graph_updated",       "Citation graph updated in Firestore.")
    GRAPH_FETCH_ERROR  = ("graph_fetch_error",   "Failed to fetch graph from Firestore.")
    DUPLICATE_NODE     = ("duplicate_node",      "Node already exists in the graph.")


# ──────────────────────────────────────────────
# Report Generation
# ──────────────────────────────────────────────
class ReportStatus:
    GENERATING         = ("generating",          "Generating literature review with Gemini 2.5 Pro.")
    STREAMING          = ("streaming",           "Streaming report chunks to client.")
    COMPLETE           = ("complete",            "Literature review generated successfully.")
    CONTEXT_OVERFLOW   = ("context_overflow",    "Input context exceeds model limit.")
    GENERATION_FAILED  = ("generation_failed",   "Report generation failed.")


# ──────────────────────────────────────────────
# Generic Error Messages
# ──────────────────────────────────────────────
class ErrorMessage:
    INVALID_JSON          = "Invalid JSON in request body."
    MISSING_FIELD         = "Required field '{}' is missing."
    INVALID_SESSION       = "Session ID is invalid or expired."
    INVALID_MODE          = "Mode must be 'copilot' or 'autopilot'."
    GCP_AUTH_ERROR        = "Google Cloud authentication failed — check credentials."
    GEMINI_API_ERROR      = "Gemini API returned an error: {}"
    FIRESTORE_WRITE_ERROR = "Failed to write to Firestore: {}"
    GCS_UPLOAD_ERROR      = "Failed to upload to Cloud Storage: {}"
    DOCAI_ERROR           = "Document AI processing error: {}"
    PLAYWRIGHT_ERROR      = "Playwright browser error: {}"
    WS_CONNECTION_LOST    = "WebSocket connection lost unexpectedly."
    RATE_LIMIT_HIT        = "API rate limit reached — retrying in {} seconds."
    MAX_RETRIES_EXCEEDED  = "Maximum retry attempts exceeded."
