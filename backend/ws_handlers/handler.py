"""
handler.py — Production-grade bidirectional WebSocket handler.

Features:
  - JSON parse error handling
  - Frame size validation
  - Structured logging
  - Heartbeat / ping-pong (idle timeout detection)
  - Graceful cleanup via finally block
  - Copilot mode: frames → Gemini Vision → guidance downstream
  - Autopilot mode: headless browser → vision loop → actions downstream
  - Real mode dispatcher with per-session state
"""

from fastapi import WebSocket, WebSocketDisconnect
from ws_handlers.models import WSMessage, WSMessageType
from core.copilot.copilot_mode import analyze_frame, set_user_command, cleanup_session
# Import active_browsers registry to allow manual overrides
from core.autopilot.autopilot_mode import run_autopilot, active_browsers
from config import settings
import json
import time
import asyncio
import logging

logger = logging.getLogger(__name__)



#  Connection Manager


class ConnectionManager:
    """Manages active WebSocket connections per session."""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self._heartbeat_tasks: dict[str, asyncio.Task] = {}
        self._session_modes: dict[str, str] = {}           # "copilot" | "autopilot"
        self._session_commands: dict[str, str] = {}         # current user command
        self._autopilot_tasks: dict[str, asyncio.Task] = {} # running autopilot tasks
        self._autopilot_pause_events: dict[str, asyncio.Event] = {} # pause events for ask_user
        self._session_task_history: dict[str, list[str]] = {}  # ordered list of past tasks per session

    async def connect(self, session_id: str, websocket: WebSocket):
        """Accept and register a new WebSocket connection."""
        # Reject duplicate connections for the same session
        if session_id in self.active_connections:
            logger.warning(f"Duplicate connection attempt for session {session_id} — closing old one")
            await self.disconnect(session_id, reason="Replaced by new connection")

        await websocket.accept()
        self.active_connections[session_id] = websocket
        self._session_modes[session_id] = "autopilot"  # Default mode
        logger.info(f"Session {session_id} connected — {len(self.active_connections)} active")

    async def disconnect(self, session_id: str, reason: str = "Disconnected"):
        """Remove a connection and cancel its heartbeat + autopilot."""
        # Cancel heartbeat task
        task = self._heartbeat_tasks.pop(session_id, None)
        if task and not task.done():
            task.cancel()

        # Cancel autopilot task
        autopilot = self._autopilot_tasks.pop(session_id, None)
        if autopilot and not autopilot.done():
            autopilot.cancel()
            try:
                await asyncio.wait_for(autopilot, timeout=2.0)
            except asyncio.CancelledError:
                pass
            except asyncio.TimeoutError:
                logger.warning(f"Session {session_id}: autopilot cancel timed out during disconnect")
            except Exception as e:
                logger.warning(f"Session {session_id}: autopilot cancel raised during disconnect: {e}")
            
        self._autopilot_pause_events.pop(session_id, None)

        # Close any browser left open by a completed autopilot run.
        browser_ctrl = active_browsers.pop(session_id, None)
        if browser_ctrl:
            try:
                await browser_ctrl.close()
            except Exception as e:
                logger.warning(f"Session {session_id}: failed to close browser during disconnect: {e}")

        # Clean up session state
        self._session_modes.pop(session_id, None)
        self._session_commands.pop(session_id, None)
        self._session_task_history.pop(session_id, None)
        cleanup_session(session_id)

        ws = self.active_connections.pop(session_id, None)
        if ws:
            try:
                await ws.close(code=1000, reason=reason)
            except Exception:
                pass  # Connection may already be closed
        logger.info(f"Session {session_id} disconnected: {reason} — {len(self.active_connections)} active")

    def get_mode(self, session_id: str) -> str:
        """Get the current mode for a session."""
        return self._session_modes.get(session_id, "copilot")

    async def send_message(self, session_id: str, message: WSMessage):
        """Send a typed message to a specific session."""
        ws = self.active_connections.get(session_id)
        if not ws:
            logger.warning(f"Cannot send to session {session_id} — not connected")
            return
        try:
            await ws.send_json(message.model_dump())
        except Exception as e:
            logger.error(f"Failed to send to session {session_id}: {e}")
            await self.disconnect(session_id, reason="Send failed")

    async def send_error(self, session_id: str, error_msg: str):
        """Send a structured error message to a session."""
        await self.send_message(session_id, WSMessage(
            type=WSMessageType.ERROR,
            payload={"error": error_msg, "timestamp": time.time()},
            session_id=session_id,
        ))

    async def broadcast(self, message: WSMessage):
        """Send a message to all connected sessions."""
        disconnected = []
        for sid, ws in self.active_connections.items():
            try:
                await ws.send_json(message.model_dump())
            except Exception:
                disconnected.append(sid)
        # Clean up failed connections
        for sid in disconnected:
            await self.disconnect(sid, reason="Broadcast send failed")

    def start_heartbeat(self, session_id: str, interval: float = 30.0, timeout: float = 10.0):
        """Start a periodic heartbeat check for a session."""
        task = asyncio.create_task(
            self._heartbeat_loop(session_id, interval, timeout)
        )
        self._heartbeat_tasks[session_id] = task

    async def _heartbeat_loop(self, session_id: str, interval: float, timeout: float):
        """Send periodic pings and disconnect if no pong received."""
        try:
            while session_id in self.active_connections:
                await asyncio.sleep(interval)
                ws = self.active_connections.get(session_id)
                if not ws:
                    break
                try:
                    # Send a ping message over the application protocol
                    await ws.send_json({
                        "type": WSMessageType.PING,
                        "payload": {"timestamp": time.time()},
                    })
                    logger.debug(f"Heartbeat ping sent to session {session_id}")
                except Exception as e:
                    logger.warning(f"Heartbeat failed for session {session_id}: {e}")
                    await self.disconnect(session_id, reason="Heartbeat failed")
                    break
        except asyncio.CancelledError:
            pass  # Normal cancellation on disconnect


manager = ConnectionManager()



#  WebSocket Endpoint


async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    Main WebSocket handler with:
      - JSON error handling
      - Frame size validation
      - Heartbeat
      - Copilot / Autopilot mode dispatch
      - Graceful cleanup
    """
    await manager.connect(session_id, websocket)
    manager.start_heartbeat(session_id, interval=30.0)

    try:
        while True:
            # ── Receive raw data ──
            raw = await websocket.receive_text()

            # ── Frame size check ──
            if len(raw) > settings.WS_MAX_FRAME_SIZE:
                logger.warning(
                    f"Session {session_id}: frame too large "
                    f"({len(raw)} bytes > {settings.WS_MAX_FRAME_SIZE} limit)"
                )
                await manager.send_error(
                    session_id,
                    f"Frame exceeds maximum size of {settings.WS_MAX_FRAME_SIZE} bytes"
                )
                continue  # Drop the frame, keep connection alive

            # ── JSON parse ──
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                logger.warning(f"Session {session_id}: invalid JSON — {e}")
                await manager.send_error(session_id, f"Invalid JSON: {e}")
                continue

            # ── Validate message structure ──
            try:
                msg = WSMessage(**data)
            except Exception as e:
                logger.warning(f"Session {session_id}: invalid message structure — {e}")
                await manager.send_error(session_id, f"Invalid message format: {e}")
                continue

            # ── Route by message type ──
            logger.debug(f"Session {session_id}: received {msg.type}")

            if msg.type == WSMessageType.SCREEN_FRAME:
                await handle_screen_frame(session_id, msg)
            elif msg.type == WSMessageType.USER_COMMAND:
                await handle_user_command(session_id, msg)
            elif msg.type == WSMessageType.MODE_SWITCH:
                await handle_mode_switch(session_id, msg)
            elif msg.type == WSMessageType.USER_ACTION:
                await handle_user_action(session_id, msg)
            elif msg.type == WSMessageType.PONG:
                logger.debug(f"Session {session_id}: pong received")
            else:
                await manager.send_error(session_id, f"Unknown message type: {msg.type}")

    except WebSocketDisconnect:
        logger.info(f"Session {session_id}: client disconnected")
    except Exception as e:
        logger.error(f"Session {session_id}: unexpected error — {e}", exc_info=True)
    finally:
        # ── Guaranteed cleanup ──
        await manager.disconnect(session_id, reason="Handler exited")



#  Message Handlers


async def handle_screen_frame(session_id: str, msg: WSMessage):
    """Process incoming screenshot frame — route to copilot vision analysis."""
    frame_data = msg.payload.get("frame", "")
    if not frame_data:
        await manager.send_error(session_id, "screen_frame payload missing 'frame' field")
        return

    mode = manager.get_mode(session_id)

    if mode != "copilot":
        # In autopilot mode, we don't process user screen frames
        logger.debug(f"Session {session_id}: frame ignored (mode={mode})")
        return

    # Send frame to Copilot for Gemini Vision analysis
    logger.info(f"Session {session_id}: analyzing frame ({len(frame_data)} chars)")

    result = await analyze_frame(session_id, frame_data)

    if result.get("status") == "success":
        await manager.send_message(session_id, WSMessage(
            type=WSMessageType.GUIDANCE,
            payload={
                "guidance": result.get("guidance", ""),
                "copilot_status": result.get("copilot_status", "guiding"),
                "timestamp": time.time(),
            },
        ))
    else:
        await manager.send_message(session_id, WSMessage(
            type=WSMessageType.LOG_UPDATE,
            payload={
                "log": f"Copilot: {result.get('message', 'Analysis failed')}",
                "timestamp": time.time(),
            },
        ))


async def handle_user_command(session_id: str, msg: WSMessage):
    """Process a user text command — store it for copilot/autopilot use."""
    command = msg.payload.get("command", "").strip()
    if not command:
        await manager.send_error(session_id, "user_command payload missing 'command' field")
        return

    logger.info(f"Session {session_id}: command — '{command}'")
    manager._session_commands[session_id] = command

    mode = manager.get_mode(session_id)

    if mode == "copilot":
        # Update copilot's active command
        set_user_command(session_id, command)
        await manager.send_message(session_id, WSMessage(
            type=WSMessageType.LOG_UPDATE,
            payload={"log": f"Command received: {command}", "timestamp": time.time()},
        ))

    elif mode == "autopilot":
        if command.lower() == "done":
            pause_event = manager._autopilot_pause_events.get(session_id)
            if pause_event and not pause_event.is_set():
                pause_event.set()
                await manager.send_message(session_id, WSMessage(
                    type=WSMessageType.LOG_UPDATE,
                    payload={"log": "Resuming autopilot task...", "timestamp": time.time()},
                ))
            else:
                await manager.send_message(session_id, WSMessage(
                    type=WSMessageType.LOG_UPDATE,
                    payload={"log": "Autopilot is not paused right now.", "timestamp": time.time()},
                ))
            return

        # Launch autopilot task for this command
        ws = manager.active_connections.get(session_id)
        if ws:
            # Cancel any existing autopilot task
            existing = manager._autopilot_tasks.get(session_id)
            if existing and not existing.done():
                existing.cancel()
                await asyncio.sleep(0.1)

            # ── Conversation memory: accumulate task history per session ──
            # Each new command is appended to the session's task history so that
            # run_autopilot can pass the full context to the vision loop, allowing
            # Gemini to understand what was already researched and continue from there.
            history = manager._session_task_history.setdefault(session_id, [])
            history.append(command)
            prior_tasks = history[:-1]  # All tasks before the current one

            # Start new autopilot task
            pause_event = asyncio.Event()
            manager._autopilot_pause_events[session_id] = pause_event
            # Pass session_id and prior_tasks to run_autopilot so it can register
            # the browser and inject conversation context into the vision loop.
            task = asyncio.create_task(
                run_autopilot(
                    session_id,
                    command,
                    ws,
                    pause_event=pause_event,
                    prior_tasks=prior_tasks,
                )
            )
            manager._autopilot_tasks[session_id] = task

            if prior_tasks:
                await manager.send_message(session_id, WSMessage(
                    type=WSMessageType.LOG_UPDATE,
                    payload={
                        "log": (
                            f"Continuing research session (query {len(history)}): {command}\n"
                            f"Previous queries: {'; '.join(prior_tasks)}"
                        ),
                        "timestamp": time.time(),
                    },
                ))
            else:
                await manager.send_message(session_id, WSMessage(
                    type=WSMessageType.LOG_UPDATE,
                    payload={"log": f"Autopilot started: {command}", "timestamp": time.time()},
                ))


async def handle_mode_switch(session_id: str, msg: WSMessage):
    """Switch between copilot and autopilot mode — real dispatcher."""
    mode = msg.payload.get("mode", "").strip().lower()
    if mode not in ("copilot", "autopilot"):
        await manager.send_error(session_id, "mode must be 'copilot' or 'autopilot'")
        return

    old_mode = manager.get_mode(session_id)

    if mode == old_mode:
        await manager.send_message(session_id, WSMessage(
            type=WSMessageType.LOG_UPDATE,
            payload={"log": f"Already in {mode} mode", "timestamp": time.time()},
        ))
        return

    # If switching FROM autopilot, cancel any running autopilot task
    if old_mode == "autopilot":
        existing = manager._autopilot_tasks.pop(session_id, None)
        if existing and not existing.done():
            existing.cancel()
            logger.info(f"Session {session_id}: cancelled autopilot task")

    # Update mode
    manager._session_modes[session_id] = mode
    logger.info(f"Session {session_id}: switched {old_mode} → {mode}")

    await manager.send_message(session_id, WSMessage(
        type=WSMessageType.AGENT_ACTION,
        payload={
            "action": f"Switched to {mode} mode",
            "mode": mode,
            "previous_mode": old_mode,
            "timestamp": time.time(),
        },
    ))


async def handle_user_action(session_id: str, msg: WSMessage):
    """Process a manual user action (like a mouse click) from the frontend and forward to BrowserController."""
    action = msg.payload.get("action")
    x = msg.payload.get("x")
    y = msg.payload.get("y")

    if not action:
        await manager.send_error(session_id, "user_action payload missing 'action'")
        return

    mode = manager.get_mode(session_id)

    if mode != "autopilot":
        logger.debug(f"Session {session_id}: user_action ignored (mode={mode})")
        return

    logger.info(f"Session {session_id}: manual user action '{action}' at [{x}, {y}]")

    # ------------------------------------------------------------------
    # Trigger the Playwright Controller
    # ------------------------------------------------------------------
    browser_ctrl = active_browsers.get(session_id)

    if browser_ctrl and browser_ctrl.page and action == "click" and x is not None and y is not None:
        try:
            # Playwright handles this safely in the background
            await browser_ctrl.page.mouse.click(x, y)
            
            await manager.send_message(session_id, WSMessage(
                type=WSMessageType.LOG_UPDATE,
                payload={
                    "log": f"Executed manual click at [{x}, {y}]",
                    "timestamp": time.time(),
                },
            ))
        except Exception as e:
            logger.error(f"Failed to execute manual click: {e}")
            await manager.send_error(session_id, f"Click failed: {e}")
    else:
        await manager.send_message(session_id, WSMessage(
            type=WSMessageType.LOG_UPDATE,
            payload={
                "log": "No active browser session to execute click.",
                "timestamp": time.time(),
            },
        ))
