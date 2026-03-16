# Gemini Live Agent Challenge

ResearchAgent is a full-stack autonomous research assistant that can:
- browse and discover papers with a vision-driven browser agent,
- extract and analyze paper content,
- build a citation network,
- generate a literature review report.

It supports both:
- `Autopilot` mode: agent controls browser + pipeline end-to-end.
- `Copilot` mode: user drives UI while agent provides visual guidance.

## Architecture
- **Frontend**: React + Vite + Tailwind + D3 citation graph.
- **Backend**: FastAPI + WebSockets + Playwright/SeleniumBase + Google Gemini/ADK.
- **Pipeline**: SearchAgent -> ExtractionAgent -> SynthesisAgent -> CitationAgent -> ReportAgent.

High-level flow:
1. User sends research task over WebSocket.
2. Vision loop navigates web pages and records candidate paper URLs.
3. Paper discovery + extraction run in backend.
4. ADK pipeline synthesizes findings and builds graph.
5. Frontend receives `graph_update` and `report_update`.

## Computer Use Agent (How It Works)
This project uses Gemini's **Computer Use** pattern as the core browser-navigation agent.

### What makes it a Computer Use agent
- The backend declares a Computer Use tool in `run_vision_loop_computer_use`.
- Each loop turn sends a browser screenshot + task context to Gemini.
- Gemini returns **function/tool calls** (click/type/scroll/navigate/go_back/etc.).
- Backend maps those tool calls to Playwright actions and executes them.
- Backend sends `FunctionResponse` + updated screenshot back to Gemini for the next decision.

This creates the closed loop:
`Screenshot -> Gemini tool call -> Browser action -> Screenshot -> Gemini ...`

### Where this is implemented
- Main loop: `backend/core/vision_loop_computer_use.py`
- Browser execution: `backend/core/stealth_browser.py`
- Session orchestration + streaming: `backend/core/autopilot/autopilot_mode.py`

### Safety and anti-loop controls
- Action fingerprint loop detection (repeated same click/scroll patterns).
- Scroll progress / bottom detection on paper pages.
- Blocked-domain and direct-nav guards for unsupported/paywalled paths.
- Screenshot recovery logic for heavy pages.
- Conversation pruning to keep context under control.

### Hybrid Computer Use + In-memory analysis
The agent remains Computer Use for navigation/discovery, but can hand off long paper reading to backend extraction:
- Computer Use captures bounded evidence screenshots on paper pages.
- PDF/content extraction runs in backend memory for deeper analysis.
- Results flow through synthesis, citation graph, and report stages.

## Repository Structure
```text
.
├─ backend/
│  ├─ main.py
│  ├─ config.py
│  ├─ constants.py
│  ├─ agents/
│  ├─ core/
│  ├─ ws_handlers/
│  └─ requirements.txt
├─ frontend/
│  ├─ src/
│  ├─ package.json
│  └─ ...
└─ README.md
```

## Prerequisites
- Python 3.10+ (project currently uses local Python 3.14 in your workspace)
- Node.js 18+
- npm
- Google credentials (Gemini/Vertex, and optionally GCP services)

## Backend Setup
From repo root:

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
playwright install
```

Create `../.env` (or `backend/.env` depending on your run style) with the keys you use, for example:

```env
GOOGLE_API_KEY=...
GOOGLE_CLOUD_PROJECT_ID=...
GOOGLE_CLOUD_PROJECT=...
VERTEX_AI_LOCATION=global

USE_STEALTH_BROWSER=true
USE_COMPUTER_USE=true
USE_HYBRID_ANALYSIS=true
ENABLE_CONTEXT_CACHING=true

MAX_SCREENSHOTS_PER_PAPER=6
MAX_PAPERS_PER_RUN=15
MAX_GEMINI_CALLS_PER_RUN=200
```

Run backend:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Health check:
- `GET http://localhost:8000/health`

## Frontend Setup
From repo root:

```bash
cd frontend
npm install
npm run dev
```

Default Vite URL is typically:
- `http://localhost:5173`

## Main Backend Endpoints
- `GET /health` - service health.
- `GET /api/sessions` - list saved sessions.
- `GET /api/sessions/{session_id}` - fetch a specific session.
- `WS /ws/{session_id}` - primary bidirectional socket (copilot/autopilot modes).
- `WS /ws/agent` - direct autopilot agent endpoint.
- `WS /ws/research` - ADK coordinator research stream.

## Notes on Metadata Quality
If a paper card shows limited metadata, it usually means extraction fell back to a weaker path (`abstract_only` / `not_extracted`). The backend includes fallback metadata synthesis, but richest cards come from successful full extraction.

## Troubleshooting
- **429 / rate limits**: reduce `MAX_GEMINI_CALLS_PER_RUN`, `MAX_PAPERS_PER_RUN`, and screenshot budget.
- **Browser stuck on heavy pages**: keep stealth browser + hybrid analysis enabled.
- **Missing report/graph**: check backend logs for extraction failures before synthesis stage.
- **PowerShell profile warnings** in this workspace are local shell policy messages and generally unrelated to app logic.

## License
This project is licensed under the terms in [LICENSE](./LICENSE).

