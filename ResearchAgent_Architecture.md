# ResearchAgent System Documentation
**Google Gemini Hackathon | UI Navigator ☸️ Track**

## 1. System Overview

ResearchAgent is an autonomous academic research system orchestrated by Google's Agent Development Kit (ADK). It solves the problem of manual literature review by autonomously finding, reading, cross-referencing, and summarizing academic papers into a comprehensive report.

The system is designed to comply with the Hackathon's "UI Navigator" track requirements by utilizing Gemini 2.0 Flash to visually interpret browser screens and output executable coordinate actions.

### 1.1 Operating Modes
ResearchAgent operates in two distinct modes:
1. **Autopilot Mode**: The agent autonomously controls a hidden, headless Playwright browser to conduct research via the ADK Coordinator.
2. **Copilot Mode**: The user shares their screen (`getDisplayMedia` API), and the agent provides real-time, visual-based guidance overlaid on the user's screen regarding what actions they should take next.

---

## 2. Core Architecture

The architecture prioritizes a **CAPTCHA-Free strategy** while adhering to Cloud Run memory constraints. It decouples the "Search" phase from the "Read" phase.

### Diagram
```text
React Frontend (SaaS)
      │
      │ WebSocket
      ▼
FastAPI Backend (Cloud Run)
      │
      ▼
ADK CoordinatorAgent (gemini-2.5-flash)
  │
  ├── SearchAgent      → 6 APIs simultaneously (zero browser)
  ├── ExtractionAgent  → PDF bytes in RAM (zero disk) + Visual Browser Fallback
  ├── SynthesisAgent   → Gemini 2.5 Pro (1M context)
  ├── CitationAgent    → Firestore graph builder
  └── ReportAgent      → Gemini 2.5 Pro (markdown report)
```

---

## 3. The Agents (ADK FunctionTools)

All specialized sub-agents are registered as `FunctionTool` inputs to the `CoordinatorAgent`.

### 3.1 `SearchAgent`
**Goal**: Find academic paper URLs without triggering CAPTCHAs.
**How it works**:
It completely bypasses the browser (and therefore Google Scholar's instant Cloud Run CAPTCHA bans). Instead, it queries 6 academic APIs concurrently using `asyncio.gather` and `httpx`:
- Semantic Scholar
- arXiv
- Europe PMC
- OpenAlex
- Crossref
- CORE
It then deduplicates the results by title, and reranks them locally using a SentenceTransformer `CrossEncoder` model (`ms-marco-MiniLM-L-6-v2`) for zero API-cost relevancy sorting.

### 3.2 `ExtractionAgent`
**Goal**: Read the actual text of the papers found by the SearchAgent.
**How it works**:
1. **RAM-Only PDF Extraction**: It attempts to download the PDF directly into memory using `httpx`. It parses the bytes using `fitz` (PyMuPDF). This adheres to the rule of never writing files to disk (Cloud Run stateless constraint).
2. **Visual Fallback (UI Navigator Requirement)**: If the PDF download fails, it boots up the `BrowserController` and uses the `vision_loop.py` ReAct agent to navigate to the URL. **Crucially, it only does this for known CAPTCHA-safe domains** (e.g., arxiv.org, semanticscholar.org). It reads the page purely visually.

### 3.3 `SynthesisAgent`
**Goal**: Combine and analyze all extracted paper texts.
**How it works**:
It feeds the extracted text of all papers (up to 800,000 characters) into a single `gemini-2.5-pro` prompt. By leveraging the model's massive 1M token context window, it avoids complex RAG/vector database architectures. It asks the model to identify key findings, consensus, contradictions, and research gaps across the entire corpus.

### 3.4 `CitationAgent`
**Goal**: Build a relationship map of the papers.
**How it works**:
It analyzes the metadata and text to build a node/edge JSON graph representing which papers cite which other papers. This data is fed back to the frontend's D3.js `CitationGraph` component.

### 3.5 `ReportAgent`
**Goal**: Write the final deliverable.
**How it works**:
It takes the structured synthesis from the `SynthesisAgent` and tasks `gemini-2.5-pro` with generating a professional, multi-section Markdown literature review, complete with methodologies and future directions.

---

## 4. Visual Navigation & Loop Resilience

The system's core hackathon feature is its ability to interact with web UIs strictly through pixels.

### 4.1 The ReAct `vision_loop.py`
The vision loop uses `gemini-2.0-flash` to process base64 screenshots taken by Playwright. The System Prompt explicitly forbids the use of DOM selectors, CSS classes, or JS injection. The model must output a JSON action dict containing exact `x`, `y` coordinates for `click` actions, or a `delta` value for `scroll` actions.

### 4.2 Infinite Loop & CAPTCHA Prevention
Because visually-driven agents can easily get stuck repeatedly clicking unclickable coordinates, the system implements:
1. **Click Delays**: `BrowserController` executes clicks with a 100ms delay to simulate physical mouse presses, registering on SPAs and heavily-scripted sites.
2. **Anti-Stuck Memory**: `vision_loop.py` tracks coordinate history. If the model predicts the exact same `(x, y)` coordinate three times consecutively, the loop intercepts the Playwright execution and feeds a dynamic error string back into the model's history (e.g., *"You clicked here multiple times and nothing happened. Try something else."*). This breaks deterministic infinite loops.
3. **Safe Domains List**: The Browser is restricted from opening high-risk heuristic domains (like Google Scholar or external research login gates) to permanently avoid un-solveable reCAPTCHAs.

---

## 5. Security & Constraints
- **Zero Disk Writes**: All PDF byte streams are loaded and garbage collected in RAM.
- **Zero `.env` Leakage**: API Keys are never printed to logs or hardcoded.
- **Strict Sync/Async Boundaries**: ADK Tools are natively synchronous, but the backend utilizes `asyncio.run()` wrappers to seamlessly execute the high-performance async HTTP fetchers and Playwright instances.
