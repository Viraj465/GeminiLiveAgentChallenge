import { useState, useRef, useEffect, useCallback } from 'react'
import ChatPanel from './components/ChatPanel'
import StatusBar from './components/StatusBar'
import ScreenPreview from './components/ScreenPreview'
import ScreenShare from './components/ScreenShare'
import CitationGraph from './components/CitationGraph'
import ReportViewer from './components/ReportViewer'
import CopilotOverlay from './components/CopilotOverlay'


const SESSION_ID = crypto.randomUUID()
const WS_URL = `ws://localhost:8000/ws/${SESSION_ID}`

interface ChatMsg {
  type: 'user' | 'thought' | 'guidance' | 'action' | 'status' | 'log' | 'error'
  message: string
  time: string
}

export default function App() {
  const [isAgentActive, setIsAgentActive] = useState(false)
  const [isConnected, setIsConnected] = useState(false)
  const [lastAction, setLastAction] = useState('')
  const [latestFrame, setLatestFrame] = useState('')
  const [agentUrl, setAgentUrl] = useState('about:blank')
  const [messages, setMessages] = useState<ChatMsg[]>([])

  // Copilot State
  const [isCopilotActive, setIsCopilotActive] = useState(false)
  const [copilotGuidance, setCopilotGuidance] = useState('')
  const [copilotStatus, setCopilotStatus] = useState('ready')

  // Phase 4 State
  const [activeView, setActiveView] = useState<'browser' | 'graph' | 'report'>('browser')
  const [citationGraphData, setCitationGraphData] = useState<any>(null)
  const [finalReportText, setFinalReportText] = useState('')

  // Token State
  const [sessionTokens, setSessionTokens] = useState({ prompt: 0, candidates: 0, total: 0 })

  const wsRef = useRef<WebSocket | null>(null)

  const addMessage = useCallback((type: ChatMsg['type'], message: string) => {
    setMessages(prev => [...prev, {
      type,
      message,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }])
  }, [])

  // ── Connect WebSocket on mount ──
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      setIsConnected(true)
      addMessage('status', 'Connected to ResearchAgent')
    }

    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data)

        switch (data.type) {
          case 'browser_frame':
            // Live Playwright screenshot
            setLatestFrame(data.payload?.frame || '')
            if (data.payload?.url) setAgentUrl(data.payload.url)
            break
          case 'agent_action':
            if (!isCopilotActive) {
              setLastAction(data.payload?.action || '')
              setIsAgentActive(true)
              addMessage('action', data.payload?.action || JSON.stringify(data.payload?.data))
            } else {
              addMessage('log', `Agent mode switched: ${data.payload?.action || ''}`)
            }
            break
          case 'guidance':
            if (data.payload?.guidance) {
              setCopilotGuidance(data.payload.guidance)
            }
            if (data.payload?.copilot_status) {
              setCopilotStatus(data.payload.copilot_status)
            }
            addMessage('guidance', data.payload?.guidance || '')
            break
          case 'log_update':
            addMessage('log', data.payload?.log || '')
            break
          case 'agent_event':
            try {
              // The backend sends stringified dicts via ADK
              const eventDataStr = data.data || '';
              // Add the string to logs
              addMessage('log', eventDataStr.substring(0, 150) + (eventDataStr.length > 150 ? '...' : ''));

              // Heuristic parsing for Phase 4 structures
              if (eventDataStr.includes("'graph_data': {") || eventDataStr.includes('"graph_data": {')) {
                // Handled in specific graph_update event if emitted properly by backend
              }
            } catch (e) {
              // ignore
            }
            break
          case 'graph_update':
            if (data.payload && data.payload.graph_data) {
              setCitationGraphData(data.payload.graph_data)
              setActiveView('graph')
              addMessage('status', 'Citation network mapped successfully.')
            }
            break
          case 'report_update':
            if (data.payload && data.payload.report) {
              setFinalReportText(data.payload.report)
              setActiveView('report')
              addMessage('status', 'Literature review generated.')
            }
            break
          case 'error':
            addMessage('error', data.payload?.error || 'Unknown error')
            break
          case 'token_update':
            if (data.payload) {
              setSessionTokens({
                prompt: data.payload.prompt_tokens || 0,
                candidates: data.payload.candidates_tokens || 0,
                total: data.payload.total_tokens || 0
              })
            }
            break
          case 'complete':
            setIsAgentActive(false)
            addMessage('status', data.payload?.message || 'Task complete')
            break
          case 'ping':
            ws.send(JSON.stringify({
              type: 'pong',
              payload: { timestamp: Date.now() / 1000 },
              session_id: SESSION_ID
            }))
            break
          default:
            if (data.payload?.log) {
              addMessage('log', data.payload.log)
            }
        }
      } catch {
        addMessage('error', `Parse error`)
      }
    }

    ws.onclose = () => {
      setIsConnected(false)
      setIsAgentActive(false)
      addMessage('status', 'Disconnected')
    }

    ws.onerror = () => {
      addMessage('error', 'Connection error')
    }
  }, [addMessage, isCopilotActive])

  // Connect on mount
  useEffect(() => {
    connectWebSocket()
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, [connectWebSocket])

  // ── Chat input → user_command (triggers autopilot) ──
  const handleSendMessage = useCallback((text: string) => {
    addMessage('user', text)

    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      connectWebSocket()
      const checkAndSend = setInterval(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          clearInterval(checkAndSend)
          wsRef.current.send(JSON.stringify({
            type: 'user_command',
            payload: { command: text },
            session_id: SESSION_ID
          }))
        }
      }, 200)
      setTimeout(() => clearInterval(checkAndSend), 5000)
      return
    }

    wsRef.current.send(JSON.stringify({
      type: 'user_command',
      payload: { command: text },
      session_id: SESSION_ID
    }))

    setIsAgentActive(true)
    addMessage('thought', `Processing: "${text}"...`)
  }, [addMessage, connectWebSocket])

  // ── Browser Click → user_action ──
  const handleBrowserClick = useCallback((x: number, y: number) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'user_action',
        payload: { action: 'click', x, y },
        session_id: SESSION_ID
      }))
      addMessage('log', `Sent manual click at [${x}, ${y}]`)
    }
  }, [addMessage])

  return (
    <div className="h-screen w-screen flex flex-col bg-[#0a0b10] text-gray-300 font-sans overflow-hidden">
      {/* Top Header / Utilities */}
      <div className="h-14 border-b border-white/[0.06] flex flex-row items-center justify-between px-6 bg-[#12131a] shrink-0">
        <div className="font-semibold text-white tracking-wide flex items-center gap-2">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="text-indigo-400">
            <path d="M12 2A10 10 0 1022 12A10 10 0 0012 2Z"></path>
            <path d="M12 16L16 12L12 8"></path>
            <path d="M8 12h8"></path>
          </svg>
          ResearchAgent
        </div>
        <div className="flex items-center gap-4">
          {/* View Toggles */}
          <div className="flex items-center bg-black/20 rounded-lg p-1 mr-4 border border-white/5">
            <button
              onClick={() => setActiveView('browser')}
              className={`px-3 py-1.5 rounded-md text-xs font-medium cursor-pointer transition-all ${activeView === 'browser' ? 'bg-indigo-500 text-white shadow' : 'text-gray-400 hover:text-gray-200'}`}
            >
              Browser
            </button>
            <button
              disabled={!citationGraphData}
              onClick={() => setActiveView('graph')}
              className={`px-3 py-1.5 rounded-md text-xs font-medium cursor-pointer transition-all ${activeView === 'graph' ? 'bg-indigo-500 text-white shadow' : !citationGraphData ? 'opacity-30 cursor-not-allowed text-gray-500' : 'text-gray-400 hover:text-gray-200'}`}
            >
              Citation Network
            </button>
            <button
              disabled={!finalReportText}
              onClick={() => setActiveView('report')}
              className={`px-3 py-1.5 rounded-md text-xs font-medium cursor-pointer transition-all ${activeView === 'report' ? 'bg-indigo-500 text-white shadow' : !finalReportText ? 'opacity-30 cursor-not-allowed text-gray-500' : 'text-gray-400 hover:text-gray-200'}`}
            >
              Final Report
            </button>
          </div>

          <ScreenShare
            wsRef={wsRef}
            sessionId={SESSION_ID}
            onCopilotStateChange={(active: boolean) => {
              setIsCopilotActive(active)
              if (active) {
                setCopilotStatus('analyzing')
                setCopilotGuidance('Starting up Copilot screen analysis...')
              } else {
                setCopilotGuidance('')
                setCopilotStatus('ready')
              }
            }}
          />
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* Left — Dynamic View Panel */}
        <div className="flex-[1.8] min-w-0">
          <div className={activeView === 'browser' ? 'h-full w-full block' : 'hidden'}>
            <ScreenPreview
              latestFrame={latestFrame}
              agentUrl={agentUrl}
              isAgentActive={isAgentActive}
              onMouseClick={handleBrowserClick}
            />
          </div>
          <div className={activeView === 'graph' ? 'h-full w-full block' : 'hidden'}>
            {citationGraphData && <CitationGraph graphData={citationGraphData} />}
          </div>
          <div className={activeView === 'report' ? 'h-full w-full block' : 'hidden'}>
            {finalReportText && <ReportViewer reportText={finalReportText} />}
          </div>
        </div>

        {/* Right — Chat Panel */}
        <div className="w-[380px] shrink-0">
          <ChatPanel
            messages={messages}
            onSendMessage={handleSendMessage}
          />
        </div>
      </div>

      {/* Bottom — Status Bar */}
      <StatusBar lastAction={lastAction} isConnected={isConnected} sessionTokens={sessionTokens} />

      {/* Copilot Floating UI */}
      {isCopilotActive && (
        <CopilotOverlay guidance={copilotGuidance} status={copilotStatus} />
      )}
    </div>
  )
}