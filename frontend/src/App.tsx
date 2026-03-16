import { useState, useRef, useEffect, useCallback } from 'react'
import ChatPanel from './components/ChatPanel'
import StatusBar from './components/StatusBar'
import ScreenPreview from './components/ScreenPreview'
import ScreenShare from './components/ScreenShare'
import CitationGraph from './components/CitationGraph'
import ReportViewer from './components/ReportViewer'
import CopilotOverlay from './components/CopilotOverlay'
import { auth, loginWithGoogle, logoutUser } from './firebase'
import { onAuthStateChanged } from 'firebase/auth'
import type { User } from 'firebase/auth'

const API_BASE_URL = ((import.meta.env.VITE_API_BASE_URL as string | undefined) || 'http://localhost:8000').replace(/\/+$/, '')
const WS_BASE_URL = (
  (import.meta.env.VITE_WS_BASE_URL as string | undefined) ||
  API_BASE_URL.replace(/^http/i, 'ws')
).replace(/\/+$/, '')


interface ChatMsg {
  type: 'user' | 'thought' | 'guidance' | 'action' | 'status' | 'log' | 'error'
  message: string
  time: string
}

const getInitialSessionId = () => {
  const urlParams = new URLSearchParams(window.location.search);
  const session = urlParams.get('session');
  if (session) return session;
  const localSession = localStorage.getItem('RESEARCH_SESSION_ID');
  if (localSession) return localSession;
  const newSession = crypto.randomUUID();
  localStorage.setItem('RESEARCH_SESSION_ID', newSession);
  return newSession;
}

function App({ user }: { user: User }) {
  const [sessionId, setSessionId] = useState(getInitialSessionId())
  const [pastSessions, setPastSessions] = useState<any[]>([])

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

  // Fetch History Sidebar
  const fetchSessionsList = useCallback(async () => {
    try {
      const token = await user.getIdToken();
      fetch(`${API_BASE_URL}/api/sessions`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
        .then(r => r.json())
        .then(d => setPastSessions(d.sessions || []))
        .catch(console.error)
    } catch (error) {
      console.error(error)
    }
  }, [user])

  useEffect(() => {
    fetchSessionsList()
  }, [fetchSessionsList])

  const loadSession = useCallback(async (id: string) => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setSessionId(id)
    localStorage.setItem('RESEARCH_SESSION_ID', id)
    window.history.pushState({}, '', `?session=${id}`)
    
    setCitationGraphData(null)
    setFinalReportText('')
    setActiveView('browser')
    setMessages([{type: 'status', message: `Loaded session ${id.substring(0, 8)}...`, time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}])

    try {
      const token = await user.getIdToken();
      fetch(`${API_BASE_URL}/api/sessions/${id}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
        .then(r => r.json())
        .then(d => {
           const data = d.session;
           if (data && data.graph_data) setCitationGraphData(data.graph_data)
           if (data && data.report_markdown) {
             setFinalReportText(data.report_markdown)
             setActiveView('report')
           } else if (data && data.graph_data) {
             setActiveView('graph')
           }
        })
        .catch(console.error)
    } catch (e) {
      console.error(e)
    }
  }, [user])

  // ── Connect WebSocket on mount & session change ──
  const connectWebSocket = useCallback(async () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    try {
      const token = await user.getIdToken();
      const WS_URL = `${WS_BASE_URL}/ws/${sessionId}?token=${encodeURIComponent(token)}`
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
              if (typeof data.payload?.frame === 'string' && data.payload.frame.length > 0) {
                setLatestFrame(data.payload.frame)
              }
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
              if (data.payload?.guidance) setCopilotGuidance(data.payload.guidance)
              if (data.payload?.copilot_status) setCopilotStatus(data.payload.copilot_status)
              addMessage('guidance', data.payload?.guidance || '')
              break
            case 'log_update':
              addMessage('log', data.payload?.log || '')
              break
            case 'agent_event':
              try {
                const eventDataStr = data.data || '';
                addMessage('log', eventDataStr.substring(0, 150) + (eventDataStr.length > 150 ? '...' : ''));
              } catch (e) {}
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
                fetchSessionsList() // refresh sidebar
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
            case 'cache_status':
              addMessage('status', data.payload?.message || 'Cache update')
              break
            case 'gcs_status':
              addMessage('status', data.payload?.message || 'GCS update')
              break
            case 'complete':
              setIsAgentActive(false)
              addMessage('status', data.payload?.message || 'Task complete')
              fetchSessionsList() // refresh sidebar
              break
            case 'ping':
              ws.send(JSON.stringify({
                type: 'pong',
                payload: { timestamp: Date.now() / 1000 },
                session_id: sessionId
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
    } catch (e) {
      console.error(e)
    }
  }, [addMessage, isCopilotActive, sessionId, fetchSessionsList, user])

  useEffect(() => {
    connectWebSocket()
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, [connectWebSocket])

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
            session_id: sessionId
          }))
        }
      }, 200)
      setTimeout(() => clearInterval(checkAndSend), 5000)
      return
    }

    wsRef.current.send(JSON.stringify({
      type: 'user_command',
      payload: { command: text },
      session_id: sessionId
    }))

    setIsAgentActive(true)
    addMessage('thought', `Processing: "${text}"...`)
  }, [addMessage, connectWebSocket, sessionId])

  const handleBrowserClick = useCallback((x: number, y: number) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'user_action',
        payload: { action: 'click', x, y },
        session_id: sessionId
      }))
      addMessage('log', `Sent manual click at [${x}, ${y}]`)
    }
  }, [addMessage, sessionId])

  return (
    <div className="h-screen w-screen flex flex-row bg-[#0a0b10] text-gray-300 font-sans overflow-hidden">
      
      {/* Left Sidebar for Session History */}
      <div className="w-[260px] bg-[#12131a] border-r border-white/[0.06] flex flex-col shrink-0">
        <div className="h-14 border-b border-white/[0.06] flex items-center px-4 shrink-0 justify-between">
          <div className="font-semibold text-white tracking-wide flex items-center gap-2">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="text-indigo-400">
              <path d="M12 2A10 10 0 1022 12A10 10 0 0012 2Z"></path>
              <path d="M12 16L16 12L12 8"></path>
              <path d="M8 12h8"></path>
            </svg>
            ResearchAgent
          </div>
          <button 
            onClick={() => {
              const newId = crypto.randomUUID();
              loadSession(newId);
              setMessages([]);
            }}
            className="text-gray-400 hover:text-white"
            title="New Session"
          >
            <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4"/></svg>
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-3 flex flex-col gap-2">
          <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2 px-1">Recent Sessions</div>
          {pastSessions.map(session => (
            <button
              key={session.id}
              onClick={() => loadSession(session.id)}
              className={`text-left px-3 py-2 rounded-md text-sm transition-colors border ${sessionId === session.id ? 'bg-indigo-500/10 border-indigo-500/30 text-indigo-200' : 'bg-transparent border-transparent hover:bg-white/5 text-gray-400'}`}
            >
              <div className="truncate font-medium">{session.title || 'Research Session'}</div>
              <div className="text-xs opacity-60 mt-1 flex justify-between">
                <span>{session.id.substring(0, 8)}...</span>
                <span>{session.has_report && '📄'} {session.has_graph && '🕸️'}</span>
              </div>
            </button>
          ))}
          {pastSessions.length === 0 && (
            <div className="text-xs text-gray-600 px-1 italic">No past sessions found.</div>
          )}
        </div>
        <div className="p-4 border-t border-white/[0.06] flex items-center justify-between text-xs text-gray-400">
            <span className="truncate pr-2">{user.email}</span>
            <button onClick={logoutUser} className="text-indigo-400 hover:text-indigo-300">Logout</button>
        </div>
      </div>

      <div className="flex-1 flex flex-col min-w-0">
        {/* Top Header / Utilities */}
        <div className="h-14 border-b border-white/[0.06] flex flex-row items-center justify-end px-6 bg-[#12131a] shrink-0">
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
              sessionId={sessionId}
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
              {citationGraphData && activeView === 'graph' && <CitationGraph graphData={citationGraphData} />}
            </div>
            <div className={activeView === 'report' ? 'h-full w-full block' : 'hidden'}>
              {finalReportText && <ReportViewer reportText={finalReportText} />}
            </div>
          </div>

          {/* Right — Chat Panel */}
          <div className="w-[380px] shrink-0 border-l border-white/[0.06]">
            <ChatPanel
              messages={messages}
              onSendMessage={handleSendMessage}
            />
          </div>
        </div>

        {/* Bottom — Status Bar */}
        <StatusBar lastAction={lastAction} isConnected={isConnected} sessionTokens={sessionTokens} />
      </div>

      {/* Copilot Floating UI */}
      {isCopilotActive && (
        <CopilotOverlay guidance={copilotGuidance} status={copilotStatus} />
      )}
    </div>
  )
}

export default function AppWrapper() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const unsub = onAuthStateChanged(auth, (u) => {
      setUser(u);
      setLoading(false);
    });
    return unsub;
  }, []);

  if (loading) return <div className="h-screen w-screen bg-[#0a0b10] flex items-center justify-center text-white">Loading...</div>;

  if (!user) {
    return (
      <div className="h-screen w-screen bg-[#0a0b10] flex items-center justify-center">
        <div className="bg-[#12131a] p-8 rounded-xl border border-white/10 shadow-2xl max-w-sm w-full text-center">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="text-indigo-400 mx-auto mb-4">
            <path d="M12 2A10 10 0 1022 12A10 10 0 0012 2Z"></path>
            <path d="M12 16L16 12L12 8"></path>
            <path d="M8 12h8"></path>
          </svg>
          <h1 className="text-white text-xl font-bold mb-2">ResearchAgent</h1>
          <p className="text-gray-400 text-sm mb-6">Sign in to sync your sessions and securely access your research graph.</p>
          <button 
            onClick={loginWithGoogle}
            className="w-full bg-white text-black py-2.5 rounded hover:bg-gray-100 font-medium transition flex items-center justify-center gap-2"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12.545,10.239v3.821h5.445c-0.712,2.315-2.647,3.972-5.445,3.972c-3.332,0-6.033-2.701-6.033-6.032s2.701-6.032,6.033-6.032c1.498,0,2.866,0.549,3.921,1.453l2.814-2.814C17.503,2.988,15.139,2,12.545,2C7.021,2,2.543,6.477,2.543,12s4.478,10,10.002,10c8.396,0,10.249-7.85,9.426-11.748L12.545,10.239z"/>
            </svg>
            Continue with Google
          </button>
        </div>
      </div>
    );
  }

  return <App user={user} />;
}
