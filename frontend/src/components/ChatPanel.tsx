import { useState, useRef, useEffect } from 'react'
import ChatMessage, { type ChatMsg } from './ChatMessage'

interface ChatPanelProps {
    messages: ChatMsg[];
    onSendMessage: (text: string) => void;
}

export default function ChatPanel({ messages, onSendMessage }: ChatPanelProps) {
    const [activeTab, setActiveTab] = useState<'chat' | 'history'>('chat')
    const [inputValue, setInputValue] = useState('')
    const endRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
        endRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages])

    const handleSend = () => {
        const text = inputValue.trim()
        if (!text) return
        onSendMessage(text)
        setInputValue('')
    }

    const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleSend()
        }
    }

    return (
        <div className="flex flex-col h-full bg-[#0f1117] border-l border-white/[0.08]">
            {/* ── Header ── */}
            <div className="px-4 pt-4 pb-0">
                <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2.5">
                        <div className="w-7 h-7 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                            <span className="text-white text-xs font-bold">R</span>
                        </div>
                        <h2 className="text-white font-semibold text-[15px]">Assistant</h2>
                    </div>
                    <div className="flex items-center gap-1.5">
                        <button className="p-1.5 text-gray-500 hover:text-gray-300 transition-colors rounded-lg hover:bg-white/5 cursor-pointer">
                            <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M12 5v14M5 12h14" /></svg>
                        </button>
                        <button className="p-1.5 text-gray-500 hover:text-gray-300 transition-colors rounded-lg hover:bg-white/5 cursor-pointer">
                            <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M4 6h16M4 12h16M4 18h16" /></svg>
                        </button>
                    </div>
                </div>

                {/* Tabs */}
                <div className="flex gap-0 border-b border-white/[0.08]">
                    <button
                        onClick={() => setActiveTab('chat')}
                        className={`px-4 py-2 text-sm font-medium transition-all cursor-pointer relative ${activeTab === 'chat'
                            ? 'text-white'
                            : 'text-gray-500 hover:text-gray-300'
                            }`}
                    >
                        Chat
                        {activeTab === 'chat' && (
                            <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-500 rounded-full" />
                        )}
                    </button>
                    <button
                        onClick={() => setActiveTab('history')}
                        className={`px-4 py-2 text-sm font-medium transition-all cursor-pointer relative ${activeTab === 'history'
                            ? 'text-white'
                            : 'text-gray-500 hover:text-gray-300'
                            }`}
                    >
                        History
                        {activeTab === 'history' && (
                            <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-500 rounded-full" />
                        )}
                    </button>
                </div>
            </div>

            {/* ── Messages ── */}
            <div className="flex-1 overflow-y-auto px-4 py-4 space-y-0.5">
                {messages.length === 0 && (
                    <div className="flex flex-col items-center justify-center h-full gap-3 text-gray-600">
                        <div className="w-12 h-12 rounded-full bg-white/5 flex items-center justify-center">
                            <svg width="24" height="24" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24" className="text-gray-500">
                                <path d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                            </svg>
                        </div>
                        <p className="text-sm">Ask me anything about your screen...</p>
                    </div>
                )}

                {messages.map((msg, i) => (
                    <ChatMessage key={i} msg={msg} />
                ))}
                <div ref={endRef} />
            </div>

            {/* ── Input Bar ── */}
            <div className="p-3 border-t border-white/[0.08]">
                <div className="flex items-center gap-2 bg-white/[0.06] border border-white/[0.1] rounded-2xl px-3 py-1.5">
                    <span className="text-gray-500 text-lg shrink-0">☺</span>
                    <input
                        type="text"
                        value={inputValue}
                        onChange={e => setInputValue(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder="Ask anything..."
                        className="flex-1 bg-transparent text-white text-sm placeholder-gray-500 outline-none py-1.5"
                    />
                    <button className="p-1.5 text-gray-500 hover:text-gray-300 transition-colors cursor-pointer">
                        <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" /></svg>
                    </button>
                    <button className="p-1.5 text-gray-500 hover:text-gray-300 transition-colors cursor-pointer">
                        <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" /></svg>
                    </button>
                    <button
                        onClick={handleSend}
                        disabled={!inputValue.trim()}
                        className={`w-8 h-8 rounded-full flex items-center justify-center transition-all cursor-pointer shrink-0 ${inputValue.trim()
                            ? 'bg-blue-500 hover:bg-blue-400 text-white shadow-lg shadow-blue-500/25'
                            : 'bg-white/10 text-gray-600'
                            }`}
                    >
                        <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24"><path d="M5 12h14M12 5l7 7-7 7" /></svg>
                    </button>
                </div>
            </div>
        </div>
    )
}
