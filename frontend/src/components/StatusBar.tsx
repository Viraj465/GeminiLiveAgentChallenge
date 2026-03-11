interface StatusBarProps {
    lastAction: string;
    isConnected: boolean;
    sessionTokens?: {
        total: number;
        prompt?: number;
        candidates?: number;
    } | null;
}

export default function StatusBar({ lastAction, isConnected, sessionTokens }: StatusBarProps) {
    return (
        <div className="flex items-center justify-between px-4 py-2 bg-[#0a0b10] border-t border-white/[0.06] text-xs">
            {/* Left — Last action */}
            <div className="flex items-center gap-2 text-gray-500">
                <span className={`w-2 h-2 rounded-full ${lastAction ? 'bg-amber-500' : 'bg-gray-700'}`} />
                <span>Last action: {lastAction || 'None'}</span>
            </div>

            {/* Center — System health */}
            <div className="flex items-center gap-4">
                <div className="flex items-center gap-1.5">
                    <span>System Health:</span>
                    <span className={isConnected ? 'text-emerald-400 font-medium' : 'text-red-400 font-medium'}>
                        {isConnected ? 'Good' : 'Disconnected'}
                    </span>
                    <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-emerald-400' : 'bg-red-400'}`} />
                </div>

                {/* Heartbeat pulse */}
                <svg width="40" height="16" viewBox="0 0 40 16" className="text-emerald-500/60">
                    <polyline
                        points="0,8 8,8 11,3 14,13 17,5 20,11 23,8 40,8"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1.5"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                    />
                </svg>
            </div>

            {/* Right — Session */}
            <div className="flex flex-col items-end gap-1 text-gray-700">
                <div className="flex items-center gap-3">
                    {sessionTokens && sessionTokens.total > 0 && (
                        <span className="text-emerald-400 font-medium px-2 py-0.5 rounded bg-emerald-400/10 border border-emerald-400/20">
                            Tokens parsed: {sessionTokens.total.toLocaleString()}
                        </span>
                    )}
                    <span>ResearchAgent v0.1</span>
                </div>
            </div>
        </div>
    )
}
