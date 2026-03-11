export interface ChatMsg {
    type: 'user' | 'thought' | 'guidance' | 'action' | 'status' | 'log' | 'error';
    message: string;
    time: string;
}

interface ChatMessageProps {
    msg: ChatMsg;
}

export default function ChatMessage({ msg }: ChatMessageProps) {
    // User message — right-aligned bubble
    if (msg.type === 'user') {
        return (
            <div className="flex justify-end mb-3">
                <div className="bg-[#34a853] text-white px-4 py-2.5 rounded-2xl rounded-br-md max-w-[85%] text-sm shadow-md">
                    {msg.message}
                </div>
            </div>
        )
    }

    // Thought message — reasoning card
    if (msg.type === 'thought' || msg.type === 'guidance') {
        return (
            <div className="mb-3">
                <div className="bg-white/[0.06] border border-white/10 rounded-2xl p-3.5 backdrop-blur-sm">
                    <div className="flex items-center gap-2 mb-1.5">
                        <span className="text-amber-400 text-sm">✦</span>
                        <span className="text-amber-400 text-xs font-semibold tracking-wide">
                            {msg.type === 'guidance' ? 'Guidance' : 'Thought'}
                        </span>
                    </div>
                    <p className="text-gray-300 text-sm leading-relaxed">
                        {msg.message}
                    </p>
                </div>
            </div>
        )
    }

    // Action message
    if (msg.type === 'action') {
        return (
            <div className="mb-3">
                <div className="bg-white/[0.06] border border-white/10 rounded-2xl p-3.5 backdrop-blur-sm">
                    <div className="flex items-center gap-2 mb-1.5">
                        <span className="text-blue-400">▶</span>
                        <span className="text-blue-400 text-xs font-semibold tracking-wide">Action</span>
                    </div>
                    <p className="text-gray-300 text-sm leading-relaxed flex items-center gap-2">
                        {msg.message}
                        <span className="text-gray-500 text-xs ml-auto">⏱</span>
                    </p>
                </div>
            </div>
        )
    }

    // Error message
    if (msg.type === 'error') {
        return (
            <div className="mb-3">
                <div className="bg-red-500/10 border border-red-500/20 rounded-2xl p-3.5">
                    <div className="flex items-center gap-2 mb-1.5">
                        <span className="text-red-400">✗</span>
                        <span className="text-red-400 text-xs font-semibold tracking-wide">Error</span>
                    </div>
                    <p className="text-red-300 text-sm leading-relaxed">{msg.message}</p>
                </div>
            </div>
        )
    }

    // Status / Log — subtle system message
    return (
        <div className="mb-2">
            <div className="flex items-center gap-2 px-1">
                <span className="text-emerald-500 text-xs">●</span>
                <p className="text-gray-500 text-xs">{msg.message}</p>
                <span className="text-gray-700 text-[10px] ml-auto">{msg.time}</span>
            </div>
        </div>
    )
}
