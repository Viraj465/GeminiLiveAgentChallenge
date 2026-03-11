import { useState, useEffect, useRef } from 'react'

interface ScreenPreviewProps {
    latestFrame: string;
    agentUrl: string;
    isAgentActive: boolean;
    onMouseClick?: (x: number, y: number) => void;
}

export default function ScreenPreview({ latestFrame, agentUrl, isAgentActive, onMouseClick }: ScreenPreviewProps) {
    const [displayUrl, setDisplayUrl] = useState('about:blank')
    const imgRef = useRef<HTMLImageElement>(null);

    useEffect(() => {
        if (agentUrl) setDisplayUrl(agentUrl)
    }, [agentUrl])

    const handleImageClick = (e: React.MouseEvent<HTMLImageElement>) => {
        if (!onMouseClick || !imgRef.current) return;

        const img = imgRef.current;
        const rect = img.getBoundingClientRect();

        const naturalWidth = img.naturalWidth;
        const naturalHeight = img.naturalHeight;

        if (naturalWidth === 0 || naturalHeight === 0) return;

        // Because we use object-contain, the image might have letterboxing.
        // We calculate the actual rendered scale.
        const scaleX = rect.width / naturalWidth;
        const scaleY = rect.height / naturalHeight;
        const scale = Math.min(scaleX, scaleY);

        // Calculate the actual rendered dimensions of the image on screen
        const renderedWidth = naturalWidth * scale;
        const renderedHeight = naturalHeight * scale;

        // Calculate the padding (letterboxing) inside the container
        const offsetX = (rect.width - renderedWidth) / 2;
        const offsetY = (rect.height - renderedHeight) / 2;

        // Calculate click coordinates relative to the rendered image boundaries
        const clickX = e.clientX - rect.left - offsetX;
        const clickY = e.clientY - rect.top - offsetY;

        // Ensure the user actually clicked on the image and not the black padding
        if (clickX >= 0 && clickX <= renderedWidth && clickY >= 0 && clickY <= renderedHeight) {
            // Translate the click back to the original Playwright coordinate space
            const finalX = Math.round(clickX / scale);
            const finalY = Math.round(clickY / scale);

            onMouseClick(finalX, finalY);
        }
    };

    return (
        <div className="flex flex-col h-full bg-[#0c0d12]">
            {/* ── Browser Chrome ── */}
            <div className="bg-[#1a1b23] border-b border-white/[0.06] px-4 py-2.5">
                {/* Tab bar */}
                <div className="flex items-center gap-3 mb-2.5">
                    <div className="flex gap-1.5">
                        <div className="w-3 h-3 rounded-full bg-[#ff5f57]" />
                        <div className="w-3 h-3 rounded-full bg-[#febc2e]" />
                        <div className="w-3 h-3 rounded-full bg-[#28c840]" />
                    </div>
                    <div className="flex items-center bg-white/[0.08] rounded-lg px-3 py-1 text-[11px] text-gray-400 gap-2">
                        <span className="truncate max-w-[140px]">Agent Browser</span>
                        <span className="text-gray-600 cursor-pointer hover:text-gray-400">×</span>
                    </div>
                    <span className="text-gray-600 text-lg leading-none cursor-pointer hover:text-gray-400">+</span>
                </div>
                {/* Address bar */}
                <div className="flex items-center gap-2">
                    <div className="flex items-center gap-1.5 text-gray-600">
                        <span className="cursor-pointer hover:text-gray-400">←</span>
                        <span className="cursor-pointer hover:text-gray-400">→</span>
                        <span className="cursor-pointer hover:text-gray-400">↻</span>
                    </div>
                    <div className="flex-1 flex items-center bg-white/[0.06] rounded-lg px-3 py-1.5 text-sm text-gray-400 gap-2">
                        <svg width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" className="text-gray-600 shrink-0"><path d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" /></svg>
                        <span className="truncate">{displayUrl}</span>
                    </div>
                    {isAgentActive && (
                        <div className="flex items-center gap-1.5 bg-emerald-500/15 text-emerald-400 rounded-full px-3 py-1 text-xs font-medium shrink-0">
                            <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse" />
                            AI browsing
                        </div>
                    )}
                </div>
            </div>

            {/* ── Browser Content ── */}
            <div className="flex-1 relative flex items-center justify-center overflow-hidden bg-[#0c0d12]">
                {/* Live screenshot from Playwright */}
                {latestFrame ? (
                    <img
                        ref={imgRef}
                        src={`data:image/png;base64,${latestFrame}`}
                        alt="Agent browser view"
                        className="w-full h-full object-contain cursor-crosshair"
                        onClick={handleImageClick}
                        draggable={false}
                    />
                ) : (
                    /* Placeholder — waiting for task */
                    <div className="flex flex-col items-center gap-5">
                        <div className="w-20 h-20 rounded-2xl bg-white/[0.04] border border-white/[0.08] flex items-center justify-center">
                            <svg width="36" height="36" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24" className="text-gray-600">
                                <path d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                            </svg>
                        </div>
                        <div className="text-center">
                            <p className="text-gray-400 text-sm font-medium mb-1">
                                {isAgentActive ? 'Agent is starting up...' : 'Waiting for a task'}
                            </p>
                            <p className="text-gray-600 text-xs">
                                Type a research task in the chat to start the agent
                            </p>
                        </div>
                    </div>
                )}

                {/* Active indicator overlay */}
                {isAgentActive && latestFrame && (
                    <div className="absolute top-3 right-3 flex items-center gap-1.5 bg-black/60 backdrop-blur-sm text-emerald-400 rounded-full px-3 py-1.5 text-xs font-medium pointer-events-none">
                        <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse" />
                        Live
                    </div>
                )}
            </div>
        </div>
    )
}