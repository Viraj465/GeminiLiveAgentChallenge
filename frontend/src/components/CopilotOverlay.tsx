import { useState, useEffect } from 'react';

interface CopilotOverlayProps {
    guidance: string;
    status: string;
}

export default function CopilotOverlay({ guidance, status }: CopilotOverlayProps) {
    const [isVisible, setIsVisible] = useState(false);

    useEffect(() => {
        if (guidance || status === 'analyzing') {
            setIsVisible(true);
        } else {
            setIsVisible(false);
        }
    }, [guidance, status]);

    if (!isVisible) return null;

    return (
        // FIX: Changed positioning to 'bottom-12 right-[400px]' 
        // This pushes it above the status bar and to the left of the 380px chat panel
        <div className="fixed bottom-12 right-[400px] z-50 animate-in fade-in slide-in-from-bottom-4 duration-300">
            <div className="bg-[#1a1b23]/95 backdrop-blur-md border border-white/10 rounded-2xl shadow-2xl p-5 w-[340px] flex flex-col gap-3">
                {/* Header */}
                <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-indigo-500 to-purple-500 flex items-center justify-center shrink-0 shadow-[0_0_15px_rgba(99,102,241,0.3)]">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-white">
                            <path d="M12 2v20M17 5H9.5a3.5 3.5 0 000 7h5a3.5 3.5 0 010 7H6" />
                        </svg>
                    </div>
                    <div>
                        <h3 className="text-sm font-semibold text-white leading-tight">Gemini Copilot</h3>
                        <div className="flex items-center gap-1.5 mt-0.5">
                            <span className="relative flex h-2 w-2">
                                {status === 'analyzing' ? (
                                    <>
                                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
                                        <span className="relative inline-flex rounded-full h-2 w-2 bg-indigo-500"></span>
                                    </>
                                ) : (
                                    <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                                )}
                            </span>
                            <span className="text-[10px] text-gray-400 font-medium uppercase tracking-wider">
                                {status === 'analyzing' ? 'Analyzing Screen...' : 'Guidance Ready'}
                            </span>
                        </div>
                    </div>
                </div>

                {/* Content */}
                <div className="mt-1">
                    {status === 'analyzing' ? (
                        <div className="space-y-2">
                            <div className="h-4 bg-white/5 rounded w-3/4 animate-pulse"></div>
                            <div className="h-4 bg-white/5 rounded w-1/2 animate-pulse"></div>
                        </div>
                    ) : (
                        <p className="text-sm text-gray-300 leading-relaxed max-h-[160px] overflow-y-auto pr-1">
                            {guidance}
                        </p>
                    )}
                </div>
            </div>
        </div>
    );
}