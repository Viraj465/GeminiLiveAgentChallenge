import { useState, useEffect, useRef } from 'react';

interface ScreenShareProps {
    wsRef: React.MutableRefObject<WebSocket | null>;
    sessionId: string;
    onCopilotStateChange: (active: boolean) => void;
}

export default function ScreenShare({ wsRef, sessionId, onCopilotStateChange }: ScreenShareProps) {
    const [isSharing, setIsSharing] = useState(false);
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const intervalRef = useRef<number | null>(null);

    const startSharing = async () => {
        try {
            const stream = await navigator.mediaDevices.getDisplayMedia({
                video: { frameRate: 1 } // optimize for 1fps
            });

            streamRef.current = stream;

            // Listen for system "Stop sharing" button
            stream.getVideoTracks()[0].onended = () => {
                stopSharing();
            };

            if (videoRef.current) {
                videoRef.current.srcObject = stream;
            }

            setIsSharing(true);
            onCopilotStateChange(true);

            // Switch WebSocket backend to Copilot mode
            if (wsRef.current?.readyState === WebSocket.OPEN) {
                wsRef.current.send(JSON.stringify({
                    type: 'mode_switch',
                    payload: { mode: 'copilot' },
                    session_id: sessionId
                }));
            }

            // Start 1FPS capture loop
            intervalRef.current = window.setInterval(captureFrame, 1000);

        } catch (err) {
            console.error("Error starting screen share:", err);
            setIsSharing(false);
            onCopilotStateChange(false);
        }
    };

    const stopSharing = () => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }
        if (intervalRef.current) {
            window.clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
        if (videoRef.current) {
            videoRef.current.srcObject = null;
        }

        setIsSharing(false);
        onCopilotStateChange(false);

        // Revert backend to Autopilot mode
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({
                type: 'mode_switch',
                payload: { mode: 'autopilot' },
                session_id: sessionId
            }));
        }
    };

    const captureFrame = () => {
        if (!videoRef.current || !canvasRef.current || !wsRef.current) return;
        if (wsRef.current.readyState !== WebSocket.OPEN) return;

        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Ensure canvas matches video dimensions
        if (video.videoWidth && video.videoHeight) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        } else {
            return;
        }

        // Draw current video frame to canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Convert to base64 (JPEG for smaller size)
        const frameDataUrl = canvas.toDataURL('image/jpeg', 0.6);

        // Strip the "data:image/jpeg;base64," prefix for backend
        const base64Data = frameDataUrl.split(',')[1];

        // Send over WebSocket
        wsRef.current.send(JSON.stringify({
            type: 'screen_frame',
            payload: { frame: base64Data },
            session_id: sessionId
        }));
    };

    useEffect(() => {
        return () => {
            stopSharing(); // Cleanup on unmount
        };
    }, []);

    return (
        <div className="flex items-center gap-3">
            <button
                onClick={isSharing ? stopSharing : startSharing}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2
                    ${isSharing
                        ? 'bg-red-500/10 text-red-400 hover:bg-red-500/20'
                        : 'bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20 shadow-[0_0_15px_rgba(16,185,129,0.1)]'
                    }`}
            >
                {isSharing ? (
                    <>
                        <div className="w-2 h-2 rounded-full bg-red-400 animate-pulse" />
                        Stop Copilot
                    </>
                ) : (
                    <>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14v-4z" />
                            <rect x="3" y="6" width="12" height="12" rx="2" ry="2" />
                        </svg>
                        Start screen share
                    </>
                )}
            </button>

            {/* Hidden video and canvas elements used for frame extraction */}
            <video ref={videoRef} autoPlay playsInline muted className="hidden" />
            <canvas ref={canvasRef} className="hidden" />
        </div>
    );
}
