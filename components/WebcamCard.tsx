import React, { useState, useEffect, useRef, useCallback } from 'react';
import type { Emotion } from '../types';
import { EMOTION_COLORS, EMOTION_POLL_INTERVAL } from '../constants';
import { motion } from 'framer-motion';

const CameraIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-cyan-400" style={{filter: 'drop-shadow(0 0 5px currentColor)'}}>
        <path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"></path><circle cx="12" cy="13" r="3"></circle>
    </svg>
);

const RetryIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>
);

const WebcamErrorState: React.FC<{ onRetry: () => void }> = ({ onRetry }) => (
    <div className="absolute inset-0 flex items-center justify-center bg-black/50 backdrop-blur-md z-20 p-4">
        <motion.div
            className="w-full max-w-md text-center bg-black/40 backdrop-blur-2xl border-2 border-white/10 rounded-2xl p-8 flex flex-col items-center"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ type: 'spring' }}
        >
            <div className="animate-pulse">
                <CameraIcon />
            </div>
            <h3 className="text-xl font-bold mt-4 mb-2 text-white">Camera Access Needed</h3>
            <p className="text-gray-300 mb-6">We need your permission to use the camera locally.</p>
            <ul className="text-left text-sm text-gray-400 space-y-2 mb-8">
                <li>✓ Allow camera permissions in your browser.</li>
                <li>✓ Ensure no other app is using the camera.</li>
            </ul>
            <button
                onClick={onRetry}
                className="flex items-center justify-center gap-2 px-6 py-3 bg-gradient-to-r from-cyan-500 to-purple-500 text-white font-semibold rounded-lg hover:scale-105 transition-transform shadow-lg hover:shadow-cyan-500/50"
            >
                <RetryIcon />
                Retry Connection
            </button>
        </motion.div>
    </div>
);


const WebcamCard: React.FC<{ 
    emotion: Emotion; 
    isStable?: boolean;
    onFrameCapture: (base64Image: string) => void;
}> = ({ emotion, isStable = false, onFrameCapture }) => {
    const [isLoading, setIsLoading] = useState(true);
    const [hasError, setHasError] = useState(false);
    
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const streamRef = useRef<MediaStream | null>(null);
    
    const safeEmotion = emotion ?? "Neutral";
    const currentColors = EMOTION_COLORS[safeEmotion] || EMOTION_COLORS["Neutral"];

    const startCamera = async () => {
        setIsLoading(true);
        setHasError(false);
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 640, height: 480, facingMode: "user" } 
            });
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
            }
            streamRef.current = stream;
            setIsLoading(false);
        } catch (err) {
            console.error("Error accessing webcam:", err);
            setHasError(true);
            setIsLoading(false);
        }
    };

    useEffect(() => {
        startCamera();
        return () => {
            if (streamRef.current) {
                streamRef.current.getTracks().forEach(track => track.stop());
            }
        };
    }, []);

    const [isPaused, setIsPaused] = useState(false);
    const [pauseTimeLeft, setPauseTimeLeft] = useState(0);
    const isCountingDown = useRef(false);
    
    // Track how long we have been analyzing without finding stability
    const [analysisSeconds, setAnalysisSeconds] = useState(0);
    const [gracePeriod, setGracePeriod] = useState(false);

    // Increment analysis time every second we are not paused or stable
    useEffect(() => {
        if (!isPaused && !isStable && !gracePeriod) {
            const timer = setInterval(() => {
                setAnalysisSeconds(prev => prev + 1);
            }, 1000);
            return () => clearInterval(timer);
        } else {
            setAnalysisSeconds(0); // Reset when stable or paused
        }
    }, [isPaused, isStable, gracePeriod]);
    
    // Trigger pause when either condition is met
    useEffect(() => {
        if (!isPaused && !gracePeriod && (isStable || analysisSeconds >= 4)) {
            setIsPaused(true);
            setPauseTimeLeft(5);
            setAnalysisSeconds(0); // Reset the analysis timeout tracker
        }
    }, [isStable, analysisSeconds, isPaused, gracePeriod]);

    // Handle the actual countdown timer independently of the triggers
    useEffect(() => {
        if (!isPaused) return;

        const countdownInterval = setInterval(() => {
            setPauseTimeLeft(prev => {
                if (prev <= 1) {
                    clearInterval(countdownInterval);
                    setIsPaused(false);
                    
                    // Force a 2-second grace period where pausing is illegal
                    // This forces the camera to collect fresh data instead of instantly looping
                    // on stale isStable=true states from 10 seconds ago.
                    setGracePeriod(true);
                    setTimeout(() => setGracePeriod(false), 2000);
                    
                    return 0;
                }
                return prev - 1;
            });
        }, 1000);

        return () => clearInterval(countdownInterval);
    }, [isPaused]);

    // Frame Capture Loop
    useEffect(() => {
        if (isLoading || hasError || isPaused) return;

        const intervalId = setInterval(() => {
            if (videoRef.current && canvasRef.current) {
                const video = videoRef.current;
                const canvas = canvasRef.current;
                
                // Only capture if video is playing
                if (video.readyState === video.HAVE_ENOUGH_DATA) {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    const ctx = canvas.getContext('2d');
                    if (ctx) {
                        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                        const base64 = canvas.toDataURL('image/jpeg', 0.8);
                        onFrameCapture(base64);
                    }
                }
            }
        }, EMOTION_POLL_INTERVAL);

        return () => clearInterval(intervalId);
    }, [isLoading, hasError, isPaused, onFrameCapture]);

    return (
        <motion.div
            className="relative p-1 rounded-3xl transition-all duration-700 ease-in-out"
            style={{ background: `linear-gradient(135deg, ${EMOTION_COLORS[safeEmotion]?.glow || EMOTION_COLORS["Neutral"].glow}, #1a0b2e)` }}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.7, delay: 0.2, type: "spring" }}
        >
            <div 
                className="relative bg-white/5 backdrop-blur-3xl border border-white/10 rounded-3xl p-2 h-full shadow-[0_8px_32px_rgba(0,0,0,0.5),inset_0_0_20px_rgba(255,255,255,0.05)] overflow-hidden scanline-overlay aspect-[4/3]"
                style={{ '--glow-color': currentColors.glow, '--glow-color-inner': currentColors.glowInner } as React.CSSProperties}
            >
                <div className="absolute inset-0 holographic-border rounded-3xl transition-all duration-700" />
                <div className="absolute top-4 right-4 z-10">
                    <div className={`flex items-center gap-2 text-white px-3 py-1 rounded-full text-sm font-bold shadow-lg backdrop-blur-sm border transition-colors ${
                        isPaused ? 'bg-purple-600/80 border-purple-400/50' :
                        isStable ? 'bg-green-600/80 border-green-400/50' : 'bg-yellow-600/80 border-yellow-400/50'
                    }`}>
                        <span className="relative flex h-3 w-3">
                            {!isStable && !isPaused && <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-yellow-400 opacity-75"></span>}
                            <span className={`relative inline-flex rounded-full h-3 w-3 ${
                                isPaused ? 'bg-purple-500' :
                                isStable ? 'bg-green-500' : 'bg-yellow-500'
                            }`}></span>
                        </span>
                        {isPaused ? `NEXT ANALYSIS IN ${pauseTimeLeft}s` : isStable ? 'STABLE' : 'ANALYZING'}
                    </div>
                </div>

                {isLoading && !hasError && (
                    <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-20">
                        <div className="text-center">
                           <div className="w-12 h-12 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                           <p>Initializing camera...</p>
                        </div>
                    </div>
                )}
                {hasError && <WebcamErrorState onRetry={startCamera} />}
                
                <video 
                    ref={videoRef}
                    autoPlay 
                    playsInline 
                    muted
                    className="w-full h-full object-cover rounded-2xl transform -scale-x-100" // Mirror self
                />
                
                {/* Hidden canvas for taking snapshots */}
                <canvas ref={canvasRef} className="hidden" />
                
                {!hasError && !isLoading && ['top-2 left-2', 'top-2 right-2 rotate-90', 'bottom-2 right-2 rotate-180', 'bottom-2 left-2 -rotate-90'].map(pos => (
                     <div key={pos} className={`absolute ${pos} w-8 h-8 border-cyan-400/50`} style={{borderStyle: 'solid', borderWidth: '2px 0 0 2px'}}></div>
                ))}
            </div>
        </motion.div>
    );
};

export default WebcamCard;