from collections import deque
from collections import Counter
import threading

class EmotionSmoother:
    def __init__(self, window_size=15, confidence_threshold=0.55, required_frames=8):
        """
        Manages sliding windows of emotion predictions to provide stable output state.
        By default, an emotion must appear in 8 of the last 15 frames with high confidence.
        """
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.required_frames = required_frames
        
        # Dictionary mapping session_id -> deque
        self.sessions = {}
        # Dictionary mapping session_id -> string (last stable emotion state)
        self.stable_states = {}
        
        self.lock = threading.Lock()

    def _ensure_session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = deque(maxlen=self.window_size)
            self.stable_states[session_id] = "Neutral"

    def update(self, session_id, emotion, confidence):
        """
        Registers a new frame prediction.
        Returns the stable emotion state for that session.
        High-confidence predictions (>= 0.70) bypass the window for instant response.
        """
        with self.lock:
            self._ensure_session(session_id)
            
            # HIGH-CONFIDENCE BYPASS: Instant state update
            if confidence >= 0.70 and emotion != "Unknown":
                self.stable_states[session_id] = emotion
                # Update window to anchor the new state
                self.sessions[session_id].append(emotion)
                return {
                    "stable_emotion": emotion,
                    "is_stable": True
                }

            # Normal smoothing logic for lower confidence
            if confidence < self.confidence_threshold or emotion == "Unknown":
                self.sessions[session_id].append("Unknown")
            else:
                self.sessions[session_id].append(emotion)
                
            window = self.sessions[session_id]
            emotion_counts = Counter(window)
            dominant_emotion, count = emotion_counts.most_common(1)[0]
            
            # Check if it meets the critical mass threshold
            if count >= self.required_frames and dominant_emotion != "Unknown":
                self.stable_states[session_id] = dominant_emotion
                
            return {
                "stable_emotion": self.stable_states[session_id],
                "is_stable": count >= self.required_frames and dominant_emotion != "Unknown"
            }
            
    def clear_session(self, session_id):
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
            if session_id in self.stable_states:
                del self.stable_states[session_id]
