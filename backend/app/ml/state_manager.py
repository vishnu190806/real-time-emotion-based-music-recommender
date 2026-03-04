import time
from collections import deque
from enum import Enum
from typing import Dict, Optional, Tuple

class Emotion(str, Enum):
    ANGRY = "Angry"
    DISGUST = "Disgust"
    FEAR = "Fear"
    HAPPY = "Happy"
    SAD = "Sad"
    SURPRISE = "Surprise"
    NEUTRAL = "Neutral"
    UNKNOWN = "Unknown"

class EmotionStateManager:
    """
    Manages the stable emotion state to prevent UI flicker and erratic 
    recommendation switching.
    """
    def __init__(self, window_size: int = 15, stability_threshold: int = 3, confidence_threshold: float = 0.20):
        self.window_size = window_size
        self.stability_threshold = stability_threshold  # M stable frames needed to switch
        self.confidence_threshold = confidence_threshold
        
        # History of recent valid emotions
        self.emotion_window = deque(maxlen=window_size)
        
        # Current stable state
        self.current_stable_emotion: Emotion = Emotion.NEUTRAL
        
        # Track how long the 'pending' new state has been observed
        self.pending_emotion: Optional[Emotion] = None
        self.pending_count: int = 0
        
        self.last_update_time = time.time()
        
    def process_frame(self, detected_emotion: str, confidence: float, face_detected: bool = True) -> Tuple[Emotion, bool]:
        """
        Process a new frame prediction.
        Returns Tuple[StableEmotion, IsStable]
        """
        self.last_update_time = time.time()
        
        # 0. Allow transient face detection drops without resetting the continuity buffer
        if not face_detected:
            return self.current_stable_emotion, False
        
        # 1. Ignore low confidence frames
        if confidence < self.confidence_threshold:
            return self.current_stable_emotion, False
            
        try:
            emotion_enum = Emotion(detected_emotion)
        except ValueError:
            return self.current_stable_emotion, False
            
        self.emotion_window.append(emotion_enum)
        
        # ── FIX 4: TEMPORAL SMOOTHING ──
        # Maintain a short history buffer and only switch if an emotion appears at least 3 times.
        from collections import Counter
        counts = Counter(self.emotion_window)
        most_common_emotion, most_common_count = counts.most_common(1)[0]
        
        # We need at least 'stability_threshold' (e.g., 3) occurrences to switch to a new stable state
        if most_common_emotion != self.current_stable_emotion and most_common_count >= self.stability_threshold:
            # Add a cooldown mechanism 
            now = time.time()
            if not hasattr(self, 'last_change_time'):
                self.last_change_time = 0.0
                
            if (now - self.last_change_time) >= 6.0:  # 6.0s cooldown
                self.current_stable_emotion = most_common_emotion
                self.last_change_time = now
                
            return self.current_stable_emotion, True
            
        elif most_common_emotion == self.current_stable_emotion:
            # We are stable in our current state
            return self.current_stable_emotion, True
            
        # We are fluctuating or building up to a new state
        return self.current_stable_emotion, False
