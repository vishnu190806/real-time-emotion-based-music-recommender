import os
import sys
import base64
import time
import numpy as np
import cv2
import uuid
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# Adjust paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.emotion_inference.emotion_predictor import EmotionPredictor
from backend.emotion_smoothing.emotion_smoother import EmotionSmoother
from backend.music_engine.music_mapper import MusicMapper
from spotify_helper import LANGUAGE_CONFIG

app = FastAPI(title="Emotion Music AI API")

# Enable CORS exactly as React Vite expects
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Application State Layer
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'emotion_model_fer2013.h5'))
labels_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'emotion_labels.json'))

predictor = None
smoother = EmotionSmoother(window_size=15, confidence_threshold=0.55, required_frames=8)
music_engine = MusicMapper()

@app.on_event("startup")
def startup_event():
    global predictor
    try:
        predictor = EmotionPredictor(model_path, labels_path)
    except Exception as e:
        print(f"CRITICAL: Failed to load EmotionPredictor: {e}")

# ---- Pydantic Schemas ----

class FramePayload(BaseModel):
    image_base64: str
    session_id: Optional[str] = None

class RecommendPayload(BaseModel):
    emotion: str
    language: Optional[str] = "Mixed"
    session_id: Optional[str] = None

class EmotionResponse(BaseModel):
    emotion: str
    confidence: float
    is_stable: bool
    face_detected: bool
    
# ---- Endpoints ----

@app.post("/api/v1/session")
def create_session():
    """Generates a UUID for tracking smoothing state per client."""
    return {"session_id": str(uuid.uuid4())}

@app.get("/api/v1/languages")
def get_languages():
    """Proxy for available spotify language catalogs."""
    return {
        "languages": list(LANGUAGE_CONFIG.keys()),
        "default": "Mixed"
    }

@app.post("/api/v1/emotion", response_model=EmotionResponse)
def process_frame(payload: FramePayload):
    """
    Consumes a base64 encoded frame from WebcamCard and processes it via MediaPipe -> EfficientNet.
    Applies sliding window smoothing to ensure state stability.
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Model is currently unavailable")
        
    try:
        # Decode base64
        header, encoded = payload.image_base64.split(",", 1) if "," in payload.image_base64 else ("", payload.image_base64)
        img_data = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_data, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            raise ValueError("Decoded image is unreadable")
            
        # Run inference
        raw_result = predictor.predict(img_bgr)
        
        session_id = payload.session_id or "default"
        
        if raw_result['face_detected']:
            # Apply Temporal Smoothing
            smoothing_result = smoother.update(session_id, raw_result['emotion'], raw_result['confidence'])
        else:
            # Blank out the history smoothly if no face detected
            smoothing_result = smoother.update(session_id, "Unknown", 0.0)
            
        active_stable_emotion = smoothing_result['stable_emotion']
        stable_state = smoothing_result['is_stable']
        
        print("=== SMOOTHED EMOTION ===")
        print(f"Final emotion: {active_stable_emotion.lower()}")
        print("========================\n")
        
        return EmotionResponse(
            emotion=active_stable_emotion if raw_result['face_detected'] else "Unknown",
            confidence=raw_result['confidence'],
            is_stable=stable_state if active_stable_emotion != "Unknown" else False,
            face_detected=raw_result['face_detected']
        )
        
    except Exception as e:
        print(f"Frame processing error: {e}")
        return EmotionResponse(
            emotion="Unknown",
            confidence=0.0,
            is_stable=False,
            face_detected=False
        )

@app.post("/api/v1/recommend")
def get_music_recommendations(payload: RecommendPayload):
    """
    Fetches real spotify tracks according to the detected emotion.
    """
    tracks = music_engine.get_recommendation(
        emotion=payload.emotion, 
        language=payload.language,
        limit=8 # 8 tracks for a perfectly filled 4-column grid (2 rows)
    )
    
    return {
        "emotion": payload.emotion,
        "language": payload.language,
        "tracks": tracks
    }

if __name__ == "__main__":
    import uvicorn
    # Optional auto-start wrapper
    uvicorn.run("backend.api.server:app", host="127.0.0.1", port=8000, reload=True)
