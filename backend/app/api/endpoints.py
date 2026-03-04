import time
import uuid
import base64
import cv2
import numpy as np
from typing import List, Optional, Tuple
from fastapi import APIRouter, Depends, HTTPException, Request, Body
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.db.database import get_db
from app.db.models import SessionState, EmotionHistory, UserFeedback, RecommendationLog
from app.ml.model_inference import get_emotion_model, get_face_detector
from app.ml.state_manager import EmotionStateManager, Emotion
from app.services.spotify_service import spotify_service, LANGUAGE_CONFIG
from app.services.recommendation_engine import recommendation_engine
from app.core.logging import logger

router = APIRouter()
state_manager = EmotionStateManager(window_size=25, stability_threshold=5, confidence_threshold=0.35)

# Pydantic Schemas
class SessionCreate(BaseModel):
    client_id: Optional[str] = None

class FeedbackRequest(BaseModel):
    session_id: str
    track_uri: str
    emotion_context: str
    action: str  # "like", "skip", "neutral"

class TrackRequest(BaseModel):
    emotion: str
    language: str = "Mixed"
    session_id: Optional[str] = None

class EmotionRequest(BaseModel):
    image_base64: str  # Base64 encoded JPEG
    session_id: Optional[str] = None

def decode_base64_image(base64_string: str) -> np.ndarray:
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    try:
        img_data = base64.b64decode(base64_string)
    except Exception:
        raise ValueError("Malformed base64 string")
        
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("imdecode failed to parse image bytes")
    return img

def _process_image_cpu_bound(frame: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], str, float]:
    """Runs pure CPU computation (OpenCV + TensorFlow Inference) safely off the main event loop."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Preprocessing (CLAHE) for Face Detection Only
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)

    face_detector = get_face_detector()
    face_coords = face_detector.detect(gray_enhanced)
    
    detected_emotion = "Unknown"
    confidence = 0.0

    if face_coords is not None:
        x, y, w, h = face_coords
        
        # ── FIX 2 (Update): FACE CROP MARGIN & SIZE VALIDATION ──
        if w < 50 or h < 50:
            logger.warning(f"Face crop too small ({w}x{h}), skipping frame to prevent bad predictions.")
            return face_coords, "Unknown", 0.0
            
        # MediaPipe returns very tight bounding boxes. Expand by ~20% to better match training crops.
        margin_x = int(w * 0.2)
        margin_y = int(h * 0.2)
        
        frame_height, frame_width = frame.shape[:2]
        
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(frame_width, x + w + margin_x)
        y2 = min(frame_height, y + h + margin_y)
        
        # Extract color ROI using expanded bounds
        roi_bgr = frame[y1:y2, x1:x2]
        
        # NOTE: Color conversion to Gray->RGB is now handled directly in model.predict()
        # so we just pass the BGR crop directly (unlike before where we converted to RGB here).
        
        emotion_model = get_emotion_model()
        detected_emotion, confidence = emotion_model.predict(roi_bgr)
        
    return face_coords, detected_emotion, confidence

# Endpoints
@router.post("/session", tags=["core"])
def create_session(db: Session = Depends(get_db)):
    """Initialize a new logical session for tracking persistence and adaptation"""
    session_id = str(uuid.uuid4())
    new_session = SessionState(session_id=session_id)
    db.add(new_session)
    db.commit()
    return {"session_id": session_id}

@router.post("/emotion", tags=["ml"])
async def detect_emotion(request: Request, body: EmotionRequest, db: Session = Depends(get_db)):
    """Process a video frame asynchronously and return the smoothed stability-aware emotion."""
    from fastapi.concurrency import run_in_threadpool
    
    import asyncio
    
    # Gatekeeper Payload Validation
    if len(body.image_base64) > 1_500_000: # Broadly limit to 1.5MB to prevent memory exhaustion
        raise HTTPException(status_code=413, detail="Payload too large. Base64 string exceeds chunk limits.")
        
    try:
        frame = decode_base64_image(body.image_base64)
    except Exception as e:
        logger.warning(f"Failed to decode client image: {e}")
        raise HTTPException(status_code=400, detail="Invalid or corrupt image encoding")

    # Offload Heavy ML / Computer Vision entirely off the Async Main Loop, with Strict Timeout
    inf_start = time.time()
    try:
        # 800ms Safety Cutoff (Phase 3 Requirement)
        face_coords, detected_emotion, confidence = await asyncio.wait_for(
            run_in_threadpool(_process_image_cpu_bound, frame),
            timeout=0.8
        )
    except asyncio.TimeoutError:
        logger.warning("Inference exceeded 800ms timeout! Skipping frame to prevent blocking.")
        from fastapi import Response
        return Response(status_code=204) # HTTP 204 No Content

    inference_ms = (time.time() - inf_start) * 1000
    request.state.inference_time_ms = inference_ms
    
    # ── FIX 9: INFERENCE PERFORMANCE CHECK ──
    logger.info(f"Frame Inference Time: {inference_ms:.2f}ms")
    if inference_ms > 200:
        logger.warning(f"Performance Degradation: Inference took {inference_ms:.2f}ms (>200ms target)")

    if face_coords is None:
        return {
            "status": "no_face_detected",
            "emotion": None
        }

    # Pass through Stability Layer (Very lightweight operations on main thread)
    stable_emotion, is_stable = state_manager.process_frame(detected_emotion, confidence)
    
    # PHASE 2 - DISABLE RUNTIME DATABASE WRITES
    # if body.session_id and face_coords is not None:
    #     # Offload synchronous SQLAlchemy database commit
    #     def _write_db():
    #         hist = EmotionHistory(
    #             session_id=body.session_id,
    #             emotion=detected_emotion,
    #             confidence=confidence,
    #             is_stable=is_stable
    #         )
    #         db.add(hist)
    #         db.commit()
    #     await run_in_threadpool(_write_db)

    return {
        "emotion": stable_emotion.value,
        "is_stable": is_stable,
        "raw_detected_emotion": detected_emotion,
        "confidence": float(confidence),
        "face_detected": face_coords is not None
    }

@router.get("/languages", tags=["core"])
def get_languages():
    languages = list(LANGUAGE_CONFIG.keys())
    return {
        "languages": languages,
        "default": "Mixed"
    }

@router.post("/recommend", tags=["core"])
def recommend_tracks(body: TrackRequest, db: Session = Depends(get_db)):
    """Generate adaptive recommendations based on emotion and history."""
    if body.session_id:
        # Trigger adaptive weighting update
        recommendation_engine.adjust_weights_from_feedback(db, body.session_id)
        
    tracks = spotify_service.get_tracks_for_emotion(body.emotion, body.language, limit=8)
    
    if body.session_id and tracks:
        # Log recommendations
        for track in tracks:
            log = RecommendationLog(
                session_id=body.session_id,
                emotion_context=body.emotion,
                track_uri=track['uri']
            )
            db.add(log)
        db.commit()
    
    return {
        "emotion": body.emotion,
        "language": body.language,
        "tracks": tracks,
        "count": len(tracks)
    }

@router.post("/feedback", tags=["core"])
def post_feedback(body: FeedbackRequest, db: Session = Depends(get_db)):
    """Receive user feedback for tracks to power dynamic scoring."""
    feedback = UserFeedback(
        session_id=body.session_id,
        track_uri=body.track_uri,
        emotion_context=body.emotion_context,
        action=body.action
    )
    db.add(feedback)
    db.commit()
    return {"status": "success", "action_recorded": body.action}
