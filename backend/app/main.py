import uvicorn
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logging import PerformanceMiddleware, logger
from app.db.database import engine, Base
from app.ml.model_inference import get_emotion_model, get_face_detector

# Ensure all models are created in the database
Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Initialization
    logger.info("Initializing ML models for Production Environment...")
    
    face_detector = get_face_detector()
    if not face_detector.load():
        logger.error("Failed to load Face Detector. Shutting down.")
        raise RuntimeError("Haarcascade initialization failed")
        
    emotion_model = get_emotion_model()
    if not emotion_model.load():
        logger.error("Failed to load EfficientNet Model. Shutting down.")
        raise RuntimeError("EfficientNet initialization failed")
    
    # Run a warmup inference to force TensorFlow XLA / Graph compilation
    logger.info("Running TensorFlow computational graph warmup...")
    dummy_rgb_roi = np.zeros((100, 100, 3), dtype=np.uint8)
    emotion_model.predict(dummy_rgb_roi)
    logger.info("Warmup complete. System ready to serve traffic.")
    
    yield
    
    # Shutdown operations (if any)
    logger.info("Shutting down application...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    description="Production-ready real-time emotion music recommender.",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Performance Logging Middleware
app.add_middleware(PerformanceMiddleware)

@app.get("/api/v1/health", tags=["system"])
def health_check():
    logger.info("Health check endpoint hit")
    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "database": getattr(settings, 'DATABASE_URL', 'Unknown').split(':')[0]
    }

# We will include routers here once created
from app.api.endpoints import router as api_router
app.include_router(api_router, prefix=settings.API_V1_STR)

if __name__ == "__main__":
    import os
    # Ensure CPU-only strictness if deployment limits VRAM or triggers container leaks
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # Disable uvicorn reload for production safety 
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False, workers=1)
