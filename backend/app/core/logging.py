import logging
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from app.db.database import SessionLocal
from app.db.models import PerformanceMetrics
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)

def setup_logging():
    logger = logging.getLogger("emotion-api")
    logger.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(JSONFormatter())
    
    # Avoid duplicate logs if configured multiple times
    if not logger.handlers:
        logger.addHandler(ch)
    
    return logger

logger = setup_logging()

class PerformanceMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time_ms = (time.time() - start_time) * 1000
        
        # We can log to DB asynchronously or using a background task. 
        # For simplicity and given the requirements, we'll write it directly now, 
        # but in heavy production, this should be an async queue.
        try:
            db = SessionLocal()
            metric = PerformanceMetrics(
                endpoint=request.url.path,
                process_time_ms=process_time_ms
            )
            # Find if there was any model inference time recorded in the request state
            if hasattr(request.state, 'inference_time_ms'):
                metric.inference_time_ms = request.state.inference_time_ms
            
            db.add(metric)
            db.commit()
        except Exception as e:
            logger.error(f"Failed to save performance metrics: {str(e)}")
        finally:
            db.close()
            
        logger.info(f"Request: {request.method} {request.url.path} - Time: {process_time_ms:.2f}ms")
            
        return response
