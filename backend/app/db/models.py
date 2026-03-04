from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.sql import func
from app.db.database import Base

class SessionState(Base):
    __tablename__ = "session_states"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    active = Column(Boolean, default=True)

class EmotionHistory(Base):
    __tablename__ = "emotion_history"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("session_states.session_id"))
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    emotion = Column(String, index=True)
    confidence = Column(Float)
    is_stable = Column(Boolean, default=False)

class UserFeedback(Base):
    __tablename__ = "user_feedback"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("session_states.session_id"))
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    track_uri = Column(String, index=True)
    emotion_context = Column(String)
    action = Column(String) # 'like', 'skip', 'neutral'

class RecommendationLog(Base):
    __tablename__ = "recommendation_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("session_states.session_id"))
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    emotion_context = Column(String)
    track_uri = Column(String)

class PerformanceMetrics(Base):
    __tablename__ = "performance_metrics"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    endpoint = Column(String, index=True)
    process_time_ms = Column(Float)
    inference_time_ms = Column(Float, nullable=True)
