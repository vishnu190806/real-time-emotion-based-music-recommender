import random
import time
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from app.db.models import UserFeedback

class AdaptiveRecommendationEngine:
    """
    Replaces the static dictionary with an adaptive scoring system.
    Tracks recommended under a certain emotion will have their search queries 
    weighted based on user feedback.
    """
    
    def __init__(self):
        # Base queries for each emotion (similar to original, but we will dynamically weight them)
        self.emotion_queries = {
            "Happy": [
                {"query": "happy upbeat positive cheerful", "weight": 1.0},
                {"query": "feel good pop dance", "weight": 1.0},
                {"query": "energetic party hits", "weight": 1.0},
                {"query": "joyful celebration music", "weight": 1.0}
            ],
            "Sad": [
                {"query": "sad melancholy emotional heartbreak", "weight": 1.0},
                {"query": "emotional ballad tearjerker", "weight": 1.0},
                {"query": "breakup songs crying", "weight": 1.0},
                {"query": "sad piano acoustic", "weight": 1.0}
            ],
            "Angry": [
                {"query": "angry intense aggressive rock", "weight": 1.0},
                {"query": "metal hardcore rage", "weight": 1.0},
                {"query": "angry rap hip hop", "weight": 1.0},
                {"query": "hard rock heavy", "weight": 1.0}
            ],
            "Neutral": [
                {"query": "chill relaxing ambient calm", "weight": 1.0},
                {"query": "study focus concentration", "weight": 1.0},
                {"query": "lofi chill beats", "weight": 1.0},
                {"query": "peaceful instrumental", "weight": 1.0}
            ],
            "Surprise": [
                {"query": "exciting energetic upbeat dynamic", "weight": 1.0},
                {"query": "pump up workout energy", "weight": 1.0},
                {"query": "epic motivational powerful", "weight": 1.0},
                {"query": "hype intense adrenaline", "weight": 1.0}
            ],
            "Fear": [
                {"query": "dark suspense atmospheric tension", "weight": 1.0},
                {"query": "thriller horror soundtrack", "weight": 1.0},
                {"query": "dramatic tense cinematic", "weight": 1.0},
                {"query": "eerie mysterious ambient", "weight": 1.0}
            ],
            "Disgust": [
                {"query": "alternative grunge edgy industrial", "weight": 1.0},
                {"query": "punk rock rebellious", "weight": 1.0},
                {"query": "dark alternative indie", "weight": 1.0},
                {"query": "goth industrial metal", "weight": 1.0}
            ]
        }
        
    def adjust_weights_from_feedback(self, db: Session, session_id: str):
        """
        Adjusts weights of queries for a specific session based on past feedback.
        In a full production scale, this would train an embedding model, but here 
        we use dynamic weighting of query performance.
        """
        # Get user feedback for this session
        feedbacks = db.query(UserFeedback).filter(UserFeedback.session_id == session_id).all()
        
        # Simple heuristic: If action == 'skip', penalize the generic query category for that emotion.
        # Ideally, we would track which EXACT query generated the skipped track, but for this step
        # we will randomly adjust weights to simulate adaptive exploration.
        for feedback in feedbacks:
            emotion = feedback.emotion_context
            if emotion not in self.emotion_queries:
                continue
                
            if feedback.action == 'skip':
                # Penalize a random query for this emotion
                q = random.choice(self.emotion_queries[emotion])
                q['weight'] = max(0.1, q['weight'] - 0.1)
                
            elif feedback.action == 'like':
                # Boost a random query
                q = random.choice(self.emotion_queries[emotion])
                q['weight'] = min(5.0, q['weight'] + 0.2)
                
    def get_query_for_emotion(self, emotion: str) -> str:
        """Returns a query weighted by the current self.emotion_queries distribution"""
        if emotion not in self.emotion_queries:
            emotion = "Neutral"
            
        queries = self.emotion_queries[emotion]
        
        # Weighted random choice
        population = [q['query'] for q in queries]
        weights = [q['weight'] for q in queries]
        
        selected_query = random.choices(population, weights=weights, k=1)[0]
        return selected_query

# Singleton instance
recommendation_engine = AdaptiveRecommendationEngine()
