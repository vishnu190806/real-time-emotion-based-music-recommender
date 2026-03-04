import pytest
from app.ml.state_manager import EmotionStateManager, Emotion

def test_initial_state():
    manager = EmotionStateManager(window_size=5, stability_threshold=3, confidence_threshold=0.4)
    assert manager.current_stable_emotion == Emotion.NEUTRAL

def test_ignore_low_confidence():
    manager = EmotionStateManager(window_size=5, stability_threshold=3, confidence_threshold=0.5)
    stable, is_stable = manager.process_frame("Happy", 0.3)
    
    assert stable == Emotion.NEUTRAL
    assert is_stable == False
    assert len(manager.emotion_window) == 0

def test_stability_transition():
    manager = EmotionStateManager(window_size=5, stability_threshold=3, confidence_threshold=0.4)
    
    # Send 1 "Happy" frame - shouldn't transition yet
    stable, is_stable = manager.process_frame("Happy", 0.8)
    assert stable == Emotion.NEUTRAL
    assert is_stable == False
    
    # Send 2nd "Happy" frame
    stable, is_stable = manager.process_frame("Happy", 0.8)
    assert stable == Emotion.NEUTRAL
    assert is_stable == False
    
    # Send 3rd "Happy" frame - threshold reached!
    stable, is_stable = manager.process_frame("Happy", 0.8)
    assert stable == Emotion.HAPPY
    assert is_stable == True

def test_erratic_fluctuation():
    manager = EmotionStateManager(window_size=5, stability_threshold=3, confidence_threshold=0.4)
    
    # Alternate Happy and Sad
    manager.process_frame("Happy", 0.8)
    manager.process_frame("Sad", 0.8)
    manager.process_frame("Happy", 0.8)
    stable, is_stable = manager.process_frame("Sad", 0.8)
    
    # Should stay Neutral
    assert stable == Emotion.NEUTRAL
    assert is_stable == False

def test_reverting_to_stable_resets_pending():
    manager = EmotionStateManager(window_size=5, stability_threshold=4, confidence_threshold=0.4)
    
    # Establish Happy as stable
    for _ in range(4):
        manager.process_frame("Happy", 0.8)
        
    assert manager.current_stable_emotion == Emotion.HAPPY
    
    # 2 Sad frames (not enough to switch)
    manager.process_frame("Sad", 0.8)
    manager.process_frame("Sad", 0.8)
    assert manager.current_stable_emotion == Emotion.HAPPY
    
    # Revert to Happy 
    stable, is_stable = manager.process_frame("Happy", 0.8)
    
    # Pending should be reset, and we are back to stable Happy
    assert stable == Emotion.HAPPY
    assert is_stable == True
    assert manager.pending_count == 0
