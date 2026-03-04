import sys
import mediapipe

print(f"Mediapipe version: {mediapipe.__version__}")
print(f"Mediapipe path: {mediapipe.__file__}")
print(f"Mediapipe dir: {dir(mediapipe)}")

try:
    print(f"Mediapipe solutions: {getattr(mediapipe, 'solutions')}")
except Exception as e:
    print(f"Error accessing solutions: {e}")

try:
    from mediapipe.python.solutions import face_detection
    print("Successfully imported mediapipe.python.solutions.face_detection")
except Exception as e:
    print(f"Error importing explicitly: {e}")
