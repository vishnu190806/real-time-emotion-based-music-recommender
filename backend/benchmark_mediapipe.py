import os
import cv2
import time
import sys

# Load TF environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    from app.ml.model_inference import get_face_detector
except ImportError:
    sys.path.append("/mnt/c/Users/Vishnu/Documents/College/Projects/Real-Time Emotion-Based Music Recommender/backend")
    from app.ml.model_inference import get_face_detector

detector = get_face_detector()
if not detector.load():
    print("CRITICAL: Failed to load Face Detector")
    sys.exit(1)

test_dir = "/mnt/c/Users/Vishnu/test_frames"
expressions = ["Neutral", "Smile", "Angry", "Sad", "LookAway"]

print("========================================")
print("PHASE 5 - PERFORMANCE & STABILITY CHECK")
print("========================================")

total_time = 0.0
success_count = 0

for expr in expressions:
    img_path = os.path.join(test_dir, f"{expr}.jpg")
    if not os.path.exists(img_path):
        print(f"Skipping {expr} - Image not found.")
        continue
        
    img = cv2.imread(img_path)
    if img is None:
        continue
        
    start_t = time.perf_counter()
    bbox = detector.detect(img)
    end_t = time.perf_counter()
    
    elapsed_ms = (end_t - start_t) * 1000
    total_time += elapsed_ms
    
    if bbox is not None:
        success_count += 1
        print(f"[{expr:10s}] SUCCESS | Time: {elapsed_ms:.2f}ms | BBox: {bbox}")
    else:
        print(f"[{expr:10s}] FAILED  | Time: {elapsed_ms:.2f}ms | BBox: None")

print("\n--- Summary ---")
print(f"Total Frames: {len(expressions)}")
print(f"Success Rate: {(success_count / len(expressions)) * 100:.1f}%")
print(f"Avg Inference Time: {total_time / len(expressions):.2f}ms")
