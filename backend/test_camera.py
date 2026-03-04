import cv2
import numpy as np
from app.ml.model_inference import get_face_detector

def test_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found!")
        return
        
    success, frame = cap.read()
    cap.release()
    
    if not success:
        print("Failed to read frame!")
        return
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)
    gray_enhanced = cv2.convertScaleAbs(gray_enhanced, alpha=1.2, beta=10)

    face_detector = get_face_detector()
    face_coords = face_detector.detect(gray_enhanced)
    
    print("Face found:", face_coords)

if __name__ == "__main__":
    test_camera()
