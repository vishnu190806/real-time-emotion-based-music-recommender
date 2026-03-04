import os
import sys
import cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    from app.ml.model_inference import EfficientNetInference, FaceDetector
    from app.core.config import settings
except ImportError:
    print("CRITICAL: Application modules not found. Run from backend root.")
    sys.exit(1)

def run_validation():
    print("="*50)
    print("🧠 PRODUCTION INFERENCE VALIDATION (REAL WEBCAM FRAMES)")
    print("="*50)
    
    # 1. Load production architecture components
    model = EfficientNetInference(settings.MODEL_PATH)
    if not model.load():
        print("Failed to load EfficientNet model.")
        sys.exit(1)
        
    detector = FaceDetector(settings.CASCADE_PATH)
    if not detector.load():
        print("Failed to load Face Detector.")
        sys.exit(1)
        
    expressions = ["Neutral", "Smile", "Angry", "Sad", "LookAway"]
    test_dir = "/mnt/c/Users/Vishnu/test_frames"
    
    results = []
    
    for expr in expressions:
        img_path = os.path.join(test_dir, f"{expr}.jpg")
        print(f"\n--- Analyzing Frame: {expr} ---")
        
        if not os.path.exists(img_path):
            print(f"  Missing file: {img_path}")
            continue
            
        frame = cv2.imread(img_path)
        if frame is None:
            print("  Corrupted file.")
            continue
            
        # Matching exact /emotion endpoint preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Face detect
        face_roi = detector.detect(gray)
        if face_roi is None:
            print("  [DEBUG] No face detected in frame.")
            continue
            
        x, y, w, h = face_roi
        cropped_rgb = rgb[y:y+h, x:x+w]
        
        # Inference
        detected_emotion, confidence = model.predict(cropped_rgb)
        
        # By default model.predict logs in inference pipeline, but we also want the raw softmax:
        # We will manually do the raw softmax lookup to satisfy the audit requirement
        from tf_keras.applications.efficientnet import preprocess_input
        roi_resized = cv2.resize(cropped_rgb, (224, 224)).astype("float32")
        roi_preprocessed = preprocess_input(roi_resized)
        roi_input = np.expand_dims(roi_preprocessed, axis=0)
        raw_softmax = model.model.predict(roi_input, verbose=0)[0]
        
        print(f"  Raw Softmax : {np.round(raw_softmax, 4)}")
        print(f"  Predicted   : {detected_emotion}")
        print(f"  Confidence  : {confidence:.4f}")
        
        results.append({
            "expr": expr,
            "prediction": detected_emotion,
            "confidence": confidence,
            "softmax": raw_softmax
        })

    print("\n" + "="*50)
    print("📊 VALIDATION STATISTICS")
    print("="*50)
    
    if len(results) == 0:
        print("No faces detected across all frames!")
        return
        
    confidences = [r["confidence"] for r in results]
    mean_conf = np.mean(confidences)
    var_conf = np.var(confidences)
    
    print(f"Frames Analyzed: {len(results)}/{len(expressions)}")
    print(f"Mean Confidence: {mean_conf:.4f}")
    print(f"Confidence Variance: {var_conf:.4f}")
    
    if len(set(r["prediction"] for r in results)) > 1:
        print("✅ PASS: Predictions successfully mutate across different visual expressions.")
    else:
        print("⚠️ WARNING: Classifier was anchored to a single class. May require boundary tuning.")
        
    if mean_conf > 0.4:
        print("✅ PASS: Head is confident in explicit features.")
    else:
        print("⚠️ WARNING: Low overall confidence threshold.")

if __name__ == "__main__":
    run_validation()
