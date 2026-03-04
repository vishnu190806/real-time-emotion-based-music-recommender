import cv2
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional

from app.core.config import settings
from app.core.logging import logger

from collections import deque
from scipy.stats import entropy

# Initialize class-level bias tracking across all instances
_prediction_history = deque(maxlen=500)

class EmotionModel(ABC):
    """Abstract base class for emotion detection models.
    Allows for easy swapping between Edge (TFLite) and Server (ONNX/H5) models."""
    
    @abstractmethod
    def load(self) -> bool:
        pass
        
    @abstractmethod
    def predict(self, frame_roi: np.ndarray) -> Tuple[str, float]:
        """Returns (EmotionName, ConfidenceScore)"""
        pass

class EfficientNetInference(EmotionModel):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    def load(self) -> bool:
        try:
            import tensorflow as tf
            from tensorflow.keras.models import load_model
            
            model_file = Path(self.model_path).resolve()
            logger.info(f"Loading EfficientNet model from {model_file}")
            
            if not model_file.exists():
                logger.error(f"CRITICAL: EfficientNet model not found at {model_file}")
                return False
                
            self.model = load_model(
                str(model_file),
                compile=False,
                safe_mode=False
            )
            print("MODEL LOADED SUCCESSFULLY")
            return True
            
        except Exception as e:
            print("FULL MODEL LOAD ERROR:")
            import traceback
            traceback.print_exc()
            raise

    def predict(self, face_roi_bgr: np.ndarray) -> Tuple[str, float]:
        # ── FIX 5: SAFE FALLBACK ──
        if self.model is None or face_roi_bgr is None or face_roi_bgr.size == 0:
            return ("Neutral", 0.0)
            
        from tensorflow.keras.applications.efficientnet import preprocess_input
            
        try:
            # ── FIX 1: COLOR DISTRIBUTION MISMATCH ──
            # Training images were grayscale duplicated across 3 channels.
            
            # 1. Convert BGR to Grayscale
            gray = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2GRAY)
            
            # 2. Resize grayscale to EfficientNet input spec
            gray_resized = cv2.resize(gray, (224, 224))
            
            # 3. Recreate 3-channel tensor (duplicate grayscale across RGB channels)
            gray_3ch = np.stack([gray_resized, gray_resized, gray_resized], axis=-1)
            
            # 4. Convert dtype to float32
            img_float = gray_3ch.astype("float32")
            
            # 5. Apply EfficientNet normalization
            img_preprocessed = preprocess_input(img_float)
            
            # 6. Add batch dimension -> (1, 224, 224, 3)
            roi_input = np.expand_dims(img_preprocessed, axis=0)
            
            # 5. Run Inference
            raw_preds = self.model.predict(roi_input, verbose=0)[0]
            
            # ── AUTOMATIC CALIBRATION (Temperature Scaling via Entropy) ──
            # Calculate prediction certainty (entropy)
            pred_entropy = entropy(raw_preds)
            max_entropy = np.log(len(self.emotions))
            normalized_entropy = pred_entropy / max_entropy
            
            # Dynamic temperature: Higher entropy -> Lower T (Sharpen), Lower entropy -> Higher T (Soften overconfidence)
            # We want to specifically soften the model when it's overwhelmingly confident (usually a sign of bias)
            temperature = 1.0 + (1.0 - normalized_entropy) * 0.5 
            
            # Apply scaling
            logits = np.log(raw_preds + 1e-9)
            scaled_logits = logits / temperature
            calibrated_preds = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits))
            
            preds = calibrated_preds
            
            # ── DIAGNOSTIC: Full softmax + top-2 ──
            sorted_indices = np.argsort(preds)[::-1]
            top1_idx = sorted_indices[0]
            top2_idx = sorted_indices[1]
            
            softmax_str = np.array2string(preds, precision=4, separator=', ')
            
            print("\n----------------------------------------")
            print(f"Calibration Temp: {temperature:.3f} (Entropy: {normalized_entropy:.3f})")
            print(f"Softmax: {softmax_str}")
            print(f"Top1: {self.emotions[top1_idx]} ({float(preds[top1_idx]):.4f})")
            print(f"Top2: {self.emotions[top2_idx]} ({float(preds[top2_idx]):.4f})")
            print("----------------------------------------")
            
            top1_conf = float(preds[top1_idx])
            top2_conf = float(preds[top2_idx])
            top1_emotion = self.emotions[top1_idx]
            
            # Track predictions for bias detection
            _prediction_history.append(top1_emotion)
            
            # Periodically report bias automatically
            if len(_prediction_history) % 100 == 0:
                from collections import Counter
                counts = Counter(_prediction_history)
                print("\n[BIAS MONITOR] Recent 500 Predictions Distribution:")
                for em, count in counts.most_common():
                    freq = count / len(_prediction_history)
                    print(f"  {em}: {freq*100:.1f}%")
                    if freq > 0.35:
                        logger.warning(f"⚠️ POTENTIAL BIAS DETECTED: {em} is dominating ({freq*100:.1f}%)")
            
            # ── FIX 5: BIAS CORRECTION (CRITICAL) ──
            # Detect pattern where Disgust frequently beats Happy with small margins due to dataset bias
            happy_idx = self.emotions.index("Happy")
            happy_conf = float(preds[happy_idx])
            
            if top1_emotion == "Disgust" and happy_conf > 0.25 and (top1_conf - happy_conf) < 0.20:
                print("========================================")
                print(f"BIAS CORRECTION TRIGGERED: Disgust ({top1_conf:.3f}) OVERRIDDEN TO Happy ({happy_conf:.3f})")
                print("========================================")
                logger.warning(f"Bias correction applied: Disgust ({top1_conf:.3f}) -> Happy ({happy_conf:.3f})")
                
                detected_emotion = "Happy"
                confidence = happy_conf
            
            # ── UNCERTAINTY FILTER ──
            # If top probabilities are too close, treat as uncertain regardless of raw confidence
            elif (top1_conf - top2_conf) < 0.15:
                detected_emotion = "Unknown"
                confidence = top1_conf
                
            # ── LOW CONFIDENCE FILTER ──
            elif top1_conf < 0.40:
                detected_emotion = "Unknown"
                confidence = top1_conf
            else:
                detected_emotion = top1_emotion
                confidence = top1_conf
                
            logger.info(f"Raw predicted emotion: {detected_emotion} (Confidence: {confidence:.2f})")
            
            return detected_emotion, confidence
            
        except Exception as e:
            logger.error(f"Inference pipeline failed during preprocessing/prediction: {e}")
            return ("Neutral", 0.0)


class FaceDetector:
    def __init__(self, cascade_path: str):
        self.cascade_path = cascade_path # Kept for signature compatibility
        self.detector = None
        
    def load(self) -> bool:
        if self.detector is not None:
            return True
            
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            base_options = python.BaseOptions(model_asset_path='app/ml/models/blaze_face_short_range.tflite')
            options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
            self.detector = vision.FaceDetector.create_from_options(options)
            return True
        except Exception as e:
            logger.error(f"CRITICAL: Failed to load MediaPipe: {e}")
            return False
            
    def detect(self, image_input: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if self.detector is None:
            return None
            
        # Defensively handle BGR vs GRAY depending on upstream pipeline
        if len(image_input.shape) == 2 or image_input.shape[2] == 1:
            image_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
        results = self.detector.detect(mp_image)
        
        if not results.detections:
            return None
            
        detection = results.detections[0]
        bbox = detection.bounding_box
        
        iw, ih = image_rgb.shape[1], image_rgb.shape[0]
        
        # Clamp boundaries safely to prevent OpenCV slice out-of-bounds
        x = max(0, bbox.origin_x)
        y = max(0, bbox.origin_y)
        w = min(bbox.width, iw - x)
        h = min(bbox.height, ih - y)
        
        if w <= 0 or h <= 0:
            return None
            
        # print(f"[MP DETECT] bbox: {x},{y},{w},{h}")
        return (x, y, w, h)

# Singleton instances for DI
# Kept pure without lazy-loading to enforce Startup initialization
model_instance = EfficientNetInference(settings.MODEL_PATH)
face_detector_instance = FaceDetector(settings.CASCADE_PATH)

def get_emotion_model() -> EmotionModel:
    return model_instance
    
def get_face_detector() -> FaceDetector:
    return face_detector_instance
