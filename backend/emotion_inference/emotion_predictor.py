import os
import cv2
import json
import numpy as np

class EmotionPredictor:
    def __init__(self, model_path: str, labels_path: str):
        print(f"Loading Emotion Model from {model_path}...")

        # Prefer TF SavedModel directory (Keras 3 / TF 2.20+ fully compatible).
        # Run convert_model.py in WSL2 once to generate emotion_saved_model/.
        base_dir = os.path.dirname(model_path)
        saved_model_dir = os.path.join(base_dir, "emotion_saved_model")

        if os.path.isdir(saved_model_dir):
            print(f"Loading TF SavedModel from: {saved_model_dir}")
            import tensorflow as tf
            self._tf_model = tf.saved_model.load(saved_model_dir)
            self._infer = self._tf_model.signatures["serving_default"]
            self._use_signature = True
        elif os.path.exists(model_path):
            import keras
            print("Warning: Falling back to h5 (may fail on Keras 3)...")
            self._tf_model = keras.saving.load_model(model_path, safe_mode=False)
            self._use_signature = False
        else:
            raise FileNotFoundError(
                f"No model found at {saved_model_dir} or {model_path}. "
                "Run convert_model.py in WSL2 first."
            )

        # Load labels
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found at {labels_path}")
        with open(labels_path, "r") as f:
            self.class_names = json.load(f)
        print(f"✅ Model loaded successfully. Classes: {self.class_names}")

        # Initialize OpenCV Haar Cascade (bundled with opencv-contrib)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        print(f"✅ Face detector loaded from: {cascade_path}")

    def _detect_faces(self, bgr_image):
        """Returns list of (x, y, w, h) face rectangles from a BGR frame."""
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        # Mild CLAHE enhancement for better detection in variable lighting
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)
        faces = self.face_cascade.detectMultiScale(
            gray_enhanced,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30)
        )
        return list(faces) if len(faces) > 0 else []

    def preprocess_face(self, face_image_rgb):
        """Resize RGB face crop to (224,224) float32 for EfficientNetB0."""
        resized = cv2.resize(face_image_rgb, (224, 224))
        img_array = np.array(resized, dtype=np.float32)
        return np.expand_dims(img_array, axis=0)

    def _run_inference(self, input_batch):
        """Dispatch to the right inference backend and return predictions array."""
        if self._use_signature:
            import tensorflow as tf
            output = self._infer(tf.constant(input_batch))
            # output key is model-dependent; pick first
            out_key = list(output.keys())[0]
            return output[out_key].numpy()[0]
        else:
            return self._tf_model.predict(input_batch, verbose=0)[0]

    def predict(self, bgr_image: np.ndarray):
        """
        Takes an OpenCV BGR image.
        Returns {"emotion": str, "confidence": float, "face_detected": bool, "bbox": dict|None}
        """
        faces = self._detect_faces(bgr_image)
        if not faces:
            return {"emotion": "Unknown", "confidence": 0.0, "face_detected": False, "bbox": None}

        # Use the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # Crop + to RGB for EfficientNetB0 (trained on RGB)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        xMin, yMin = max(0, x), max(0, y)
        xMax, yMax = min(bgr_image.shape[1], x + w), min(bgr_image.shape[0], y + h)
        face_crop = rgb_image[yMin:yMax, xMin:xMax]

        if face_crop.size == 0:
            return {"emotion": "Unknown", "confidence": 0.0, "face_detected": False, "bbox": None}

        input_tensor = self.preprocess_face(face_crop)
        predictions = self._run_inference(input_tensor)

        max_idx = int(np.argmax(predictions))
        confidence = float(predictions[max_idx])
        # Labels are returned lowercase from training ('happy'); capitalize for the frontend
        emotion = self.class_names[max_idx].capitalize()

        print("\n=== MODEL DEBUG ===")
        print(f"Predicted emotion: {emotion.lower()}")
        print(f"Confidence: {confidence:.4f}")
        print(f"All probabilities: {predictions}")
        print("===================\n")

        return {
            "emotion": emotion,
            "confidence": confidence,
            "face_detected": True,
            "bbox": {"x": xMin, "y": yMin, "w": w, "h": h}
        }
