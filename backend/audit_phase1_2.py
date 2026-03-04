import os
import sys
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    from tf_keras.models import load_model
except ImportError:
    print("CRITICAL: tf-keras not found.")
    sys.exit(1)

model_path = "/mnt/c/Users/Vishnu/Documents/College/Projects/Real-Time Emotion-Based Music Recommender/backend/app/ml/models/model_indian_v3_efficientnet.h5"

print("========================================")
print("PHASE 1 - MODEL INTEGRITY VERIFICATION")
print("========================================")

try:
    model = load_model(model_path, compile=False)
    print(f"Model loaded: {model_path}")
    
    for i in range(5):
        dummy = np.random.rand(1, 224, 224, 3).astype("float32") * 255.0
        preds = model.predict(dummy, verbose=0)[0]
        argmax_val = np.argmax(preds)
        print(f"Prediction {i+1}:")
        print(f"  Softmax: {np.round(preds, 4)}")
        print(f"  Argmax: {argmax_val} (Conf: {preds[argmax_val]:.4f})")
except Exception as e:
    print(f"Error loading model: {e}")

print("\n========================================")
print("PHASE 2 - CLASS LABEL ORDER VERIFICATION")
print("========================================")

# We can import from the training config and inference models
sys.path.append("/mnt/c/Users/Vishnu/Documents/College/Projects/Real-Time Emotion-Based Music Recommender")

from backend.app.ml.model_inference import EfficientNetInference
inf_model = EfficientNetInference(model_path)
print(f"Inference Pipeline Classes: {inf_model.emotions}")

from scripts.training.config import EMOTIONS
print(f"Training Pipeline Classes:  {EMOTIONS}")

if inf_model.emotions == EMOTIONS:
    print("STATUS: MATCH")
else:
    print("STATUS: MISMATCH")
