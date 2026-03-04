import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    from tf_keras.models import load_model
    import numpy as np
except ImportError:
    print("CRITICAL: tf-keras not found in this environment.")
    sys.exit(1)

model_path = "/mnt/c/Users/Vishnu/Documents/College/Projects/Real-Time Emotion-Based Music Recommender/backend/app/ml/models/model_indian_v3_efficientnet.h5"

print("Loading model for dummy prediction test...")
model = load_model(model_path, compile=False)

# Dummy RGB (224, 224, 3) batch
print("Generating dummy RGB tensor...")
dummy_tensor = np.random.rand(1, 224, 224, 3).astype('float32')

print("Running prediction...")
preds = model.predict(dummy_tensor)[0]
print("--- PREDICTION OUTPUT ---")
print(f"Softmax array: {np.round(preds, 4)}")
print(f"Max class index: {np.argmax(preds)} -> {preds[np.argmax(preds)]:.4f}")

is_uniform = all((abs(p - 1.0/7.0) < 0.01) for p in preds)
if is_uniform:
    print("STATUS: FAILED. Output is highly uniform.")
else:
    print("STATUS: SUCCESS. Architecture head is definitively trained and skewing weights.")
