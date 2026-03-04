import os
import sys

# Suppress annoying TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    from tf_keras.models import load_model
    import numpy as np
except ImportError:
    print("CRITICAL: tf-keras not found in this environment.")
    sys.exit(1)

model_path = "/mnt/c/Users/Vishnu/Documents/College/Projects/Real-Time Emotion-Based Music Recommender/backend/app/ml/models/model_indian_v3_efficientnet.h5"

if not os.path.exists(model_path):
    print(f"CRITICAL: Model file missing at {model_path}")
    sys.exit(1)

print(f"Loading model from {model_path}...")
model = load_model(model_path, compile=False)

print("\n--- MODEL SUMMARY ---")
model.summary()

print("\n--- CLASSIFICATION HEAD WEIGHTS CHECK ---")
top_layers = model.layers[-5:]
for layer in top_layers:
    if hasattr(layer, 'get_weights') and layer.get_weights():
        weights = layer.get_weights()[0]
        biases = layer.get_weights()[1] if len(layer.get_weights()) > 1 else None
        
        weight_sum = np.sum(np.abs(weights))
        print(f"Layer: {layer.name}")
        print(f"  Shape: {weights.shape}")
        print(f"  Absolute Sum of Weights: {weight_sum:.4f}")
        
        if weight_sum < 1e-5:
            print("  WARNING: Weights are nearly zero. Layer might be initialized but untrained.")
        else:
            print("  STATUS: Weights contain non-zero data.")

print("\nDiagnostic complete.")
