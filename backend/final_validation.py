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

print("Loading final definitive model from path:")
print(model_path)
model = load_model(model_path, compile=False)

print("\n--- MODEL SUMMARY ---")
model.summary()

print("\n--- CLASSIFICATION HEAD WEIGHTS CHECK ---")
top_layers = model.layers[-5:]
for layer in top_layers:
    if hasattr(layer, 'get_weights') and layer.get_weights():
        weights = layer.get_weights()[0]
        weight_sum = np.sum(np.abs(weights))
        print(f"Layer: {layer.name}")
        print(f"  Shape: {weights.shape}")
        print(f"  Absolute Sum of Weights: {weight_sum:.4f}")

print("\n--- 3 DUMMY PREDICTIONS TEST ---")
for i in range(1, 4):
    print(f"\nEvaluating Dummy Image {i}/3...")
    dummy_tensor = np.random.rand(1, 224, 224, 3).astype('float32')
    preds = model.predict(dummy_tensor)[0]
    
    print(f"Softmax array: {np.round(preds, 4)}")
    max_idx = np.argmax(preds)
    max_val = preds[max_idx]
    print(f"Max class index: {max_idx} -> Confidence: {max_val:.4f}")
    
    is_uniform = all((abs(p - 1.0/7.0) < 0.01) for p in preds)
    if is_uniform:
        print("  ❌ STATUS: FAILED. Output remains highly uniform.")
    elif max_val > 0.40:
        print("  ✅ STATUS: SUCCESS. Non-uniform output with strong confidence.")
    else:
        print("  ⚠️ STATUS: PASSABLE. Non-uniform, but confidence is < 0.4 on random noise.")

print("\nProduction Verification complete.")
