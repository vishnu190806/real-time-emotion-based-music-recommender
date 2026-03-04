import os
import sys
import base64
import time

try:
    from fastapi.testclient import TestClient
    from app.main import app
    from app.api.endpoints import state_manager
except ImportError:
    print("CRITICAL: application not found.")
    sys.exit(1)

client = TestClient(app)

print("========================================")
print("PHASE 3 - RAW INFERENCE TEST")
print("========================================")

test_dir = "/mnt/c/Users/Vishnu/test_frames"
expressions = ["Neutral", "Smile", "Angry", "Sad", "LookAway"]
session_id = "debug_audit_session"

# We will just run 3 frames each to show raw output
for expr in expressions:
    img_path = os.path.join(test_dir, f"{expr}.jpg")
    if not os.path.exists(img_path):
        continue
    
    with open(img_path, "rb") as f:
        data_uri = f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode('utf-8')}"
        
    print(f"\n--- Testing Raw Inference: {expr} ---")
    for i in range(3):
        resp = client.post("/api/v1/emotion", json={"image_base64": data_uri, "session_id": session_id})
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "no_face_detected":
                print(f"  Frame {i+1}: NO FACE DETECTED")
            else:
                print(f"  Frame {i+1}: Raw={data.get('raw_detected_emotion')}, Conf={data.get('confidence'):.4f}")
                
print("\n========================================")
print("PHASE 4 - STATE MANAGER AUDIT")
print("========================================")

# Reset state manager
state_manager.current_stable_emotion = "Neutral"
state_manager.pending_emotion = None
state_manager.pending_count = 0
# We also wipe the cooldown to test immediate transitions
state_manager.last_change_time = 0.0

expr = "Smile"
img_path = os.path.join(test_dir, f"{expr}.jpg")
with open(img_path, "rb") as f:
    data_uri = f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode('utf-8')}"

print(f"\n--- Injecting 10 frames of '{expr}' into Neutral StateManager ---")
for i in range(10):
    resp = client.post("/api/v1/emotion", json={"image_base64": data_uri, "session_id": session_id})
    if resp.status_code == 200:
        data = resp.json()
        raw = data.get('raw_detected_emotion')
        stable = data.get('emotion')
        pending_count = state_manager.pending_count
        print(f"  Frame {i+1}: Raw={raw} | Stable={stable} | PendingCount={pending_count}")
