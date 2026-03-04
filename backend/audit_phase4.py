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
print("PHASE 4 - STATE MANAGER AUDIT")
print("========================================")

# Reset state manager to a completely different emotion to force a transition
state_manager.current_stable_emotion = "Happy"
state_manager.pending_emotion = None
state_manager.pending_count = 0
state_manager.last_change_time = 0.0

test_dir = "/mnt/c/Users/Vishnu/test_frames"
expr = "Neutral"
img_path = os.path.join(test_dir, f"{expr}.jpg")
with open(img_path, "rb") as f:
    data_uri = f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode('utf-8')}"

print(f"\n--- Injecting 10 frames of '{expr}' into Happy StateManager ---")
for i in range(10):
    resp = client.post("/api/v1/emotion", json={"image_base64": data_uri, "session_id": "audit_sess"})
    if resp.status_code == 200:
        data = resp.json()
        print(f"  Frame {i+1} JSON: {data} | Internal PendingCount: {state_manager.pending_count}")
    else:
        print(f"  Frame {i+1} ERROR: {resp.status_code}")
    time.sleep(0.1)
