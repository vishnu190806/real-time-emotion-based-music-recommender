import os
import time
import base64
from fastapi.testclient import TestClient

try:
    from app.main import app
except ImportError:
    import sys
    print("CRITICAL: application not found.")
    sys.exit(1)

def test_demo_stabilization_sync():
    print("="*50)
    print("🚀 API STABILIZATION DEMO TEST (TestClient)")
    print("="*50)

    # Initialize client (this will trigger the lifespan startup events synchronously!)
    print("Initializing FastAPI TestClient (Loading Models)...")
    client = TestClient(app)
    print("FastAPI App initialized successfully.")

    test_dir = "/mnt/c/Users/Vishnu/test_frames"
    expressions = ["Neutral", "Smile", "Angry", "Sad", "LookAway"]
    session_id = "demo_stabilization_test_sync"
    
    for expr in expressions:
        print(f"\n--- Testing Endpoint with {expr}.jpg ---")
        img_path = os.path.join(test_dir, f"{expr}.jpg")
        
        if not os.path.exists(img_path):
            print(f"  Missing file: {img_path}")
            continue
            
        with open(img_path, "rb") as f:
            b64_str = base64.b64encode(f.read()).decode('utf-8')
            
        data_uri = f"data:image/jpeg;base64,{b64_str}"
        
        # StateManager increased threshold from 3 to 5. We send 6 identical frames to trigger the switch.
        for frame_idx in range(6):
            start_t = time.time()
            resp = client.post("/api/v1/emotion", json={"image_base64": data_uri, "session_id": session_id})
            elapsed = (time.time() - start_t) * 1000
            
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status")
                if status == "no_face_detected":
                    print(f"  Frame {frame_idx+1}: [No Face Detected] -> Emotion: {data.get('emotion')} ({elapsed:.1f}ms)")
                else:
                    print(f"  Frame {frame_idx+1}: Raw={data.get('raw_detected_emotion')} | Stable={data.get('emotion')} | IsStable={data.get('is_stable')} | Conf={data.get('confidence'):.2f} ({elapsed:.1f}ms)")
            else:
                print(f"  Frame {frame_idx+1}: ERROR {resp.status_code} - {resp.text}")
                

if __name__ == "__main__":
    test_demo_stabilization_sync()
