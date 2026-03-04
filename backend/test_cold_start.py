import base64
import requests
import cv2
import numpy as np
import time

def run_test():
    print("Cold Start Test Initiated...")
    
    # Create dummy face to trigger cascade
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(img, (320, 240), 100, (255, 255, 255), -1)
    
    _, buffer = cv2.imencode('.jpg', img)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    data_uri = f"data:image/jpeg;base64,{b64_str}"
    
    sess_id = "test_script_session_" + str(int(time.time()))
    
    stable_emotion = "Neutral"
    
    print("1. Sending 10 frames to stabilize EmotionStateManager...")
    for i in range(10):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/api/v1/emotion",
                json={"image_base64": data_uri, "session_id": sess_id},
                timeout=2.0
            )
            data = response.json() if response.status_code == 200 else {}
            if response.status_code == 200:
                print(f"  Frame {i+1}: Raw={data.get('raw_detected_emotion')}, Stable={data.get('emotion')}, IsStable={data.get('is_stable')}")
                stable_emotion = data.get('emotion', 'Neutral')
        except Exception as e:
            print(f"  Frame {i+1} Request Failed: {e}")
            
    print(f"2. Fetching Spotify Tracks for Emotion: {stable_emotion}...")
    try:
        response = requests.post(
            "http://127.0.0.1:8000/api/v1/recommend",
            json={"emotion": stable_emotion, "language": "Mixed", "session_id": sess_id},
            timeout=5.0
        )
        data = response.json()
        tracks = data.get('tracks', [])
        print(f"  Received {len(tracks)} tracks.")
        if len(tracks) > 0:
            print(f"  Top Track: {tracks[0]['name']} by {tracks[0]['artist']}")
    except Exception as e:
        print(f"  Recommend Request Failed: {e}")
        
    print("Cold Start Test Complete.\n")

if __name__ == "__main__":
    run_test()
