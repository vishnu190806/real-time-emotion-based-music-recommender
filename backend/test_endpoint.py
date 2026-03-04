import base64
import requests
import cv2
import numpy as np

# Create a dummy image mimicking a webcam capture (e.g. 640x480 black image with a drawn white circle as a "face")
img = np.zeros((480, 640, 3), dtype=np.uint8)
# Draw a face-like structure so haarcascade might pick it up, or just send a blank one to see if the pipeline throws an error
cv2.circle(img, (320, 240), 100, (255, 255, 255), -1)

# Encode to base64
_, buffer = cv2.imencode('.jpg', img)
b64_str = base64.b64encode(buffer).decode('utf-8')
data_uri = f"data:image/jpeg;base64,{b64_str}"

try:
    response = requests.post(
        "http://127.0.0.1:8000/api/v1/emotion",
        json={"image_base64": data_uri, "session_id": "test_script_session"}
    )
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())
except Exception as e:
    print("Request failed:", e)
