import cv2
import time
import os

def capture_expressions():
    output_dir = "test_frames"
    os.makedirs(output_dir, exist_ok=True)
    
    expressions = ["Neutral", "Smile", "Angry", "Sad", "LookAway"]
    
    print("\n" + "="*50)
    print("📸 REAL-TIME INFERENCE VALIDATION CAPTURE 📸")
    print("="*50)
    print("This script will guide you through 5 expressions.")
    print("Please follow the prompts.\n")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    # Warmup camera
    for _ in range(10):
        cap.read()
        
    for expr in expressions:
        print(f"\n--- Get ready for: {expr.upper()} ---")
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        
        print("📸 SNAP!")
        ret, frame = cap.read()
        if ret:
            filepath = os.path.join(output_dir, f"{expr}.jpg")
            cv2.imwrite(filepath, frame)
            print(f"Saved {filepath}")
        else:
            print(f"Failed to capture frame for {expr}")
            
    cap.release()
    print("\n✅ All frames captured successfully!")
    print("Please inform the AI Agent to run the inference verification.")

if __name__ == "__main__":
    capture_expressions()
