# Emotion Music AI 🎭🎵

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=FastAPI&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Spotify](https://img.shields.io/badge/Spotify-1ED760?style=for-the-badge&logo=spotify&logoColor=white)

Your vibe. Your soundtrack. Powered by real-time emotion detection and AI-driven music recommendations.

---

## 🎬 Demo

Emotion Music AI strictly performs real-time emotion detection using your webcam, and automatically curates a Spotify playlist that perfectly aligns with your detected mood.

### Emotion Detection and Lanuage Selection
<img width="2559" height="1462" alt="image" src="https://github.com/user-attachments/assets/a9d86da4-757b-414b-bbc4-142632e23f41" />

### Music Recommeneded
<img width="2535" height="1462" alt="image" src="https://github.com/user-attachments/assets/23761e5f-31c9-4524-ab28-2a6f9d6f7f54" />


---

## 📖 Overview
Emotion Music AI is a high-performance, real-time music recommendation system. It uses your webcam to detect facial emotions and instantly curates personalized, mood-matching playlists from Spotify. 

This version features a ground-up backend rewrite using **FastAPI** for asynchronous, high-speed inference, **EfficientNetB0** for superior emotion classification, and a **sliding-window temporal smoother** to prevent UI flickering.


---

## ⭐ Key Features

- 🎭 **7-Emotion Detection**: Accurately detects Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise using a custom-trained EfficientNetB0 model.
- ⚡ **Intelligent Temporal Smoothing**: Uses a 15-frame sliding window (requires 8 votes to change state) to maintain stability. Includes a **High-Confidence Bypass** (>= 0.70 certainty) to react instantly to strong expressions like smiles.
- 🎵 **Spotify Deep Integration**: Dynamically fetches 8 unique tracks per request via `spotipy`. Includes built-in deduplication and randomized offsets to ensure you never get the exact same playlist twice.
- 🌍 **14+ Music Languages**: Built-in market awareness and search modifiers for Hindi, Spanish, Korean (K-Pop), Japanese (J-Pop), etc.
- 🎨 **Modern Glassmorphic UI**: React + Vite frontend featuring real-time webcam overlays, particle backgrounds, and dynamic gradient shifts based on your mood.

---

## 🏗️ System Architecture

### 📁 Project Structure
```text
emotion-music-ai/
├── backend/
│   ├── api/
│   │   └── server.py                  # FastAPI application & endpoints
│   ├── emotion_inference/
│   │   └── emotion_predictor.py       # OpenCV Face Detection + TF Inference
│   ├── emotion_smoothing/
│   │   └── emotion_smoother.py        # Sliding window & high-confidence logic
│   └── music_engine/
│       └── music_mapper.py            # Wrapper for the Spotify engine
├── components/                            # React Frontend Components
├── models/                                # Trained Neural Networks (.h5 & SavedModel)
├── App.tsx                                # Main React Application
└── spotify_helper.py                      # Core Spotify API integration logic
```

### ⚙️ The Pipeline
1. **Frontend**: The React app captures webcam frames at 250ms intervals and sends them as Base64 encoded images.
2. **Inference**: The FastAPI backend decodes the image. OpenCV's Haar Cascade detects the face, which is then cropped, resized, and fed into the EfficientNetB0 TensorFlow model.
3. **Smoothing**: The `EmotionSmoother` evaluates the prediction. If confidence >= 0.70, it updates immediately. Otherwise, it contributes to a 15-frame rolling average.
4. **Music Engine**: When you click "Discover Music," the `spotify_helper` constructs a targeted search query based on emotion and language, grabbing 8 unique tracks from the Spotify catalog.

---

## 🚀 Installation & Setup

### 1. Prerequisites
- **Windows OS** (Primary environment for inference & UI)
- **Python 3.10+** (For the GPU-enabled `venv_gpu` environment)
- **Node.js 18+** (For the Vite frontend)
- **WSL2 (Ubuntu)** (For the one-time model conversion script)
- **Spotify Developer Account** (For API keys)

### 2. Configuration
1. Clone the repository to your local machine.
2. Create a `.env` file in the project root:
   ```env
   SPOTIFY_CLIENT_ID=your_spotify_client_id
   SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
   ```

### 3. Model Conversion (One-Time Only)
Because the model was trained in WSL2 (Keras 2) but is run on Windows (Keras 3), you must perform a one-time conversion to the universal `SavedModel` format.
Open your **WSL2 terminal**:
```bash
source ~/tf_gpu_env/bin/activate
cd /mnt/c/path/to/project
python convert_model.py
```
This will generate `models/emotion_saved_model/` for high-speed Keras 3 compatibility.

---

## ⚡ Quick Start

**Start Backend**
```powershell
.\venv_gpu\Scripts\python.exe -m uvicorn backend.api.server:app --host 127.0.0.1 --port 8000
```

**Start Frontend**
```bash
npm run dev
```

**Open Application**
http://localhost:3000

---

## 💻 Running the Application

You will need two terminals running simultaneously.

### Terminal 1: Start the Backend (FastAPI)
Open a Windows PowerShell in the project directory:
```powershell
.\venv_gpu\Scripts\python.exe -m uvicorn backend.api.server:app --host 127.0.0.1 --port 8000
```
*The server will start at `http://127.0.0.1:8000`.*

### Terminal 2: Start the Frontend (Vite)
Open another terminal:
```bash
npm install
npm run dev
```
*Open `http://localhost:3000` in your browser.*

---

## 📡 API Endpoints (v1)

The backend exposes a fully asynchronous REST API:

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/session` | Initializes a unique session ID for temporal smoothing. |
| `GET` | `/api/v1/languages` | Returns the list of supported Spotify music languages. |
| `POST` | `/api/v1/emotion` | Accepts `image_base64`. Returns the smoothed `{emotion, confidence}`. |
| `POST` | `/api/v1/recommend`| Accepts `{emotion, language}`. Returns 8 unique track objects. |

---

## 🛠️ Customization

- **Tune Smoothing**: Adjust the `window_size`, `confidence_threshold`, and `required_frames` in `backend/api/server.py`.
- **Add Languages**: Add new regions and search modifiers to the `LANGUAGE_CONFIG` list inside `spotify_helper.py`.
- **Emotion Queries**: Update the music tags mapped to each emotion in the `EMOTION_TO_SEARCH` vocabulary in `spotify_helper.py`.

---

## 📝 License
MIT License. Created for the Real-Time Emotion-Based Music Recommender project.
