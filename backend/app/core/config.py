import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Emotion Music AI Production"
    API_V1_STR: str = "/api/v1"
    
    # SQLite default, or can be overridden with Postgres URI via .env
    DATABASE_URL: str = "sqlite:///./emotion_music.db"
    
    # Spotify
    SPOTIFY_CLIENT_ID: str = ""
    SPOTIFY_CLIENT_SECRET: str = ""
    
    # ML settings
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    MODEL_PATH: str = str(BASE_DIR / "backend" / "app" / "ml" / "models" / "model_indian_v3_efficientnet.h5")
    CASCADE_PATH: str = str(BASE_DIR / "backend" / "app" / "ml" / "assets" / "haarcascade_frontalface_default.xml")
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
