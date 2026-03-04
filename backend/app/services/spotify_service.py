import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import random
import time
from typing import List, Dict, Any

from app.core.config import settings
from app.core.logging import logger
from app.services.recommendation_engine import recommendation_engine

LANGUAGE_CONFIG = {
    "Mixed": {"markets": ["US", "IN", "GB", "BR", "KR", "JP", "ES", "FR", "DE", "IT"], "search_suffix": ""},
    "English": {"markets": ["US", "GB", "CA", "AU"], "search_suffix": ""},
    "Hindi": {"markets": ["IN"], "search_suffix": " hindi bollywood"},
    "Spanish": {"markets": ["ES", "MX", "AR"], "search_suffix": " spanish latino"},
    "Korean": {"markets": ["KR"], "search_suffix": " kpop korean"},
    "Japanese": {"markets": ["JP"], "search_suffix": " jpop japanese"},
    "Portuguese": {"markets": ["BR", "PT"], "search_suffix": " portuguese brasileiro"},
    "French": {"markets": ["FR"], "search_suffix": " french"},
    "Arabic": {"markets": ["AE", "SA", "EG"], "search_suffix": " arabic"},
    "German": {"markets": ["DE", "AT"], "search_suffix": " german deutsch"},
    "Italian": {"markets": ["IT"], "search_suffix": " italian italiano"},
    "Chinese": {"markets": ["TW", "HK"], "search_suffix": " mandarin chinese cpop"},
    "Tamil": {"markets": ["IN"], "search_suffix": " tamil kollywood"},
    "Telugu": {"markets": ["IN"], "search_suffix": " telugu tollywood"}
}

class SpotifyService:
    def __init__(self):
        self.enabled = False
        try:
            if settings.SPOTIFY_CLIENT_ID and settings.SPOTIFY_CLIENT_SECRET:
                auth_manager = SpotifyClientCredentials(
                    client_id=settings.SPOTIFY_CLIENT_ID,
                    client_secret=settings.SPOTIFY_CLIENT_SECRET
                )
                self.sp = spotipy.Spotify(auth_manager=auth_manager)
                self.enabled = True
                logger.info("Spotify client initialized successfully")
            else:
                logger.warning("Spotify credentials missing, service disabled")
        except Exception as e:
            logger.error(f"Failed to initialize Spotify: {e}")

    def get_tracks_for_emotion(self, emotion: str, language: str = "Mixed", limit: int = 5) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []

        lang_config = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG["Mixed"])
        market = random.choice(lang_config["markets"])
        search_suffix = lang_config["search_suffix"]

        # Use adaptive engine for base query
        base_query = recommendation_engine.get_query_for_emotion(emotion)
        full_query = base_query + search_suffix

        # Randomize offset for diversity
        random_seed = int(time.time() * 1000) % 10000
        random.seed(random_seed)

        all_items = []
        try:
            for _ in range(2):
                search_offset = random.randint(0, 100)
                results = self.sp.search(
                    q=full_query, 
                    type='track', 
                    limit=limit * 3,
                    market=market,
                    offset=search_offset
                )
                items = (results or {}).get('tracks', {}).get('items', []) or []
                all_items.extend(items)
                
            random.shuffle(all_items)
            
            tracks = []
            seen_track_names = set()
            artist_count = {}
            
            items_sorted = sorted(
                all_items, 
                key=lambda x: x.get('popularity', 0) + random.randint(-20, 20), 
                reverse=True
            )
            
            for t in items_sorted:
                if len(tracks) >= limit:
                    break
                    
                name = (t or {}).get('name') or "Untitled"
                artists = (t or {}).get('artists') or []
                artist = (artists[0].get('name') if artists and artists[0] else "Unknown")
                
                track_key = f"{name.lower()}_{artist.lower()}"
                if track_key in seen_track_names or artist_count.get(artist, 0) >= 2:
                    continue
                    
                album_obj = (t or {}).get('album') or {}
                album_images = album_obj.get('images') or []
                album_art = next((img['url'] for img in album_images if img.get('height') == 300), None)
                if not album_art and album_images:
                    album_art = album_images[0].get('url')
                    
                tracks.append({
                    'name': name,
                    'artist': artist,
                    'album': album_obj.get('name', ''),
                    'album_art': album_art,
                    'preview_url': (t or {}).get('preview_url'),
                    'url': ((t or {}).get('external_urls') or {}).get('spotify', ''),
                    'uri': (t or {}).get('uri', ''),
                    'popularity': (t or {}).get('popularity', 0),
                    'query_used': base_query # Tracing for feedback adjustments
                })
                
                seen_track_names.add(track_key)
                artist_count[artist] = artist_count.get(artist, 0) + 1
                
            random.seed()
            return tracks
            
        except Exception as e:
            logger.error(f"Error fetching tracks from Spotify: {e}")
            random.seed()
            return []

spotify_service = SpotifyService()
