import os
import sys

# Append parent dir so we can import the spotify_helper gracefully from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from spotify_helper import SpotifyMoodRecommender
    print("✅ Successfully linked to existing Spotify Recommender")
except ImportError as e:
    print(f"❌ Failed to find spotify_helper: {e}")
    SpotifyMoodRecommender = None

class MusicMapper:
    def __init__(self):
        self.spotify = None
        if SpotifyMoodRecommender:
            try:
                self.spotify = SpotifyMoodRecommender()
            except Exception as e:
                print(f"Initialization of Spotify failed (likely missing .env credentials): {e}")

    def get_recommendation(self, emotion: str, language: str = "Mixed", limit: int = 8):
        """
        Pulls a dynamic list of tracks for the detected emotion and language.
        Gracefully handles API failures by returning empty arrays.
        """
        if not self.spotify:
            return []
            
        try:
            tracks = self.spotify.get_tracks_for_emotion(emotion, limit=limit, language=language)
            return tracks
        except Exception as e:
            print(f"Error fetching tracks from Spotify: {e}")
            return []
