import React, { useState, useEffect, useCallback, useRef } from 'react';
import type { EmotionData, Track, Emotion, LanguagesResponse } from './types';
import { API_BASE_URL, EMOTION_COLORS, EMOTION_BG_GRADIENTS } from './constants';
import Hero from './components/Hero';
import WebcamCard from './components/WebcamCard';
import EmotionDisplayCard from './components/EmotionDisplayCard';
import ActionButton from './components/ActionButton';
import TrackGrid from './components/TrackGrid';
import ParticleBackground from './components/ParticleBackground';
import { motion, AnimatePresence } from 'framer-motion';
import LanguageSelector from './components/LanguageSelector';
import { AudioPlayerProvider } from './contexts/AudioPlayerContext';

const normalizeEmotion = (raw: string | null | undefined): Emotion => {
  if (!raw || typeof raw !== "string") return "Neutral";

  const formatted =
    raw.charAt(0).toUpperCase() + raw.slice(1).toLowerCase();

  return formatted as Emotion;
};

const App: React.FC = () => {
  const [emotionData, setEmotionData] = useState<EmotionData>({
    emotion: 'Neutral',   // 🔥 SAFE DEFAULT
    confidence: 0,
    is_stable: false
  });

  const [tracks, setTracks] = useState<Track[] | null>(null);
  const [tracksEmotion, setTracksEmotion] = useState<Emotion>('Neutral');
  const [isLoadingTracks, setIsLoadingTracks] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [languages, setLanguages] = useState<string[]>(['Mixed']);
  const [selectedLanguage, setSelectedLanguage] = useState<string>('Mixed');
  const [isLoadingLanguages, setIsLoadingLanguages] = useState(true);
  
  const [sessionId, setSessionId] = useState<string | null>(null);
  const isProcessingFrame = useRef(false);

  // Initialize session & languages
  useEffect(() => {
    const initSession = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/session`, { method: 'POST' });
        if (response.ok) {
          const data = await response.json();
          setSessionId(data.session_id);
        }
      } catch (err) {
        console.error("Failed to initialize session", err);
      }
    };
    
    const fetchLanguages = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/languages`);
        if (response.ok) {
          const data: LanguagesResponse = await response.json();
          const defaultLang = data.default || 'Mixed';
          const otherLangs = data.languages.filter(lang => lang !== defaultLang);
          setLanguages([defaultLang, ...otherLangs]);
          setSelectedLanguage(defaultLang);
        }
      } catch {
        setLanguages(['Mixed', 'English', 'Hindi', 'Spanish']);
      } finally {
        setIsLoadingLanguages(false);
      }
    };

    initSession();
    fetchLanguages();
  }, []);

  const handleFrameCapture = useCallback(async (base64Image: string) => {
    if (isProcessingFrame.current) return;
    
    isProcessingFrame.current = true;

    try {
      const payload = {
        image_base64: base64Image,
        session_id: sessionId
      };

      const response = await fetch(`${API_BASE_URL}/emotion`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      
      if (response.ok) {
        const data = await response.json();

        setEmotionData({
          emotion: normalizeEmotion(data.emotion),
          confidence: data.confidence ?? 0,
          is_stable: data.is_stable ?? false
        });
      }

    } catch (err) {
      console.error("Error sending frame:", err);
    } finally {
      isProcessingFrame.current = false;
    }
  }, [sessionId]);

  const handleDiscoverMusic = useCallback(async (langOverride?: string) => {
    if (isLoadingTracks) return;

    setIsLoadingTracks(true);
    setError(null);
    setTracks(null);
    setTracksEmotion(emotionData.emotion);

    const languageToUse = langOverride || selectedLanguage;

    try {
      const response = await fetch(`${API_BASE_URL}/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          emotion: emotionData.emotion, 
          language: languageToUse,
          session_id: sessionId
        }),
      });

      if (!response.ok) throw new Error('Could not find tracks for your mood.');

      const data = await response.json();
      setTracks(data.tracks);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred.');
      setTracks([]);
    } finally {
      setIsLoadingTracks(false);
    }

  }, [isLoadingTracks, emotionData.emotion, selectedLanguage, sessionId]);

  const handleLanguageChange = (newLanguage: string) => {
    setSelectedLanguage(newLanguage);
    if (tracks !== null && !isLoadingTracks) {
      handleDiscoverMusic(newLanguage);
    }
  };

  const safeEmotion = emotionData.emotion;
  const safeBackground = EMOTION_BG_GRADIENTS[safeEmotion] || '';

  return (
    <AudioPlayerProvider>
      <div className="min-h-screen bg-[#0a0118] text-white overflow-hidden relative" style={{ isolation: 'isolate' }}>
        <ParticleBackground />

        <AnimatePresence>
          <motion.div
            key={safeEmotion}
            className="absolute inset-0 z-[-1] opacity-15 pointer-events-none"
            style={{ backgroundImage: safeBackground }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.15 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 1.5 }}
          />
        </AnimatePresence>

        <main className="relative z-10 p-4 sm:p-6 md:p-8 max-w-7xl mx-auto flex flex-col gap-8">
          <Hero />

          <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
            <div className="lg:col-span-3">
              <WebcamCard 
                emotion={safeEmotion} 
                isStable={emotionData.is_stable}
                onFrameCapture={handleFrameCapture}
              />
            </div>

            <div className="lg:col-span-2 flex flex-col gap-8">
              <EmotionDisplayCard emotionData={emotionData} />

              <LanguageSelector
                languages={languages}
                selectedLanguage={selectedLanguage}
                onLanguageChange={handleLanguageChange}
                emotion={safeEmotion}
                isLoading={isLoadingLanguages}
              />

              <ActionButton
                onClick={() => handleDiscoverMusic()}
                isLoading={isLoadingTracks}
                emotion={safeEmotion}
                disabled={false}
              />
            </div>
          </div>

          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="text-center bg-red-500/20 border border-red-500 text-red-300 p-4 rounded-lg"
              >
                {error}
              </motion.div>
            )}
          </AnimatePresence>

          <TrackGrid
            tracks={tracks}
            isLoading={isLoadingTracks}
            emotion={tracksEmotion}
            selectedLanguage={selectedLanguage}
            sessionId={sessionId}
          />
        </main>
      </div>
    </AudioPlayerProvider>
  );
};

export default App;