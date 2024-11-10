import pickle
import io
import time
from pydantic import BaseModel
import json
import numpy as np
import soundfile as sf
import librosa
import speech_recognition as sr
from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

# Initialize the FastAPI app
app = FastAPI()
API_KEY = "ezeebot-2024"
API_KEY_NAME = "ezeebot"

# Allow CORS for the Flutter application
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Flutter app's origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket for real-time voice analysis
@app.websocket("/analyze/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            # Receive audio chunks
            audio_chunk = await websocket.receive_bytes()

            # Process the audio chunk (analyze pitch, energy, etc.)
            voice_features = analyze_voice_features(audio_chunk)

            # Send results back to the client
            await websocket.send_json({"voice_features": voice_features})
        
        except Exception as e:
            print(f"Error: {e}")
            await websocket.close()
            break

# Load pre-trained emotion analysis model
emotion_analyzer = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# Placeholder for trained RandomForest model
model = RandomForestClassifier()

# Additional emotions to be recognized
additional_emotions = [
    "Depression", "Anxiety", "Feeling hopeless", "Detachment from reality",
    "Anger", "Sadness", "Amusement", "Fear", "Disgust", "Happiness",
    "Excitement", "Love", "Doubt", "Hesitation"
]

# Function to analyze pitch, energy, pauses from audio
def analyze_voice_features(audio_data):
    try:
        audio, sample_rate = sf.read(io.BytesIO(audio_data))

        # Pitch analysis
        pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sample_rate)
        pitch_values = [pitches[magnitudes[:, i].argmax(), i] for i in range(pitches.shape[1]) if pitches[magnitudes[:, i].argmax(), i] > 0]

        # Energy analysis
        energy = np.mean(librosa.feature.rms(y=audio))

        # Speech rate calculation (syllables/second)
        duration = librosa.get_duration(y=audio, sr=sample_rate)
        speech_rate = len(pitch_values) / duration if duration > 0 else 0

        # Calculate pauses (silences)
        pauses = librosa.effects.split(audio, top_db=30)
        total_pause_time = sum([(pause[1] - pause[0]) / sample_rate for pause in pauses])

        # Voice modulation: standard deviation of pitch values
        pitch_variability = np.std(pitch_values) if pitch_values else 0

        return {
            "mean_pitch": np.mean(pitch_values),
            "energy": energy,
            "speech_rate": speech_rate,
            "total_pause_time": total_pause_time,
            "pitch_variability": pitch_variability
        }

    except Exception as e:
        print(f"Error in voice feature analysis: {e}")
        return None

# Function to analyze emotions from recognized text
def analyze_emotion(text):
    if text:
        emotions = emotion_analyzer(text)
        detected_emotions = {emotion['label']: emotion['score'] for emotion in emotions}

        # Add additional emotions if not present
        for additional in additional_emotions:
            if additional not in detected_emotions:
                detected_emotions[additional] = 0.0

        return detected_emotions
    return None

# Function to aggregate voice and emotion features for prediction
def aggregate_scores(voice_features, emotions):
    if voice_features:
        scores = {
            "mean_pitch": voice_features['mean_pitch'],
            "energy": voice_features['energy'],
            "speech_rate": voice_features['speech_rate'],
            "total_pause_time": voice_features['total_pause_time'],
            "pitch_variability": voice_features['pitch_variability'],
        }

        # Combine voice and emotion scores
        combined_scores = {**scores, **emotions}

        # Normalize the scores
        scaler = MinMaxScaler()
        score_values = np.array(list(combined_scores.values())).reshape(-1, 1)
        normalized_scores = scaler.fit_transform(score_values)

        return dict(zip(combined_scores.keys(), normalized_scores.flatten()))
    return None

# Function to predict mental health status
def predict_mental_health(combined_scores):
    if combined_scores:
        # Predict mental health state using trained model
        prediction = model.predict(np.array(list(combined_scores.values())).reshape(1, -1))
        return prediction[0]
    return None

# REST API for analyzing audio file
@app.post("/analyze/")
async def analyze_audio(file: UploadFile = File(...)):
    audio_data = await file.read()

    # Step 1: Analyze voice features
    voice_features = analyze_voice_features(audio_data)

    # Step 2: Analyze emotion (optional, add text recognition if required)
    text = ""  # Placeholder for text recognition logic
    emotions = analyze_emotion(text)

    # Step 3: Aggregate features for prediction
    combined_scores = aggregate_scores(voice_features, emotions)

    # Step 4: Predict mental health status
    prediction = predict_mental_health(combined_scores)

    # Return the prediction and analyzed data
    return {
        "voice_features": voice_features,
        "emotions": emotions,
        "prediction": prediction
    }
@app.post("/analyzeTest/")
async def analyze_audio():
    
    file_path = "D:\Python_projects\Pitchanalysis\audio_records\audio_20241007_193219.wav"  
    
    with open(file_path, "rb") as file:
        audio_data = file.read()

    # Step 1: Analyze voice features
    voice_features = analyze_voice_features(audio_data)

    # Step 2: Analyze emotion (optional, add text recognition if required)
    text = ""  # Placeholder for text recognition logic
    emotions = analyze_emotion(text)

    # Step 3: Aggregate features for prediction
    combined_scores = aggregate_scores(voice_features, emotions)

    # Step 4: Predict mental health status
    prediction = predict_mental_health(combined_scores)

    # Return the prediction and analyzed data
    return {
        "voice_features": voice_features,
        "emotions": emotions,
        "prediction": prediction
    }
# Health check endpoint (Optional)
@app.get("/health")
async def health_check():
    return {"status": "API is running"}

# Run the app using Uvicorn (Optional for standalone execution)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
