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

app = FastAPI()
API_KEY = "ezeebot-2024"
API_KEY_NAME = "ezeebot"




app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/analyze/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
         
            audio_chunk = await websocket.receive_bytes()

            voice_features = analyze_voice_features(audio_chunk)

            
            await websocket.send_json({"voice_features": voice_features})
        
        except Exception as e:
            print(f"Error: {e}")
            await websocket.close()
            break


emotion_analyzer = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")


recognizer = sr.Recognizer()


model = RandomForestClassifier()  


additional_emotions = [
    "Depression", "Anxiety", "Feeling hopeless", "Detachment from reality",
    "Anger", "Sadness", "Amusement", "Fear", "Disgust", "Happiness",
    "Excitement", "Love", "Doubt", "Hesitation"
]

def analyze_voice_features(audio_data):
    try:
      
        audio, sample_rate = sf.read(io.BytesIO(audio_data))

        
        pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sample_rate)
        pitch_values = [pitches[magnitudes[:, i].argmax(), i] for i in range(pitches.shape[1]) if pitches[magnitudes[:, i].argmax(), i] > 0]

       
        energy = np.mean(librosa.feature.rms(y=audio))

        
        duration = librosa.get_duration(y=audio, sr=sample_rate)
        speech_rate = len(pitch_values) / duration if duration > 0 else 0

        
        pauses = librosa.effects.split(audio, top_db=30)
        total_pause_time = sum([(pause[1] - pause[0]) / sample_rate for pause in pauses])

        
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


def analyze_emotion(text):
    if text:
        emotions = emotion_analyzer(text)
        detected_emotions = {}
        
        for emotion in emotions:
            detected_emotions[emotion['label']] = emotion['score']

        
        for additional in additional_emotions:
            if additional not in detected_emotions:
                detected_emotions[additional] = 0.0  

        return detected_emotions
    return None


def aggregate_scores(voice_features, emotions):
    if voice_features:
        scores = {
            "mean_pitch": voice_features['mean_pitch'],
            "energy": voice_features['energy'],
            "speech_rate": voice_features['speech_rate'],
            "total_pause_time": voice_features['total_pause_time'],
            "pitch_variability": voice_features['pitch_variability'],
        }

        
        combined_scores = {**scores, **emotions}

        
        scaler = MinMaxScaler()
        score_values = np.array(list(combined_scores.values())).reshape(-1, 1)
        normalized_scores = scaler.fit_transform(score_values)

        return dict(zip(combined_scores.keys(), normalized_scores.flatten()))
    return None


def predict_mental_health(combined_scores):
    if combined_scores:
        
        prediction = model.predict(np.array(list(combined_scores.values())).reshape(1, -1))
        return prediction[0]  
    else:
        return None


@app.post("/analyze/")
async def analyze_audio(file: UploadFile = File(...)):
    audio_data = await file.read()  

    
    voice_features = analyze_voice_features(audio_data)

   
    text = ""  

    emotions = analyze_emotion(text)

    
    combined_scores = aggregate_scores(voice_features, emotions)

    prediction = predict_mental_health(combined_scores)

 
    return {
        "voice_features": voice_features,
        "emotions": emotions,
        "prediction": prediction
    }

if __name__ == "__main__":
    import uvicorn # type: ignore
    uvicorn.run(app, host="0.0.0.0", port=8000)
