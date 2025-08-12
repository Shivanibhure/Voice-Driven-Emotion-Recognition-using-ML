# predict_emotion.py

import librosa
import numpy as np
import joblib
import sys

# Load trained SVM model
model = joblib.load("svm_emotion_model.pkl")

# Function to extract features
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# Take input audio file from command line
if len(sys.argv) != 2:
    print("Usage: python predict_emotion.py path_to_audio.wav")
    sys.exit()

file_path = sys.argv[1]
features = extract_features(file_path).reshape(1, -1)
predicted_emotion = model.predict(features)[0]

print(f"🎙️ Emotion detected in the voice: **{predicted_emotion.upper()}**")
