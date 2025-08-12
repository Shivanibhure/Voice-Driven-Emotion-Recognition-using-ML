import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import kagglehub 
import joblib

# Path to your dataset folder
DATA_PATH = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")




# Emotion labels mapping based on RAVDESS naming convention
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Extract MFCC features from audio
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)  # Shape: (40,)
    return mfcc_mean

# Load dataset
features = []
labels = []

for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            try:
                # Example filename: 03-01-05-01-01-01-01.wav
                emotion_code = file.split("-")[2]
                emotion = emotion_map.get(emotion_code)
                if emotion:
                    path = os.path.join(root, file)
                    mfcc = extract_features(path)
                    features.append(mfcc)
                    labels.append(emotion)
            except Exception as e:
                print("Error processing file:", file, str(e))

# Convert to DataFrame
X = np.array(features)
y = np.array(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Train SVM model
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print (y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, "svm_emotion_model.pkl")
print("Model saved as svm_emotion_model.pkl")