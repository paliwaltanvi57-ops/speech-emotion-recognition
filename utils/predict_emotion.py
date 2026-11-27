import os
import numpy as np
import librosa
import tensorflow as tf
import joblib

from utils.audio_features import extract_mfcc

# Build paths from this file's location, so the app works no matter which
# folder you start it from.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# RAVDESS audio is 48 kHz and training loaded it at its native rate.
# Uploaded audio with a different rate is resampled to match, so the
# MFCC frames mean the same thing as they did during training.
TRAIN_SAMPLE_RATE = 48000

model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "emotion_model.keras"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

# Normalisation statistics saved by train_model.py. Without these the
# model receives numbers on a different scale from training and its
# answers are meaningless.
feature_mean = np.load(os.path.join(MODEL_DIR, "feature_mean.npy"))
feature_std = np.load(os.path.join(MODEL_DIR, "feature_std.npy"))

print("Model Loaded Successfully!")
print("Label Encoder Loaded Successfully!")


def predict_emotion(audio_path):

    # Load audio at the same sample rate as training
    signal, sample_rate = librosa.load(audio_path, sr=TRAIN_SAMPLE_RATE)

    # Same steps as training: trim silence, MFCC, per-clip mean removal,
    # pad/cut to 300 frames. One shared function guarantees it matches.
    mfcc = extract_mfcc(signal, sample_rate)

    # Add batch dimension -> (1, 40, 300), then normalise exactly like training
    x = mfcc[np.newaxis, :, :]
    x = (x - feature_mean) / feature_std

    # Add channel dimension -> (1, 40, 300, 1)
    x = x[..., np.newaxis]

    # Predict
    prediction = model.predict(x, verbose=0)

    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    # Convert number back to emotion name
    emotion = str(label_encoder.inverse_transform([predicted_index])[0])

    return emotion, confidence


