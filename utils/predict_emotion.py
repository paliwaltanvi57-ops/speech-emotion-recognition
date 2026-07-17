import numpy as np
import librosa
import tensorflow as tf
import joblib

model = tf.keras.models.load_model("models/emotion_model.keras")
label_encoder = joblib.load("models/label_encoder.pkl")

print("Model Loaded Successfully!")
print("Label Encoder Loaded Successfully!")

def predict_emotion(audio_path):

    # Load audio
    signal, sample_rate = librosa.load(audio_path, sr=None)

    # Extract MFCC
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=40
    )

    # Make MFCC width = 300 (same as training)
    if mfcc.shape[1] > 300:
        mfcc = mfcc[:, :300]
    else:
        pad_width = 300 - mfcc.shape[1]
        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, pad_width)),
            mode="constant"
        )

    # Add channel dimension
    mfcc = mfcc[np.newaxis, ..., np.newaxis]

    # Predict
    prediction = model.predict(mfcc, verbose=0)

    # Highest probability
    predicted_index = np.argmax(prediction)

    confidence = float(np.max(prediction))

    # Convert number back to emotion name
    emotion = str(label_encoder.inverse_transform([predicted_index])[0])

    return emotion, confidence

