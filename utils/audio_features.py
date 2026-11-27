import numpy as np
import librosa

N_MFCC = 40
MAX_FRAMES = 300


def extract_mfcc(signal, sample_rate):
    """Turn raw audio into a (40, 300) MFCC array.

    Used by BOTH training (extract_feature.py) and prediction
    (predict_emotion.py), so the two can never drift apart.
    """
    # Trim silence from the start and end of the clip
    trimmed, _ = librosa.effects.trim(signal, top_db=25)

    # If a clip was nearly all silence, keep the original instead
    if len(trimmed) < sample_rate * 0.2:
        trimmed = signal

    mfcc = librosa.feature.mfcc(y=trimmed, sr=sample_rate, n_mfcc=N_MFCC)

    # Keep at most 300 frames
    if mfcc.shape[1] > MAX_FRAMES:
        mfcc = mfcc[:, :MAX_FRAMES]

    # Per-clip mean removal (done on the REAL frames only, before padding).
    # Each speaker/microphone adds a roughly constant offset to every MFCC
    # coefficient (loudness, voice "colour"). Subtracting the clip's own
    # mean removes much of that, so the model is pushed to learn how the
    # voice CHANGES (emotion) instead of who is speaking.
    mfcc = mfcc - mfcc.mean(axis=1, keepdims=True)

    # Pad short clips with zeros (= the mean now) up to 300 frames
    if mfcc.shape[1] < MAX_FRAMES:
        pad_width = MAX_FRAMES - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")

    return mfcc
