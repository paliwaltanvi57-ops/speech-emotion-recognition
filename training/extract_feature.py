import os
import sys
import numpy as np
import librosa

# Make "utils" importable when this script is run as
#   python training/extract_feature.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.audio_features import extract_mfcc

DATASET_PATH = "data/raw/Audio_Speech_Actors_01-24"

emotion_map = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fear",
    "07": "Disgust",
    "08": "Surprised"
}

features = []
labels = []
actor_ids = []   # NEW: which actor spoke each clip

actors = sorted(os.listdir(DATASET_PATH))

for actor in actors:

    actor_path = os.path.join(DATASET_PATH, actor)

    # skip stray files that aren't actor folders
    if not os.path.isdir(actor_path):
        continue

    files = sorted(os.listdir(actor_path))

    for file in files:

        if not file.lower().endswith(".wav"):
            continue

        file_path = os.path.join(actor_path, file)

        # Filename looks like 03-01-06-01-02-01-13.wav
        parts = file.replace(".wav", "").split("-")
        emotion = emotion_map[parts[2]]
        actor_id = int(parts[-1])          # last number = actor

        # Load audio
        signal, sample_rate = librosa.load(file_path, sr=None)

        # Trim silence + MFCC + pad to 300 frames (shared with prediction)
        mfcc = extract_mfcc(signal, sample_rate)

        features.append(mfcc)
        labels.append(emotion)
        actor_ids.append(actor_id)

print("\nDataset Ready!")
print("Total Features:", len(features))
print("Total Labels:", len(labels))
print("Shape of First Feature:", features[0].shape)
print("First Label:", labels[0])

# Convert lists to NumPy arrays
features = np.array(features)
labels = np.array(labels)
actor_ids = np.array(actor_ids)

print("\nConverted to NumPy arrays")
print("Features Shape:", features.shape)
print("Labels Shape:", labels.shape)
print("Actors found:", sorted(set(actor_ids.tolist())))

# Save processed data (create the folder if it doesn't exist)
os.makedirs("data/processed", exist_ok=True)
np.save("data/processed/features.npy", features)
np.save("data/processed/labels.npy", labels)
np.save("data/processed/actor_ids.npy", actor_ids)

print("\nProcessed dataset saved successfully!")
