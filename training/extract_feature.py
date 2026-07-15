import os
import numpy as np
import librosa

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


# First actor
actors = sorted(os.listdir(DATASET_PATH))

for actor in actors:

    actor_path = os.path.join(DATASET_PATH, actor)

    files = sorted(os.listdir(actor_path))

    for file in files:
        file_path = os.path.join(actor_path, file)
        # Get emotion label
        emotion_code = file.split("-")[2]
        emotion = emotion_map[emotion_code]
        # Load audio
        signal, sample_rate = librosa.load(file_path, sr=None)
        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sample_rate,
            n_mfcc=40
            )
        # Make all MFCCs the same width (300 frames)
        if mfcc.shape[1] > 300:
            mfcc = mfcc[:, :300]
        else:
            pad_width = 300 - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
        # Store feature and label
        features.append(mfcc)
        labels.append(emotion)

       

print("\nDataset Ready!")
print("Total Features:", len(features))
print("Total Labels:", len(labels))
print("Shape of First Feature:", features[0].shape)
print("First Label:", labels[0])

# Convert lists to NumPy arrays
features = np.array(features)
labels = np.array(labels)

print("\nConverted to NumPy arrays")
print("Features Shape:", features.shape)
print("Labels Shape:", labels.shape)

# Save processed data
np.save("data/processed/features.npy", features)
np.save("data/processed/labels.npy", labels)

print("\nProcessed dataset saved successfully!")