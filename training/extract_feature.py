import os
import librosa

DATASET_PATH = "data/raw/Audio_Speech_Actors_01-24"

# First actor
actor = "Actor_01"

actor_path = os.path.join(DATASET_PATH, actor)

# First audio file
audio_file = os.listdir(actor_path)[0]

audio_path = os.path.join(actor_path, audio_file)

print("Audio File:", audio_file)

# Load audio
signal, sample_rate = librosa.load(audio_path, sr=None)

print("Sample Rate:", sample_rate)
print("Total Samples:", len(signal))

print("\nFirst 10 Samples:")
print(signal[:10])