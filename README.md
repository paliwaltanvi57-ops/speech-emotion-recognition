# Speech Emotion Recognition

A small deep-learning project that detects emotion in speech from audio. It extracts MFCC features from `.wav` clips, trains a CNN (TensorFlow/Keras) to classify 8 emotions, and includes a Flask web app for uploading audio and viewing the results.

**Emotions:** Neutral, Calm, Happy, Sad, Angry, Fear, Disgust, Surprised

## Results

Trained and tested on the [RAVDESS](https://zenodo.org/record/1188976) speech dataset (24 actors, 1,440 clips).

The data is split **by actor**, so the model is always tested on voices it has never heard:

| Set | Actors |
|---|---|
| Train | 1–15 |
| Validation | 16–19 |
| Test | 20–24 |

| Metric | Value |
|---|---|
| Test accuracy (unseen actors) | **~53%** |
| Random guessing (8 classes) | 12.5% |

**Why split by actor?** An earlier version split clips randomly, so the same actor appeared in both training and test sets. That reported ~43%, but the model was partly recognising *voices*, not emotions (data leakage). Splitting by actor dropped the score to ~25%, and the improvements below raised it to ~53% on genuinely unseen speakers.

Per-class results on the test set show some emotions are much harder than others (for example Surprised and Angry are detected well, while Happy and Neutral are weak). Calm, Neutral and Sad are often confused with each other. The test set has only 5 actors, so treat the exact number as approximate.

## What improved the model

- Split train/validation/test by actor (no shared voices)
- Trimmed silence from each clip
- Removed each clip's own mean from the MFCCs (reduces speaker-specific bias)
- Normalised features using training-set statistics only
- Replaced `Flatten` with `GlobalAveragePooling2D` (far fewer parameters)
- Added BatchNormalization and Dropout
- Added data augmentation (frequency/time masking and noise) on training data only
- Early stopping and learning-rate reduction, using a separate validation set

## Project structure

```
speech-emotion-recognition/
├── app.py                  # Flask web app
├── requirements.txt
├── training/
│   ├── extract_feature.py  # builds features, labels and actor IDs from the audio
│   └── train_model.py      # trains and evaluates the model
├── utils/
│   ├── audio_features.py   # shared MFCC extraction (used for training and prediction)
│   ├── chunk_audio.py      # splits uploaded audio into chunks
│   └── predict_emotion.py  # runs the trained model on a chunk
├── templates/              # HTML pages
├── static/                 # CSS and assets
├── data/                   # dataset (not included in the repo)
├── models/                 # saved model, label encoder, normalisation stats
├── uploads/  chunks/  results/   # created automatically at runtime
```

## Setup

Requires Python 3.10 or 3.11 and [ffmpeg](https://ffmpeg.org/download.html).

```bash
git clone https://github.com/paliwaltanvi57-ops/speech-emotion-recognition.git
cd speech-emotion-recognition

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

## Train the model

1. Download the RAVDESS speech audio (`Audio_Speech_Actors_01-24`) and place it at:
   ```
   data/raw/Audio_Speech_Actors_01-24/
   ```
2. Run both scripts from the project root:
   ```bash
   python training/extract_feature.py
   python training/train_model.py
   ```
   Training prints a classification report and confusion matrix, and saves the model to `models/`.

## Run the web app

```bash
python app.py
```

Open http://127.0.0.1:5000, upload a `.wav` file, and the app splits it into chunks, predicts an emotion for each, and shows an overall result. A JSON report can be downloaded from the results page.

## Known limitations / to do

- The web app must use the same feature steps as training (`utils/audio_features.py`, plus the saved `feature_mean.npy` / `feature_std.npy`).
- ffmpeg paths should be read from the system PATH rather than hardcoded.
- Accuracy is measured on only 5 test actors; cross-validation by actor would give a more reliable estimate.
- RAVDESS contains acted emotions from North American actors, so the model may not generalise to natural speech, other accents or noisy recordings.
- The model should not be used for any real decision-making about people.

## Tech

Python, TensorFlow/Keras, librosa, scikit-learn, Flask, pydub