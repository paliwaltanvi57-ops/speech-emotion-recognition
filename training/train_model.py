import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt

import tensorflow as tf

from keras.models import Sequential

from keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dense,
    Dropout
)

from keras.callbacks import EarlyStopping, ReduceLROnPlateau


# Load processed dataset
features = np.load("data/processed/features.npy")
labels = np.load("data/processed/labels.npy")
print(labels[:10])
print(labels.dtype)

print("Features Shape:", features.shape)
print("Labels Shape:", labels.shape)


# ---------------------------------------------------------------
# Actor-based split
# ---------------------------------------------------------------
# RAVDESS filenames encode the actor as the LAST number, e.g.
#   03-01-06-01-02-01-13.wav  ->  actor 13
# train_test_split on random rows lets the same actor appear in both
# train and test, so the model can learn "whose voice is this" instead
# of "what emotion is this". We split by actor instead.
#
# This requires actor_ids.npy, saved by your extraction script. If you
# haven't added that yet, see the note at the bottom of this file for
# the one-line change needed in your feature-extraction script.

actor_ids_path = "data/processed/actor_ids.npy"

if os.path.exists(actor_ids_path):
    actor_ids = np.load(actor_ids_path)
    print("\nUsing actor-based split (no voice shared between train/test).")

    unique_actors = sorted(set(actor_ids.tolist()))
    n_actors = len(unique_actors)

    # ~80% of actors for train+val, ~20% for test, held out completely
    n_test_actors = max(1, round(n_actors * 0.2))
    test_actors = set(unique_actors[-n_test_actors:])
    trainval_actors = [a for a in unique_actors if a not in test_actors]

    # carve a validation slice out of the train actors (not the test actors)
    n_val_actors = max(1, round(len(trainval_actors) * 0.2))
    val_actors = set(trainval_actors[-n_val_actors:])
    train_actors = set(trainval_actors[:-n_val_actors])

    train_idx = [i for i, a in enumerate(actor_ids) if a in train_actors]
    val_idx = [i for i, a in enumerate(actor_ids) if a in val_actors]
    test_idx = [i for i, a in enumerate(actor_ids) if a in test_actors]

    print(f"Train actors: {sorted(train_actors)}")
    print(f"Val actors:   {sorted(val_actors)}")
    print(f"Test actors:  {sorted(test_actors)}")

else:
    print(
        "\nWARNING: data/processed/actor_ids.npy not found. "
        "Falling back to a random split. This will overstate accuracy "
        "because the same voice can appear in both train and test. "
        "See the note at the bottom of this file to fix your extraction "
        "script, then re-run extraction before trusting these numbers."
    )
    all_idx = np.arange(len(labels))
    trainval_idx, test_idx = train_test_split(
        all_idx, test_size=0.2, random_state=42, stratify=labels
    )
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=0.2, random_state=42,
        stratify=labels[trainval_idx]
    )
    train_idx, val_idx, test_idx = train_idx.tolist(), val_idx.tolist(), test_idx.tolist()

X_train_raw, y_train_raw = features[train_idx], labels[train_idx]
X_val_raw, y_val_raw = features[val_idx], labels[val_idx]
X_test_raw, y_test_raw = features[test_idx], labels[test_idx]

print("\nDataset Split:")
print("Training:", X_train_raw.shape)
print("Validation:", X_val_raw.shape)
print("Testing:", X_test_raw.shape)


# ---------------------------------------------------------------
# Encode labels (fit ONLY on training data's classes, but labels are
# shared across the whole set so this is fine to fit on all of them)
# ---------------------------------------------------------------
label_encoder = LabelEncoder()
label_encoder.fit(labels)

y_train = label_encoder.transform(y_train_raw)
y_val = label_encoder.transform(y_val_raw)
y_test = label_encoder.transform(y_test_raw)

print("\nEmotion Classes:")
print(label_encoder.classes_)


# ---------------------------------------------------------------
# Normalize features using ONLY training-set statistics
# ---------------------------------------------------------------
# Your features currently range roughly -960 to +242, which makes
# gradient descent much harder than it needs to be. Compute mean/std
# on the training set only (never on val/test, that would leak info),
# then apply the same transform everywhere, including at prediction time.

# One mean/std per MFCC coefficient (40 of them), averaged over all clips
# AND all time frames. Shape (1, 40, 1). Do NOT compute it per time
# position: late frames are mostly zero padding, so their std is tiny and
# the few real values there would be blown up to huge numbers.
feat_mean = X_train_raw.mean(axis=(0, 2), keepdims=True)
feat_std = X_train_raw.std(axis=(0, 2), keepdims=True) + 1e-8

X_train = (X_train_raw - feat_mean) / feat_std
X_val = (X_val_raw - feat_mean) / feat_std
X_test = (X_test_raw - feat_mean) / feat_std

os.makedirs("models", exist_ok=True)
np.save("models/feature_mean.npy", feat_mean)
np.save("models/feature_std.npy", feat_std)
print("\nSaved feature_mean.npy / feature_std.npy — predict_emotion.py "
      "MUST apply this same normalization before predicting, or "
      "predictions will be wrong.")


# ---------------------------------------------------------------
# Augmentation (TRAINING data only)
# ---------------------------------------------------------------
# With only 15 training voices the model memorises them: training accuracy
# climbed to 60% while validation collapsed. We make "new" training
# examples by randomly blanking a band of MFCC coefficients, blanking a
# stretch of time, and adding a little noise (SpecAugment-style). Val and
# test sets are never augmented.
rng = np.random.default_rng(42)

def augment(X):
    X = X.copy()
    for i in range(len(X)):
        f = rng.integers(0, 7)                  # blank up to 6 coefficients
        f0 = rng.integers(0, 40 - f + 1)
        X[i, f0:f0 + f, :] = 0.0
        t = rng.integers(0, 31)                 # blank up to 30 frames
        t0 = rng.integers(0, 300 - t + 1)
        X[i, :, t0:t0 + t] = 0.0
    X += rng.normal(0, 0.05, X.shape)           # small noise
    return X

X_train = np.concatenate([X_train, augment(X_train), augment(X_train)])
y_train = np.concatenate([y_train, y_train, y_train])
print("Training set after augmentation:", X_train.shape)


# Add channel dimension for CNN
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print("\nAfter Adding Channel Dimension")
print("Training Shape:", X_train.shape)


# ---------------------------------------------------------------
# Model — GlobalAveragePooling2D instead of Flatten
# ---------------------------------------------------------------
# Flatten() into Dense(128) was creating ~4.8M parameters for ~1,100
# training examples, which mostly memorizes rather than generalizes.
# GlobalAveragePooling2D collapses each feature map to one number,
# cutting parameters drastically while keeping the useful signal.

model = Sequential([

    Input(shape=(40, 300, 1)),

    # Block 1
    Conv2D(32, (3, 3), padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Block 2
    Conv2D(64, (3, 3), padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Block 3 (new: the first version was too small to learn much)
    Conv2D(128, (3, 3), padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    GlobalAveragePooling2D(),

    Dense(128, activation="relu"),
    Dropout(0.5),

    Dense(8, activation="softmax")

])
model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# validation_data is now the held-out VAL set, not the test set —
# the test set stays untouched until the final evaluate() call below.
# Up to 80 epochs, but stop automatically when validation loss stops
# improving, and keep the BEST version of the model (not the last one).
early_stop = EarlyStopping(
    monitor="val_loss", patience=12, restore_best_weights=True
)
# Lower the learning rate when progress stalls
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=80,
    batch_size=32,
    callbacks=[early_stop, reduce_lr]
)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("\nTest Accuracy:", test_accuracy)
print("Test Loss:", test_loss)


# ---------------------------------------------------------------
# Per-class results — always look at this, not just overall accuracy
# ---------------------------------------------------------------
y_pred = model.predict(X_test).argmax(axis=1)

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    labels=range(len(label_encoder.classes_)),
    target_names=label_encoder.classes_,
    zero_division=0
))

print("Confusion Matrix (rows = true, cols = predicted):")
print(label_encoder.classes_)
print(confusion_matrix(y_test, y_pred, labels=range(len(label_encoder.classes_))))


# Save trained model
model.save("models/emotion_model.keras")
print("\nModel saved successfully!")

joblib.dump(label_encoder, "models/label_encoder.pkl")
print("Label encoder saved successfully!")

# Accuracy Graph
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("models/accuracy_graph.png")
plt.show()

# Loss Graph
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("models/loss_graph.png")
plt.show()


# ---------------------------------------------------------------
# TO ENABLE THE ACTOR-BASED SPLIT:
# In your feature-extraction script, wherever you currently build the
# `labels` list from each filename, also build an `actor_ids` list:
#
#   actor_id = int(filename.split('-')[-1].split('.')[0])
#   actor_ids.append(actor_id)
#
# ...then save it alongside features/labels:
#
#   np.save("data/processed/actor_ids.npy", np.array(actor_ids))
#
# Re-run extraction once, then re-run this script. You'll see the
# WARNING disappear and the real (lower, honest) accuracy show up.
# ---------------------------------------------------------------
