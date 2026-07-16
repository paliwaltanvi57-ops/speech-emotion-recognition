import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt


import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)


# Load processed dataset
features = np.load("data/processed/features.npy")
labels = np.load("data/processed/labels.npy")
print(labels[:10])
print(labels.dtype)


print("Features Shape:", features.shape)
print("Labels Shape:", labels.shape)



# Encode emotion labels
label_encoder = LabelEncoder()

labels_encoded = label_encoder.fit_transform(labels)

print("\nEncoded Labels Shape:", labels_encoded.shape)

print("First 10 Encoded Labels:")
print(labels_encoded[:10])

print("\nEmotion Classes:")
print(label_encoder.classes_)


X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels_encoded,
    test_size=0.2,
    random_state=42,
    stratify=labels_encoded
)

print("\nDataset Split Successfully!")
print("Training Features:", X_train.shape)
print("Testing Features:", X_test.shape)

print("Training Labels:", y_train.shape)
print("Testing Labels:", y_test.shape)


# Add channel dimension for CNN
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print("\nAfter Adding Channel Dimension")
print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)



# model building
model = Sequential([
    
    Conv2D(
        filters=32,
        kernel_size=(3,3),
        activation="relu",
        input_shape=(40,300,1)
    ),

    MaxPooling2D(pool_size=(2,2)),

    Conv2D(
        filters=64,
        kernel_size=(3,3),
        activation="relu"
    ),

    MaxPooling2D(pool_size=(2,2)),

    Flatten(),

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

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    batch_size=32
)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("\nTest Accuracy:", test_accuracy)
print("Test Loss:", test_loss)


# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save trained model
model.save("models/emotion_model.keras")

print("\nModel saved successfully!")



joblib.dump(label_encoder, "models/label_encoder.pkl")

print("Label encoder saved successfully!")

# Accuracy Graph
plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("models/accuracy_graph.png")
plt.show()


# Loss Graph
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("models/loss_graph.png")
plt.show()