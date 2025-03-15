import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import tensorflow_datasets as tfds

# Load EMNIST Balanced dataset
ds_train = tfds.load('emnist/balanced', split='train', as_supervised=True)
ds_test = tfds.load('emnist/balanced', split='test', as_supervised=True)

# Convert dataset to NumPy arrays
def preprocess(dataset):
    X, y = [], []
    for img, label in tfds.as_numpy(dataset):
        img = np.array(img, dtype=np.float32) / 255.0  # Normalize
        X.append(img)
        y.append(label)
    return np.array(X), np.array(y)

X_train, y_train = preprocess(ds_train)
X_test, y_test = preprocess(ds_test)

# Reshape data for CNN
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Build CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(47, activation='softmax')  # 47 classes (digits + letters)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Save Model in Keras format
model.save("handwritten_character_model.keras")
print("Model saved successfully as handwritten_character_model.keras")
