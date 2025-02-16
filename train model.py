import numpy as np
import tensorflow as tf
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define dataset paths
train_path = "D:/python/Capstonep/train"
test_path = "D:/python/Capstonep/test"

# Get updated emotion categories
emotions = sorted([folder for folder in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, folder))])
print("Updated Emotion Categories:", emotions)

# Map emotions to numerical labels
emotion_to_label = {emotion: idx for idx, emotion in enumerate(emotions)}

# Load OpenCV Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def apply_clahe(image):
    """Apply CLAHE to enhance contrast."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def detect_face(img):
    """Detect faces and return cropped region. If no face is found, return the original image."""
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return img[y:y+h, x:x+w]
    return img

def load_images_from_directory(directory):
    """Load and preprocess images."""
    X, y = [], []
    for emotion in emotions:
        emotion_path = os.path.join(directory, emotion)
        for img_name in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img_name)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    face = detect_face(img)
                    face = cv2.resize(face, (48, 48))
                    face = apply_clahe(face)
                    X.append(face)
                    y.append(emotion_to_label[emotion])
            except Exception as e:
                print(f"Error loading image {img_name}: {e}")
    if len(X) == 0 or len(y) == 0:
        raise ValueError("No images were loaded. Check dataset path and file integrity.")
    return np.array(X), np.array(y)

# Load dataset
X_train, y_train = load_images_from_directory(train_path)
X_test, y_test = load_images_from_directory(test_path)

# Apply Mean-Std Normalization
X_mean, X_std = np.mean(X_train), np.std(X_train)
X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

# Reshape for CNN input
X_train = X_train.reshape(-1, 48, 48, 1)
X_test = X_test.reshape(-1, 48, 48, 1)

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=len(emotions))
y_test = to_categorical(y_test, num_classes=len(emotions))

# Split validation set (Handle small datasets)
if len(X_train) > 1 and len(y_train) > 1:
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
else:
    X_val, y_val = X_test, y_test  # If dataset is too small, use test set as validation

# **Reduced Data Augmentation (More Natural Enhancements)**
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True
)

train_generator = datagen.flow(X_train, y_train, batch_size=32, shuffle=True)

# Improved CNN Model
model = Sequential([
    Input(shape=(48, 48, 1)),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(len(emotions), activation='softmax')
])

# Compile Model with Lower Learning Rate and Gradient Clipping
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
model_checkpoint = ModelCheckpoint("emotion_model_v3.keras", save_best_only=True, monitor='val_accuracy')

# Train Model with 30 Epochs
history = model.fit(train_generator,
                    epochs=30,
                    steps_per_epoch=len(train_generator),
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping, model_checkpoint])

# Save Model
model.save("emotion_model_v3.keras")

# Plot Training History
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()
