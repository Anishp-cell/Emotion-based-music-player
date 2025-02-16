import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("emotion_model_v3.keras")

# Updated emotion labels (based on the modified dataset)
emotion_labels = ["Angry", "Happy", "Neutral", "Sad", "Surprised"]

# Load OpenCV Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def apply_clahe(image):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def preprocess_image(image):
    """Preprocess image: face detection, resizing, CLAHE, and normalization."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None, None  # No face detected

    x, y, w, h = faces[0]  # Use the first detected face
    face = gray[y:y+h, x:x+w]  # Crop face
    face = cv2.resize(face, (48, 48))  # Resize to model input size
    face = apply_clahe(face)  # Apply CLAHE
    face = (face - np.mean(face)) / np.std(face)  # Normalize (Mean-Std)
    face = np.expand_dims(face, axis=0).reshape(-1, 48, 48, 1)  # Reshape for model

    return face, (x, y, w, h)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_face, face_coords = preprocess_image(frame)

    if processed_face is not None:
        # Predict emotion
        predictions = model.predict(processed_face)
        emotion = emotion_labels[np.argmax(predictions)]

        # Draw rectangle and label
        x, y, w, h = face_coords
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show output
    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
