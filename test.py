import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('emotion_recognition_model_vgg16.h5') #add the file name in single quotes which is created after training the model

# Emotion labels
emotion_labels = ['happy', 'sad', 'angry', 'surprised', 'fatigue', 'neutral']

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect face using OpenCV's Haar cascades or any face detection method
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]  # Crop the face from the frame
        face = cv2.resize(face, (224, 224))  # Resize to match model input size
        face = face / 255.0  # Normalize pixel values to [0, 1]
        face = np.expand_dims(face, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model.predict(face)
        emotion = emotion_labels[np.argmax(predictions)]

        # Display emotion on the video
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
