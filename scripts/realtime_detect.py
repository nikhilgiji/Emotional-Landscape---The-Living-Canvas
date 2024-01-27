import cv2
import dlib
import numpy as np 
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load pre-trained models
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
emotion_model = load_model("/Users/nikhilgiji/Drive D/cs/deep_learning/hci_artistic/model.h5")  # You need to have a pre-trained model for emotion recognition

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Open a connection to the camera (0 represents the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detector(gray)

    for face in faces:
        # Get facial landmarks
        landmarks = landmark_predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Extract the region of interest (ROI) for facial expression classification
        (x, y, w, h) = cv2.boundingRect(landmarks)
        roi = gray[y:y+h, x:x+w]

        # Resize ROI to match the input size of the emotion model
        roi = cv2.resize(roi, (48, 48))
        roi = np.expand_dims(np.expand_dims(roi, -1), 0)

        # Predict facial expression using the emotion model
        emotion_prediction = emotion_model.predict(roi)
        emotion_index = np.argmax(emotion_prediction)

        # Draw bounding box and emotion label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_labels[emotion_index], (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Facial Expression Recognition', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
