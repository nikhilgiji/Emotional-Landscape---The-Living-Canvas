import pygame
import cv2
import numpy as np
from keras.models import load_model

# Load pre-trained emotion detection model
emotion_model = load_model('model.h5')  # Replace 'your_emotion_model.h5' with your model file path

# Initialize Pygame
pygame.init()

# Set up the display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Emotion-based Visualization')

# Define colors
BLACK = (0, 0, 0)

# Function to preprocess and detect emotions from image frames
def detect_emotion(frame):
    # Resize the frame to match the expected input shape (48x48)
    resized_frame = cv2.resize(frame, (48, 48))

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # Normalize the pixel values to be in the range [0, 1]
    normalized_frame = gray_frame / 255.0

    # Reshape the frame to match the model's input shape
    input_frame = np.reshape(normalized_frame, (1, 48, 48, 1))

    # Emotion detection code using the loaded model
    predictions = emotion_model.predict(input_frame)

    # Map model predictions to human-readable emotion labels (adjust as per your model's output)
    emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    detected_emotion_index = np.argmax(predictions)
    detected_emotion = emotion_labels[detected_emotion_index]

    # Return the detected emotion
    return detected_emotion


# Main loop
cap = cv2.VideoCapture(0)  # Replace '0' with the camera index or video file path
running = True
while running:
    ret, frame = cap.read()
    if not ret:
        break

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Detect emotions
    detected_emotion = detect_emotion(frame)

    # Clear the screen
    screen.fill(BLACK)

    # Draw based on detected emotion
    if detected_emotion == "sad":
        # Example: Draw a simple representation for joy
        for i in range(0, width, 20):
            for j in range(0, height, 20):
                color = (i % 255, j % 255, (i + j) % 255)
                pygame.draw.rect(screen, color, (i, j, 20, 20))

    # Update the display
    pygame.display.flip()

# Release the camera and quit Pygame
cap.release()
pygame.quit()
