import cv2
from keras.models import load_model
import numpy as np

# Load the pre-trained Haar Cascade classifiers for face and eye detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Load the pre-trained emotion recognition model
emotion_model = load_model(r'E:\model.h5')

# Define emotion labels
emotion_labels = ('angry', 'disgust', 'fear', 'happy','neutral', 'sad', 'surprise', )

# Specify paths to the age and gender models
age_model_prototxt = r'E:\age_deploy.prototxt'
age_model_caffemodel = r'E:\age_net.caffemodel'
gender_model_prototxt = r'E:\gender_deploy.prototxt'
gender_model_caffemodel = r'E:\gender_net.caffemodel'

# Load the pre-trained age and gender prediction models
try:
    age_net = cv2.dnn.readNetFromCaffe(age_model_prototxt, age_model_caffemodel)
    gender_net = cv2.dnn.readNetFromCaffe(gender_model_prototxt, gender_model_caffemodel)
except Exception as e:
    print("Error loading age and gender models:", e)
    exit()

# Define age and gender lists
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# Function to draw a grid on the face region
def draw_face_grid(frame, x, y, w, h, grid_size=5):
    step_x = w // grid_size
    step_y = h // grid_size
    
    for i in range(1, grid_size):
        # Draw vertical lines
        cv2.line(frame, (x + i * step_x, y), (x + i * step_x, y + h), (0, 255, 255), 1)
        # Draw horizontal lines
        cv2.line(frame, (x, y + i * step_y), (x + w, y + i * step_y), (0, 255, 255), 1)

# Function to detect faces and predict emotions, age, and gender
def detect_features(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        face_roi = gray_image[y:y+h, x:x+w]
        preprocessed_face = preprocess_input(face_roi)
        prediction = emotion_model.predict(preprocessed_face)
        emotion_index = np.argmax(prediction)
        emotion_label = emotion_labels[emotion_index]
        print("Predicted Emotion:", emotion_label)
        
        # Draw rectangle around the face and display emotion label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        # Draw face grid
        draw_face_grid(frame, x, y, w, h)
        
        # Detect eyes within the face region
        face_region_color = frame[y:y+h, x:x+w]
        eyes = eye_classifier.detectMultiScale(face_region_color)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_region_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
        
        # Predict age and gender
        face_rgb = cv2.cvtColor(face_region_color, cv2.COLOR_BGR2RGB)
        blob = cv2.dnn.blobFromImage(face_rgb, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        
        # Predict gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = GENDER_LIST[gender_preds[0].argmax()]
        
        # Predict age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = AGE_LIST[age_preds[0].argmax()]
        
        label = f"{gender}, {age}"
        cv2.putText(frame, label, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
    return frame

# Function to preprocess input image for emotion recognition
def preprocess_input(face):
    face = cv2.resize(face, (48, 48))
    face = np.expand_dims(face, axis=-1)  # Add channel dimension
    face = np.expand_dims(face, axis=0)  # Add batch dimension
    face = face / 255.0  # Normalize to [0,1]
    return face

# Access the webcam
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open camera.")
    exit()

# Main loop for capturing frames from the webcam
while True:
    ret, frame = video_capture.read()

    if not ret:
        break

    # Detect faces and predict emotions, age, and gender
    frame_with_features = detect_features(frame)

    # Display the processed frame
    cv2.imshow("Emotion Detection, Age, Gender, and Eye Tracking", frame_with_features)

    # Press 'q' to exit the 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
