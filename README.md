# Emotion Detection using Facial Recognition amd CNN

This project demonstrates facial detection combined with emotion detection using a Convolutional Neural Network (CNN) model trained with TensorFlow. Additionally, it predicts age, gender, and tracks eye movements.

## Requirements

- Python 3.x
- OpenCV
- Keras
- TensorFlow
- NumPy

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/emotion-detection.git
    cd emotion-detection
    ```

2. **Install the required packages:**

    ```bash
    pip install opencv-python-headless keras tensorflow numpy
    ```

3. **Download Pre-trained Models:**
    - Place the pre-trained emotion recognition model (`model.h5`), age and gender model files (`age_deploy.prototxt`, `age_net.caffemodel`, `gender_deploy.prototxt`, `gender_net.caffemodel`) in the project directory.

## Usage

1. **Run the script:**

    ```bash
    python main.py
    ```

2. **Press `q` to quit** the application.

## Code Overview

### Importing Libraries

```python
import cv2
from keras.models import load_model
import numpy as np
```

### Loading trained Models

```python
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
emotion_model = load_model(r'E:\model.h5')

# Paths to age and gender models
age_model_prototxt = r'E:\age_deploy.prototxt'
age_model_caffemodel = r'E:\age_net.caffemodel'
gender_model_prototxt = r'E:\gender_deploy.prototxt'
gender_model_caffemodel = r'E:\gender_net.caffemodel'

# Load age and gender models
age_net = cv2.dnn.readNetFromCaffe(age_model_prototxt, age_model_caffemodel)
gender_net = cv2.dnn.readNetFromCaffe(gender_model_prototxt, gender_model_caffemodel)
```

### Emotion Labels

```python
emotion_labels = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')
```

### Age and Gender Lists

```python
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
```

### Functions

- **Draw a Grid on the Face:**

    ```python
    def draw_face_grid(frame, x, y, w, h, grid_size=5):
        step_x = w // grid_size
        step_y = h // grid_size
        
        for i in range(1, grid_size):
            cv2.line(frame, (x + i * step_x, y), (x + i * step_x, y + h), (0, 255, 255), 1)
            cv2.line(frame, (x, y + i * step_y), (x + w, y + i * step_y), (0, 255, 255), 1)
    ```

- **Detect Features:**

    ```python
    def detect_features(frame):
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        for (x, y, w, h) in faces:
            face_roi = gray_image[y:y+h, x:x+w]
            preprocessed_face = preprocess_input(face_roi)
            prediction = emotion_model.predict(preprocessed_face)
            emotion_index = np.argmax(prediction)
            emotion_label = emotion_labels[emotion_index]
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            
            draw_face_grid(frame, x, y, w, h)
            
            face_region_color = frame[y:y+h, x:x+w]
            eyes = eye_classifier.detectMultiScale(face_region_color)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(face_region_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
            
            face_rgb = cv2.cvtColor(face_region_color, cv2.COLOR_BGR2RGB)
            blob = cv2.dnn.blobFromImage(face_rgb, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = GENDER_LIST[gender_preds[0].argmax()]
            
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = AGE_LIST[age_preds[0].argmax()]
            
            label = f"{gender}, {age}"
            cv2.putText(frame, label, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        return frame
    ```

- **Preprocess Input:**

    ```python
    def preprocess_input(face):
        face = cv2.resize(face, (48, 48))
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)
        face = face / 255.0
        return face
    ```

### Access Webcam and Main Loop

```python
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_with_features = detect_features(frame)
    cv2.imshow("Emotion Detection, Age, Gender, and Eye Tracking", frame_with_features)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
```

## Contributing

If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

---

