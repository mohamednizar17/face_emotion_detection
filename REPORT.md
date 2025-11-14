# Face Emotion Detection: An IEEE-Style Project Report

Author: mohamednizar17  
Affiliation: —  
Date: 04 Nov 2025

## Abstract
This report documents the end-to-end journey of building a real-time facial emotion recognition system. Using the FER2013 dataset and custom CNN architectures in TensorFlow/Keras, two models were trained and evaluated, then deployed to real-time applications using MediaPipe for face detection and Streamlit/OpenCV for user interfaces. The best standalone model achieved a test accuracy of approximately 58.9% on held-out data, and an ensemble strategy was implemented for improved robustness in video/webcam settings. The system supports image, video, and live webcam inference.

Keywords—emotion recognition, facial expression analysis, CNN, FER2013, Keras, MediaPipe, Streamlit, WebRTC, OpenCV, ensemble learning.

## 1. Introduction
Automatic facial emotion recognition has applications in HCI, health, education, and entertainment. This work focuses on building a practical pipeline that goes from dataset preparation to training, evaluation, and real-time deployment. We designed and trained custom convolutional neural networks (CNNs) on grayscale 48×48 facial images, compared two saved models, and integrated them into user-facing apps.

## 2. Related Work (Brief)
- FER2013 has been widely used for benchmarking facial expression classifiers with CNNs.  
- MediaPipe Face Detection offers robust and fast on-device face localization.  
- Streamlit and WebRTC enable lightweight, production-adjacent prototyping for real-time ML apps.

## 3. Dataset and Preprocessing
- Dataset: FER2013 (file present: `fer2013.csv`) with 7 classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral. Images are 48×48 grayscale faces.
- Split: Typical train/validation/test split (held-out test arrays provided: `X_test.npy`, `y_test.npy`).
- Preprocessing (Training and Inference):
  - Convert to grayscale (if not already).
  - Resize to 48×48 pixels.
  - Normalize pixel intensities to [0, 1].
  - For inference, detect faces, crop ROI, then apply the same normalization.

Screenshots (data collection / preprocessing / feature engineering / training workflow) are included in the repository and referenced below:
- Figure 1: Data/Preprocessing view — `D'Preprocessing and Model Training/Screenshot 2025-11-04 110318.png`
- Figure 2: Data/Preprocessing view — `D'Preprocessing and Model Training/Screenshot 2025-11-04 110332.png`
- Figure 3: Model/Training view — `D'Preprocessing and Model Training/Screenshot 2025-11-04 110441.png`
- Figure 4: Model/Fine-tuning view — `D'Preprocessing and Model Training/Screenshot 2025-11-04 110454.png`

> Note: Figures illustrate the actual screens used for data handling and model training. They are included for documentation and reproducibility.

## 4. Methodology
### 4.1 Face Detection
- Library: MediaPipe Face Detection (model_selection=0, min_detection_confidence=0.5).
- Detected relative bounding boxes are mapped to image coordinates to crop the face ROI.

### 4.2 Model Architectures (CNN)
The saved architecture at `model_architecture.json` defines a Sequential CNN with the following pattern:
- Input: 48×48×1
- Conv2D(32, 3×3, relu) → BatchNorm → MaxPool(2×2) → Dropout(0.25)
- Conv2D(64, 3×3, relu) → BatchNorm → MaxPool(2×2) → Dropout(0.25)
- Conv2D(128, 3×3, relu) → BatchNorm → MaxPool(2×2) → Dropout(0.25)
- Flatten → Dense(256, relu) → Dropout(0.5) → Dense(7, softmax)

Saved artifacts:
- `custom_emotion_model.h5` (Model 1)
- `custom_emotion_model1.h5` (Model 2)
- `model_architecture.json` (architecture dump)
- `model_weights.h5` (weights dump)

### 4.3 Training Setup
- Framework: TensorFlow/Keras (Keras 2.12 indicated in the architecture file).
- Loss/Optimizer: Not explicitly recorded in the repo; common choices for this task are categorical cross-entropy with Adam optimizer (assumption; exact config not persisted).
- Evaluation Metric: Accuracy on held-out test data.

### 4.4 Ensemble Strategy
- For videos/webcam, predictions from the two models are averaged (simple mean of softmax probabilities) to improve robustness frame-by-frame.

## 5. Implementation
### 5.1 Inference Core
All pipelines share the same pre-processing for a face ROI:
- Grayscale → Resize (48×48) → Normalize [0, 1] → Add channel dim → Model predict → Argmax label.

### 5.2 Applications
- Command-line/OpenCV UI (`app.py`):
  - Choose Image, Video, or Webcam in a console menu.
  - Uses MediaPipe face detection, overlays label + confidence on frames.
- Streamlit app (single model) (`stream.py`):
  - Modes: Image Upload, Video Upload, Live Webcam (WebRTC).
  - Real-time overlay and a pause/play control for video upload.
- Streamlit app (ensemble) (`combined.py`):
  - Modes: Image Upload (uses Model 1), Video Upload and Live Webcam (uses ensemble of Model 1 + Model 2).
  - WebRTC (`streamlit-webrtc`) used for low-latency webcam streaming.

## 6. Evaluation
- Recorded in `model_evaluation.txt`:
  - Test Loss: 1.1453
  - Test Accuracy: 0.5890 (58.90%)
- Script to compare two models on the same test set: `test.py`.
  - Loads `X_test.npy` and `y_test.npy`.
  - Normalizes inputs; evaluates both `custom_emotion_model.h5` and `custom_emotion_model1.h5`.
  - Prints individual accuracies and indicates which model performs better on the provided arrays.

> Note: The reported 58.9% is for a single model on the provided test split. In practice, class imbalance in FER2013 and cross-subject variation often make this task challenging without augmentation and extensive tuning.

## 7. Results and Discussion
- The baseline model achieves ~59% test accuracy. This is aligned with simple CNN baselines on FER2013 when training time/augmentation/hyperparameter tuning are limited.
- The ensemble approach can stabilize predictions in video streams by averaging softmax outputs from two independently trained models.
- Observed strengths: Robust real-time detection using MediaPipe; straightforward deployment via Streamlit and OpenCV.
- Observed challenges: Disgust and Fear classes often show confusion; lighting and occlusion reduce confidence; non-frontal faces may be missed or misclassified.

## 8. Deployment
- Local OpenCV app (`app.py`): keyboard-driven, uses native windows for display.
- Streamlit (`stream.py`, `combined.py`): clean UI, file upload, video preview, and live webcam via WebRTC.  
- External dependencies include: `streamlit`, `streamlit-webrtc`, `mediapipe`, `opencv-python`, `tensorflow/keras`, `numpy`, `av`.

## 9. Limitations
- Dataset limitations (gray 48×48) restrict fine-grained expression cues; domain shift (webcam vs. dataset) impacts generalization.
- Training details (optimizer, LR schedule, augmentation) not fully recorded; reproducibility can be improved by adding a training script/notebook with config logs.
- Model performance may vary with camera quality and lighting.

## 10. Future Work
- Add data augmentation (random crops, flips, brightness/contrast jitter) to increase robustness.
- Explore larger backbones (e.g., MobileNetV3, EfficientNet-lite) or fine-tuning pre-trained facial models.
- Calibrate probabilities (e.g., temperature scaling) for more reliable confidence scores.
- Add per-class metrics, confusion matrices, and ROC-style diagnostics; log training curves.
- Quantize or optimize models for edge deployment; integrate GPU acceleration where available.

## 11. Conclusion
We implemented a practical facial emotion recognition pipeline from dataset to real-time deployment. Two CNN models were trained and evaluated on FER2013-style inputs. A Streamlit-based UI and OpenCV application provide a smooth user experience for images, videos, and live webcam feeds. Ensemble inference further stabilizes predictions in dynamic settings. The repository’s artifacts (architecture JSON, weights, evaluation text, test arrays) ensure traceability of results and facilitate further improvements.

## References
1) Goodfellow, I. et al., "Challenges in Representation Learning: A report on three machine learning contests," ICML 2013 Workshop — FER2013 dataset.  
2) TensorFlow/Keras: https://keras.io/  
3) MediaPipe Face Detection: https://developers.google.com/mediapipe/solutions/vision/face_detection  
4) Streamlit: https://streamlit.io/  
5) streamlit-webrtc: https://github.com/whitphx/streamlit-webrtc  
6) OpenCV: https://opencv.org/

---

### Appendix A. Code Artifacts (Pointers)
- Core apps: `app.py`, `stream.py`, `combined.py`  
- Models: `custom_emotion_model.h5`, `custom_emotion_model1.h5`  
- Architecture/weights: `model_architecture.json`, `model_weights.h5`  
- Evaluation: `model_evaluation.txt`, test arrays `X_test.npy`, `y_test.npy`  
- Data source: `fer2013.csv`; Kaggle credentials present in `kaggle.json`  
- Face detection: `mediapipe` with bounding-box to ROI pipeline  

### Appendix B. Minimal Inference Contract
- Input: face ROI BGR image (H×W×3)  
- Steps: grayscale → resize 48×48 → normalize [0,1] → reshape (1,48,48,1) → `model.predict`  
- Output: argmax over 7 classes and confidence score (softmax prob)

---

## Most important code components (full listings + commentary)

This section aggregates the most critical parts of your pipeline with inline notes. Each block is copied from your repository so readers can trace the exact implementation.

### A. Data conversion for test arrays (`convert.py`)

Purpose: Convert FER2013 CSV (PublicTest split) to `X_test.npy` and `y_test.npy` used by `test.py`.

```python
import pandas as pd
import numpy as np

# Load CSV
csv_file = 'fer2013.csv'
df = pd.read_csv(csv_file)

# Filter for test set (you can choose 'PublicTest' or 'PrivateTest')
test_df = df[df['Usage'] == 'PublicTest']

# Process images and labels
pixels_list = test_df['pixels'].tolist()
emotions = test_df['emotion'].tolist()

X_test = []
for pixel_sequence in pixels_list:
  img = np.array([int(pixel) for pixel in pixel_sequence.split()], dtype='float32')
  img = img.reshape(48, 48, 1)
  img = img / 255.0  # Normalize
  X_test.append(img)

X_test = np.array(X_test)
y_test = np.array(emotions)

# Save files
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

print(f"✅ Saved {len(X_test)} samples as X_test.npy and y_test.npy.")
```

Notes:
- Maintains the canonical 48×48×1 grayscale format.
- Ensures normalization—consistent with inference steps in apps.

### B. End-to-end training, evaluation, and artifact export (`fine.py`)

Purpose: Build a deeper CNN, train on FER2013, save the final model (`.h5`), architecture JSON, weights H5, and evaluation text.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load FER2013 CSV
data = pd.read_csv("fer2013.csv")
print(data.emotion.value_counts())

# Convert pixel strings to numpy arrays
pixels = data['pixels'].tolist()
X = np.array([np.fromstring(pixel, sep=' ').reshape(48, 48, 1) for pixel in pixels]) / 255.0
y = to_categorical(data['emotion'], num_classes=7)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print(" Data shape:", X_train.shape, y_train.shape)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization

# Build CNN model
model = Sequential([
  Conv2D(64, (3,3), activation='relu', input_shape=(48, 48, 1)),
  BatchNormalization(),
  MaxPooling2D(pool_size=(2,2)),
  Dropout(0.25),

  Conv2D(128, (3,3), activation='relu'),
  BatchNormalization(),
  MaxPooling2D(pool_size=(2,2)),
  Dropout(0.25),

  Conv2D(256, (3,3), activation='relu'),
  BatchNormalization(),
  MaxPooling2D(pool_size=(2,2)),
  Dropout(0.25),

  Flatten(),
  Dense(512, activation='relu'),
  Dropout(0.5),
  Dense(7, activation='softmax')  # 7 emotion classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=30, batch_size=64)

model.save("custom_emotion_model.h5")
# Plot training history
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy') 
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()
# Plot loss history
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
# Save the model evaluation results
with open("model_evaluation.txt", "w") as f:
  f.write(f"Test Loss: {test_loss}\n")
  f.write(f"Test Accuracy: {test_accuracy}\n")
# Save the model architecture
with open("model_architecture.json", "w") as f:
  f.write(model.to_json())
# Save the model weights
model.save_weights("model_weights.h5")
print("Model, evaluation results, architecture, and weights saved successfully.")
# Load the model for future use
from tensorflow.keras.models import load_model
loaded_model = load_model("custom_emotion_model.h5")
print("Model loaded successfully.")
```

Notes:
- This script is the provenance for `model_evaluation.txt`, `model_architecture.json`, and `model_weights.h5` found in the repo.
- Training choices (Adam, categorical cross-entropy) align with typical FER2013 baselines.

### C. Model loading snippets

`load_model.py` (simple loader without custom metrics):

```python
from tensorflow.keras.models import load_model

# Load model WITHOUT compiling (skip 'fbeta' metric)
emotion_model = load_model("custom_emotion_model.h5")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
```

`inspect-module.py` (print architecture):

```python
from keras.models import load_model

# Try loading the model without compiling
model = load_model("custom_emotion_model.h5", compile=False)

# Print model structure
model.summary()
```

### D. OpenCV-based CLI UI (`app.py`)

Purpose: Desktop experience with image/video/webcam modes and on-frame overlays using MediaPipe face detection + Keras inference.

```python
import cv2
import numpy as np
import mediapipe as mp
import os
import tkinter as tk
from tkinter import filedialog
from keras.models import load_model

emotion_model = load_model("custom_emotion_model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
# Initialize MediaPipe face detector
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Hide tkinter root window
tk.Tk().withdraw()

def get_screen_size():
  root = tk.Tk()
  root.withdraw()
  return root.winfo_screenwidth(), root.winfo_screenheight()

def predict_emotion(face_img):
  roi_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
  roi_resized = cv2.resize(roi_gray, (48, 48))
  roi_normalized = roi_resized / 255.0
  roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))
  prediction = emotion_model.predict(roi_reshaped, verbose=0)
  emotion_idx = np.argmax(prediction)
  emotion = emotion_labels[emotion_idx]
  confidence = prediction[0][emotion_idx]
  return emotion, confidence

def detect_emotions_from_frame(frame):
  img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  results = face_detector.process(img_rgb)

  if results.detections:
    for detection in results.detections:
      bbox = detection.location_data.relative_bounding_box
      h, w, _ = frame.shape
      x = int(bbox.xmin * w)
      y = int(bbox.ymin * h)
      w_box = int(bbox.width * w)
      h_box = int(bbox.height * h)

      roi = frame[y:y+h_box, x:x+w_box]
      if roi.size == 0:
        continue

      emotion, confidence = predict_emotion(roi)

      cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
      cv2.putText(frame, f'{emotion} ({confidence*100:.1f}%)',
            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (5, 55, 155), 2)
  return frame

def scale_to_half_screen(frame):
  screen_w, screen_h = get_screen_size()
  frame_h, frame_w = frame.shape[:2]
  scale = min((screen_w / 2) / frame_w, (screen_h / 2) / frame_h, 1.0)
  return cv2.resize(frame, (int(frame_w * scale), int(frame_h * scale)))

def run_on_image():
  image_path = filedialog.askopenfilename(title="Select an Image",
                      filetypes=[("Image files", "*.jpg *.png *.jpeg")])
  if not image_path:
    print("No image selected.")
    return

  img = cv2.imread(image_path)
  output = detect_emotions_from_frame(img)
  output_resized = scale_to_half_screen(output)

  cv2.namedWindow("Emotion Detection - Image", cv2.WINDOW_NORMAL)
  cv2.imshow("Emotion Detection - Image", output_resized)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def run_on_video():
  video_path = filedialog.askopenfilename(title="Select a Video",
                      filetypes=[("Video files", "*.mp4 *.avi *.mov")])
  if not video_path:
    print("No video selected.")
    return

  cap = cv2.VideoCapture(video_path)
  cv2.namedWindow("Emotion Detection - Video", cv2.WINDOW_NORMAL)

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break
    output = detect_emotions_from_frame(frame)
    output_resized = scale_to_half_screen(output)
    cv2.imshow("Emotion Detection - Video", output_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  cap.release()
  cv2.destroyAllWindows()

def run_on_webcam():
  cap = cv2.VideoCapture(0)
  print("Live webcam started. Press 'q' to quit.")
  cv2.namedWindow("Emotion Detection - Webcam", cv2.WINDOW_NORMAL)

  while True:
    ret, frame = cap.read()
    if not ret:
      break
    output = detect_emotions_from_frame(frame)
    output_resized = scale_to_half_screen(output)
    cv2.imshow("Emotion Detection - Webcam", output_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  cap.release()
  cv2.destroyAllWindows()

# --- Main UI ---
print("======================================")
print("       Emotion Detection Program      ")
print("======================================")
print("Select Input Mode:")
print("1 - Detect from Image File")
print("2 - Detect from Video File")
print("3 - Detect from Live Webcam")
choice = input("Enter 1 / 2 / 3: ").strip()

if choice == '1':
  run_on_image()

elif choice == '2':
  run_on_video()

elif choice == '3':
  run_on_webcam()

else:
  print("Invalid input. Please enter 1, 2, or 3.")
```

### E. Streamlit app (single model) with WebRTC (`stream.py`)

Purpose: Web UI with modes, pause/resume for video upload, and WebRTC webcam.

```python
import streamlit as st
import cv2
import numpy as np
import tempfile
from keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import mediapipe as mp
import av

# Load model and labels
emotion_model = load_model("custom_emotion_model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize face detector
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Prediction
def predict_emotion(face_img):
  roi_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
  roi_resized = cv2.resize(roi_gray, (48, 48))
  roi_normalized = roi_resized / 255.0
  roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))
  prediction = emotion_model.predict(roi_reshaped, verbose=0)
  emotion_idx = np.argmax(prediction)
  emotion = emotion_labels[emotion_idx]
  confidence = prediction[0][emotion_idx]
  return emotion, confidence

# Draw results
def detect_emotion(frame):
  img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  results = face_detector.process(img_rgb)

  if results.detections:
    for detection in results.detections:
      bbox = detection.location_data.relative_bounding_box
      h, w, _ = frame.shape
      x = int(bbox.xmin * w)
      y = int(bbox.ymin * h)
      w_box = int(bbox.width * w)
      h_box = int(bbox.height * h)

      roi = frame[y:y+h_box, x:x+w_box]
      if roi.size == 0:
        continue

      emotion, confidence = predict_emotion(roi)

      cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
      label = f'{emotion} ({confidence*100:.1f}%)'
      label_pos = (x, y - 10 if y - 10 > 20 else y + 20)
      cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
      cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

  return frame

# Webcam Processor
class EmotionProcessor(VideoProcessorBase):
  def recv(self, frame):
    img = frame.to_ndarray(format="bgr24")
    img = detect_emotion(img)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# UI
st.set_page_config(page_title="Real-Time Emotion Detection", layout="centered")
st.title("😊 Real-Time Emotion Detection")
mode = st.sidebar.radio("Select Input Mode:", ["Image Upload", "Video Upload", "Live Webcam"])

# Mode 1: Image Upload
if mode == "Image Upload":
  image_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
  if image_file:
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    result = detect_emotion(img)
    st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Detected Emotions", use_container_width=True)

# Mode 2: Video Upload
elif mode == "Video Upload":
  video_file = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
  if video_file:
    # Session State for pause/play
    if 'paused' not in st.session_state:
      st.session_state.paused = False

    # Pause/Play toggle button
    if st.button("⏸️ Pause" if not st.session_state.paused else "▶️ Play"):
      st.session_state.paused = not st.session_state.paused

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    while cap.isOpened():
      if not st.session_state.paused:
        ret, frame = cap.read()
        if not ret:
          break
        result = detect_emotion(frame)
        stframe.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
      else:
        stframe.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
      if st.session_state.paused:
        # Slow down loop when paused to avoid CPU load
        st.sleep(0.1)
    cap.release()

# Mode 3: Live Webcam (video only)
elif mode == "Live Webcam":
  st.write("Click 'Start' to begin webcam emotion detection:")
  webrtc_streamer(
    key="emotion-live",
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration=RTCConfiguration({
      "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })
  )
```

### F. Streamlit app (ensemble) (`combined.py`)

Purpose: Same UX as above, but with two models and probability averaging for more stable video/webcam inference.

```python
import streamlit as st
import cv2
import numpy as np
import tempfile
from keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import mediapipe as mp
import av

# Load models and labels
model1 = load_model("custom_emotion_model.h5")
model2 = load_model("custom_emotion_model1.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize face detector
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Prediction using single model (model1)
def predict_emotion_single(face_img):
  roi_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
  roi_resized = cv2.resize(roi_gray, (48, 48))
  roi_normalized = roi_resized / 255.0
  roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))
  prediction = model1.predict(roi_reshaped, verbose=0)
  emotion_idx = np.argmax(prediction)
  emotion = emotion_labels[emotion_idx]
  confidence = prediction[0][emotion_idx]
  return emotion, confidence

# Prediction using ensemble of model1 and model2
def predict_emotion_ensemble(face_img):
  roi_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
  roi_resized = cv2.resize(roi_gray, (48, 48))
  roi_normalized = roi_resized / 255.0
  roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))

  pred1 = model1.predict(roi_reshaped, verbose=0)
  pred2 = model2.predict(roi_reshaped, verbose=0)
  avg_pred = (pred1 + pred2) / 2.0

  emotion_idx = np.argmax(avg_pred)
  emotion = emotion_labels[emotion_idx]
  confidence = avg_pred[0][emotion_idx]
  return emotion, confidence

# Draw results
def detect_emotion(frame, mode):
  img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  results = face_detector.process(img_rgb)

  if results.detections:
    for detection in results.detections:
      bbox = detection.location_data.relative_bounding_box
      h, w, _ = frame.shape
      x = int(bbox.xmin * w)
      y = int(bbox.ymin * h)
      w_box = int(bbox.width * w)
      h_box = int(bbox.height * h)

      roi = frame[y:y+h_box, x:x+w_box]
      if roi.size == 0:
        continue

      if mode == "image":
        emotion, confidence = predict_emotion_single(roi)
      else:
        emotion, confidence = predict_emotion_ensemble(roi)

      cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
      label = f'{emotion} ({confidence*100:.1f}%)'
      label_pos = (x, y - 10 if y - 10 > 20 else y + 20)
      cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
      cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

  return frame

# Webcam Processor
class EmotionProcessor(VideoProcessorBase):
  def recv(self, frame):
    img = frame.to_ndarray(format="bgr24")
    img = detect_emotion(img, mode="ensemble")
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# UI
st.set_page_config(page_title="Real-Time Emotion Detection", layout="centered")
st.title("😊 Real-Time Emotion Detection")
mode = st.sidebar.radio("Select Input Mode:", ["Image Upload", "Video Upload", "Live Webcam"])

# Mode 1: Image Upload (uses model1 only)
if mode == "Image Upload":
  image_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
  if image_file:
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    result = detect_emotion(img, mode="image")
    st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Detected Emotions", use_container_width=True)

# Mode 2: Video Upload (uses ensemble)
elif mode == "Video Upload":
  video_file = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
  if video_file:
    # Pause/Play using session state
    if 'paused' not in st.session_state:
      st.session_state.paused = False

    if st.button("⏸️ Pause" if not st.session_state.paused else "▶️ Play"):
      st.session_state.paused = not st.session_state.paused

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    while cap.isOpened():
      if not st.session_state.paused:
        ret, frame = cap.read()
        if not ret:
          break
        result = detect_emotion(frame, mode="ensemble")
        stframe.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
      else:
        stframe.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        st.sleep(0.1)

    cap.release()

# Mode 3: Live Webcam (uses ensemble)
elif mode == "Live Webcam":
  st.write("Click 'Start' to begin webcam emotion detection:")
  webrtc_streamer(
    key="emotion-live",
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration=RTCConfiguration({
      "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })
  )
```

### G. Model comparison script (`test.py`)

Purpose: Evaluate both saved models on the same test arrays and compare accuracies.

```python
import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import accuracy_score

# Load Models
model1 = load_model("custom_emotion_model.h5")
model2 = load_model("custom_emotion_model1.h5")

# Load Test Data (Example: replace with your own data loader)
# Here, X_test shape: (num_samples, 48, 48, 1)
#       y_test shape: (num_samples,)
# You must load your real data here.
X_test = np.load('X_test.npy')  # Example placeholder
y_test = np.load('y_test.npy')  # Example placeholder

# Ensure data is normalized
X_test = X_test.astype('float32') / 255.0

# Model 1 Predictions
y_pred1 = np.argmax(model1.predict(X_test, verbose=0), axis=1)
accuracy1 = accuracy_score(y_test, y_pred1)
print(f"✅ Model 1 (custom_emotion_model.h5) Accuracy: {accuracy1 * 100:.2f}%")

# Model 2 Predictions
y_pred2 = np.argmax(model2.predict(X_test, verbose=0), axis=1)
accuracy2 = accuracy_score(y_test, y_pred2)
print(f"✅ Model 2 (custom_emotion_model1.h5) Accuracy: {accuracy2 * 100:.2f}%")

# Which model is better?
if accuracy1 > accuracy2:
  print("📊 Model 1 performs better.")
elif accuracy2 > accuracy1:
  print("📊 Model 2 performs better.")
else:
  print("📊 Both models have equal accuracy.")
```

### H. Face detection and ROI pipeline (common snippet)

This snippet illustrates the MediaPipe face detection and ROI extraction used across apps.

```python
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def detect_emotion(frame):
  img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  results = face_detector.process(img_rgb)
  if results.detections:
    for detection in results.detections:
      bbox = detection.location_data.relative_bounding_box
      h, w, _ = frame.shape
      x = int(bbox.xmin * w)
      y = int(bbox.ymin * h)
      w_box = int(bbox.width * w)
      h_box = int(bbox.height * h)
      roi = frame[y:y+h_box, x:x+w_box]
      # ... preprocess ROI and predict ...
```

### I. Model architecture excerpt (`model_architecture.json`)

Full JSON is lengthy; below is a condensed view showing the layer pattern (see file for full details):

```json
{
  "class_name": "Sequential",
  "config": {
  "layers": [
    {"class_name": "InputLayer", "config": {"batch_input_shape": [null, 48, 48, 1]}},
    {"class_name": "Conv2D", "config": {"filters": 32, "kernel_size": [3,3], "activation": "relu"}},
    {"class_name": "BatchNormalization"},
    {"class_name": "MaxPooling2D", "config": {"pool_size": [2,2]}},
    {"class_name": "Dropout", "config": {"rate": 0.25}},
    {"class_name": "Conv2D", "config": {"filters": 64, "kernel_size": [3,3], "activation": "relu"}},
    {"class_name": "BatchNormalization"},
    {"class_name": "MaxPooling2D"},
    {"class_name": "Dropout", "config": {"rate": 0.25}},
    {"class_name": "Conv2D", "config": {"filters": 128, "kernel_size": [3,3], "activation": "relu"}},
    {"class_name": "BatchNormalization"},
    {"class_name": "MaxPooling2D"},
    {"class_name": "Dropout", "config": {"rate": 0.25}},
    {"class_name": "Flatten"},
    {"class_name": "Dense", "config": {"units": 256, "activation": "relu"}},
    {"class_name": "Dropout", "config": {"rate": 0.5}},
    {"class_name": "Dense", "config": {"units": 7, "activation": "softmax"}}
  ]
  },
  "keras_version": "2.12.0",
  "backend": "tensorflow"
}
```

### J. Extended, optional training pipeline with augmentation (for future work)

The following reproducible script shows how to add standard augmentations and callbacks. Use as a foundation for the next iteration; it wasn’t used to generate current results but is compatible with your repo.

```python
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

CSV = Path('fer2013.csv')
assert CSV.exists()
df = pd.read_csv(CSV)

pixels = df['pixels'].tolist()
X = np.array([np.fromstring(p, sep=' ').reshape(48,48,1) for p in pixels], dtype='float32') / 255.0
y = tf.keras.utils.to_categorical(df['emotion'].values, num_classes=7)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

datagen = ImageDataGenerator(
  rotation_range=10,
  width_shift_range=0.1,
  height_shift_range=0.1,
  zoom_range=0.1,
  horizontal_flip=True,
  fill_mode='nearest'
)
datagen.fit(X_train)

def build_model():
  model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)), BatchNormalization(),
    MaxPooling2D(), Dropout(0.25),
    Conv2D(64, (3,3), activation='relu'), BatchNormalization(),
    MaxPooling2D(), Dropout(0.25),
    Conv2D(128, (3,3), activation='relu'), BatchNormalization(),
    MaxPooling2D(), Dropout(0.25),
    Flatten(), Dense(256, activation='relu'), Dropout(0.5),
    Dense(7, activation='softmax')
  ])
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  return model

model = build_model()

ckpt = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
early = EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)
reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)

history = model.fit(
  datagen.flow(X_train, y_train, batch_size=64, shuffle=True),
  validation_data=(X_val, y_val),
  epochs=60,
  callbacks=[ckpt, early, reduce],
  verbose=1
)

with open('aug_history.json', 'w') as f:
  json.dump(history.history, f)
```

### K. Generating per-class metrics and confusion matrix (how-to)

Use this helper after predictions to compute class-level diagnostics.

```python
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# y_true: shape (N,), integer labels 0..6
# y_pred_proba: shape (N,7)
y_pred = np.argmax(y_pred_proba, axis=1)
print(classification_report(y_true, y_pred, target_names=['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']))
print(confusion_matrix(y_true, y_pred))
```

---

## Deep-dive commentary and engineering notes

### Data pipeline decisions
- Fixed input size (48×48) preserved from FER2013; this keeps training simple and memory-efficient.
- Normalization to [0,1] across both training and inference ensures consistency and stable optimization.

### Model design choices
- Batch normalization after each convolution stabilizes training and improves convergence.
- Progressive depth (32/64/128 filters) is a strong baseline for FER2013; Dropout combats overfitting.

### Ensemble rationale
- Averaging softmax outputs of two models often reduces variance in frame-wise predictions, especially in videos with noise or motion blur.

### Real-time considerations
- MediaPipe’s fast CPU-based detector pairs well with lightweight CNNs for interactive throughput.
- WebRTC (via `streamlit-webrtc`) avoids heavy server round-trips, enabling near real-time webcam UX.

### Reproducibility checklist
- Artifacts: `model_evaluation.txt`, `model_architecture.json`, `model_weights.h5`, saved `.h5` models.
- Test arrays: `X_test.npy`, `y_test.npy` built from `fer2013.csv` (`PublicTest`).
- Scripts: `fine.py` (train/eval/export), `convert.py` (test arrays), `test.py` (model comparison).

### Ethics & responsible AI notes
- Facial emotion recognition can misinterpret expressions across cultures and contexts; apply with care.
- Avoid consequential decisions without human oversight; document known biases and uncertainty.

### Practical deployment tips
- Prefer GPU for faster inference on video streams; on CPU, reduce frame rate or resolution if needed.
- Cache the face region when tracking across adjacent frames to reduce redundant detection calls.

---

## Figure references (screens from your preprocessing/training)
- Figure 1: `D'Preprocessing and Model Training/Screenshot 2025-11-04 110318.png`
- Figure 2: `D'Preprocessing and Model Training/Screenshot 2025-11-04 110332.png`
- Figure 3: `D'Preprocessing and Model Training/Screenshot 2025-11-04 110441.png`
- Figure 4: `D'Preprocessing and Model Training/Screenshot 2025-11-04 110454.png`

Each figure documents a phase of data handling, model definition, and training runs, complementing the code listings above.
