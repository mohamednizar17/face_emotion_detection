import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from tensorflow.keras.models import load_model
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
        if 'last_result' not in st.session_state:
            st.session_state.last_result = None

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
                st.session_state.last_result = result
            else:
                result = st.session_state.last_result

            if result is not None:
                stframe.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            else:
                # Handle pause on first iteration
                ret, frame = cap.read()
                if not ret:
                    break
                result = detect_emotion(frame, mode="ensemble")
                st.session_state.last_result = result
                stframe.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

            if st.session_state.paused:
                time.sleep(0.1)

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
