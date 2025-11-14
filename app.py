import cv2
import numpy as np
import mediapipe as mp
import os
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model

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
