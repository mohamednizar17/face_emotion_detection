Real-Time Emotion Detection System 😄😐😠

This project detects human emotions in real-time using a custom-trained CNN model (custom_emotion_model.h5), powered by OpenCV, MediaPipe, and TensorFlow/Keras. It supports:

📷 Image emotion detection

📹 Video file emotion detection

🎥 Real-time webcam emotion detection (local)

🌐 Streamlit web interface (image/video upload)



---

📁 Project Structure

emotion_detection/
├── app.py                 # Main OpenCV-based application
├── load_model.py          # Loads the custom emotion model
├── streamlit_app.py       # Optional: Streamlit-based web interface
├── custom_emotion_model.h5 # Your trained Keras model
├── requirements.txt
└── README.md


---

🚀 Features

Real-time face detection using MediaPipe.

Emotion prediction using your custom-trained CNN model.

Supports Image, Video File, and Live Webcam modes.

Optional Streamlit interface for modern web-based UI.

Auto-resizing of output windows.

Confidence percentage displayed on faces.



---

🛆 Installation

1. Clone Repository



git clone https://github.com/your-username/emotion_detection.git
cd emotion_detection

2. Install Dependencies
Ensure you're using TensorFlow 2.14 or compatible.



pip install -r requirements.txt

If TensorFlow conflicts:

pip install tensorflow==2.14 keras==2.14 numpy==1.23.5

3. Place Your Model
Ensure your trained model is named:



custom_emotion_model.h5

and placed inside the project folder.


---

▶️ Running Locally (OpenCV App)

python app.py

It will prompt:

1 - Upload and detect from Image file

2 - Upload and detect from Video file

3 - Start Live Webcam detection



---

🌐 Running Web Interface (Streamlit)

streamlit run app.py

Features:

Upload image/video directly via web UI.

View predictions online.

Webcam support (experimental via streamlit-webrtc).



---

📋 Requirements

TensorFlow 2.14.x

Keras 2.14.x

OpenCV

MediaPipe

Numpy

Streamlit (for web app)

streamlit-webrtc (optional, for webcam in web app)

Tkinter (for file dialogs in local app)



---

⚙️ Model Format

Ensure your .h5 model is trained for:

Input Shape: (48, 48, 1) (grayscale 48x48)

Output: Softmax probabilities over emotion classes


Example classes:

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


---

📷 Webcam in Colab? (No!)

Colab doesn't support live webcam. Use your local system for real-time webcam detection.


---

📄 License

MIT License.


---

🤝 Contribution

Feel free to fork, improve, or build upon this project.


