from tensorflow.keras.models import load_model

# Load model WITHOUT compiling (skip 'fbeta' metric)
emotion_model = load_model("custom_emotion_model.h5")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
