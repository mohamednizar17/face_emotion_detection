from tensorflow.keras.models import load_model

# Try loading the model without compiling
model = load_model("custom_emotion_model.h5", compile=False)

# Print model structure
model.summary()
