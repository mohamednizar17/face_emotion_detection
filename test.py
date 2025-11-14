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
