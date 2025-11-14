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
