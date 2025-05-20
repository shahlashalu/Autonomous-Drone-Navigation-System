# cnn_predict.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Paths
model_path = os.path.abspath("../data/best_model.h5")
img_path = os.path.abspath(r"C:\Users\hp\DroneNavVision\data\test\Fire\fi10.jpg")  # Replace with real path

# Load model
model = tf.keras.models.load_model(model_path)

# Load and preprocess image
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
pred = model.predict(img_array)
class_idx = np.argmax(pred[0])

# Get class labels from the training set
train_dir = os.path.abspath("../data/train")
class_labels = sorted(os.listdir(train_dir))
print(f"🔍 Predicted Class: {class_labels[class_idx]}")
