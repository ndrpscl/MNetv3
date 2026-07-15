import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# 👇  Fix: register the function for loading
model = tf.keras.models.load_model(
    "best_model.keras",
    custom_objects={"preprocess_input": preprocess_input}
)

class_names = ["Butterhead", "Crisphead", "Looseleaf", "Oak", "Others", "RedLeaf", "Romaine"]  # adjust if reversed
test_image_path = "test_img/looseleaf1.jpg"

img = cv2.imread(test_image_path)
if img is None:
    raise FileNotFoundError(f"Image not found: {test_image_path}")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (320, 320))
img_array = np.expand_dims(img, axis=0)
img_array = preprocess_input(img_array)

pred = model.predict(img_array)
predicted = np.argmax(pred[0])
conf = np.max(pred[0])

print(f"Prediction: {class_names[predicted]}")
print(f"Confidence: {conf:.2f}")
