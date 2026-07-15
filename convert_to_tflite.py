import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

SOURCE_MODEL = "best_model.keras"
OUTPUT_MODEL = "models/zenithV2.tflite"

model = tf.keras.models.load_model(
    SOURCE_MODEL,
    custom_objects={"preprocess_input": preprocess_input}
)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(OUTPUT_MODEL, "wb") as f:
    f.write(tflite_model)

print(f"Saved {OUTPUT_MODEL} ({len(tflite_model) / 1024:.0f} KB)")
