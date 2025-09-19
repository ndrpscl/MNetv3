import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras import layers, models

NUM_CLASSES = 3

base_model = MobileNetV3Small(
    input_shape=(320, 320, 3),
    include_top=False,      
    weights="imagenet"      
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(320, 320, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.summary()
