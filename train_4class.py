import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.preprocessing import image_dataset_from_directory

# ============================================================
# 1. Load Datasets
# ============================================================
train_dir = "pr/train"
val_dir = "pr/val"

train_ds = image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="int",
    image_size=(320, 320),
    batch_size=32,
    shuffle=True
)

val_ds = image_dataset_from_directory(
    val_dir,
    labels="inferred",
    label_mode="int",
    image_size=(320, 320),
    batch_size=32,
    shuffle=False
)

# Optimize dataset loading (prefetch + cache)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ============================================================
# 2. Build Model (MobileNetV3 Small)
# ============================================================
# Use pretrained ImageNet weights to speed up convergence
base_model = MobileNetV3Small(
    input_shape=(320, 320, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze base model to preserve pretrained features
base_model.trainable = False

# Add custom classification head
model = models.Sequential([
    layers.Input(shape=(320, 320, 3)),
    layers.Lambda(preprocess_input),       # Apply same preprocessing as pretrained model
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(4, activation="softmax")  # 4 classes: Romaine, Oak, Butterhead, and Others
])

# ============================================================
# 3. Compile Model
# ============================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ============================================================
# 4. Train Model
# ============================================================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# ============================================================
# 5. Save Model
# ============================================================
model.save("4classifier.keras")

print("✅ Training complete. Model saved as mobilenetv3_romaine_classifier.h5")
