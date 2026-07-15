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
    layers.Dense(7, activation="softmax")  # 7 classes: Romaine, Oak, Butterhead, and Others
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
# 6. Fine-Tuning Stage
# ============================================================

# First, see how many layers the base model has
print(f"Total layers in base model: {len(base_model.layers)}")

# Unfreeze the base model
base_model.trainable = True

# Freeze all layers EXCEPT the last 15
# This preserves the general features learned early in the network
# and only fine-tunes the higher-level features
fine_tune_at = len(base_model.layers) - 15

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile with a much lower learning rate
# Important: always recompile after changing trainable layers
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # 50x lower than before
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Add callbacks for fine-tuning
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,              # stops if no improvement for 5 epochs
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,              # halves the learning rate when plateau is detected
        patience=3,
        min_lr=1e-7
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_model.keras",
        monitor="val_accuracy",
        save_best_only=True      # only saves when val_accuracy improves
    )
]

# Continue training from where Stage 1 left off
fine_tune_epochs = 30
total_epochs = 20 + fine_tune_epochs  # initial epochs + fine-tune epochs

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1] + 1,  # continues from last epoch
    callbacks=callbacks
)

# Save the fine-tuned model
model.save("/models/zenithV2.keras")
print("✅ Fine-tuning complete. Model saved as zenithV2.keras")
