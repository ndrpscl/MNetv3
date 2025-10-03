import os
import cv2
import numpy as np
import albumentations as A
from sklearn.model_selection import train_test_split

input_dir = "un/Iceberg"
output_dir = "pr/Iceberg"
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

train_files, val_files = train_test_split(image_files, test_size=0.3, random_state=42)

augment = A.Compose([
    A.Rotate(limit=20, p=0.5),                   # Rotation
    A.HorizontalFlip(p=0.5),                     # Horizontal flip
    A.VerticalFlip(p=0.5),                       # Vertical flip
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3), # Noise
    A.Perspective(scale=(0.05, 0.1), p=0.3)      # Perspective
])

def process_and_save(image_list, save_dir):
    for idx, filename in enumerate(image_list, start=1):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Skipping {filename}, not a valid image")
            continue

        img = cv2.resize(img, (320, 320))

        augmented = augment(image=img)
        img = augmented['image']

        img = img.astype(np.float32) / 255.0

        img_to_save = (img * 255).astype(np.uint8)

        save_path = os.path.join(save_dir, f"{idx}.jpg")
        cv2.imwrite(save_path, img_to_save)

# Process training and validation sets
process_and_save(train_files, train_dir)
process_and_save(val_files, val_dir)

print(f"Processed {len(train_files)} training images and {len(val_files)} validation images.")