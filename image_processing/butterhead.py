import os
import cv2
import numpy as np
import albumentations as A
from sklearn.model_selection import train_test_split

input_dir = "../un/Butterhead"
output_dir = "../pr"
train_dir = os.path.join(output_dir, "train/Butterhead")
val_dir = os.path.join(output_dir, "val/Butterhead")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

train_files, val_files = train_test_split(image_files, test_size=0.3, random_state=42)

augment = A.Compose([
    A.Rotate(limit=20, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.Perspective(scale=(0.05, 0.1), p=0.3)
])

NUM_AUGMENTATIONS = 10

def process_and_save(image_list, save_dir, is_train=False):
    counter = 1
    for filename in image_list:
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Skipping {filename}, not a valid image")
            continue

        img = cv2.resize(img, (320, 320))

        # Save original image first
        save_path = os.path.join(save_dir, f"{counter}.jpg")
        cv2.imwrite(save_path, img)
        counter += 1

        # Generate augmented copies for training only
        if is_train:
            for _ in range(NUM_AUGMENTATIONS):
                augmented = augment(image=img)
                aug_img = augmented['image']
                save_path = os.path.join(save_dir, f"{counter}.jpg")
                cv2.imwrite(save_path, aug_img)
                counter += 1

    if is_train:
        total = len(image_list) + (len(image_list) * NUM_AUGMENTATIONS)
        print(f"✅ Saved {total} training images to {save_dir}")
        print(f"   ({len(image_list)} original + {len(image_list) * NUM_AUGMENTATIONS} augmented)")
    else:
        print(f"✅ Saved {len(image_list)} validation images to {save_dir}")

# Process training and validation sets
process_and_save(train_files, train_dir, is_train=True)
process_and_save(val_files, val_dir, is_train=False)