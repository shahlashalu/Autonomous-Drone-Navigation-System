import os
import shutil
import random

# Set your dataset path here
base_dir = r"C:\Users\hp\DroneNavVision\data\dataset"
train_dir = r"C:\Users\hp\DroneNavVision\data\train"
test_dir = r"C:\Users\hp\DroneNavVision\data\test"

# Set your train/test split ratio
split_ratio = 0.8

# Make sure train and test directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Iterate over each class folder in the dataset
for class_name in os.listdir(base_dir):
    class_path = os.path.join(base_dir, class_name)
    
    if os.path.isdir(class_path):
        images = [img for img in os.listdir(class_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)

        split_index = int(len(images) * split_ratio)
        train_images = images[:split_index]
        test_images = images[split_index:]

        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)

        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # Copy training images
        for img in train_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_class_dir, img))

        # Copy testing images
        for img in test_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(test_class_dir, img))

print("Dataset successfully split into training and testing folders.")
