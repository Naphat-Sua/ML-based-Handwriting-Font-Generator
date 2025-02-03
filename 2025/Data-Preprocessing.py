# src/Data-Preprocessing.py
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    image = image / 255.0
    return image

def preprocess_data(raw_data_dir, processed_data_dir):
    images = []
    labels = []
    
    for user_dir in os.listdir(raw_data_dir):
        user_path = os.path.join(raw_data_dir, user_dir)
        for sample in os.listdir(user_path):
            image_path = os.path.join(user_path, sample)
            image = preprocess_image(image_path)
            images.append(image)
            labels.append(user_dir)
    
    images = np.array(images)
    labels = np.array(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    np.save(os.path.join(processed_data_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(processed_data_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(processed_data_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(processed_data_dir, 'y_test.npy'), y_test)

if __name__ == "__main__":
    raw_data_dir = "data/raw/handwriting_samples/"
    processed_data_dir = "data/processed/"
    preprocess_data(raw_data_dir, processed_data_dir)
