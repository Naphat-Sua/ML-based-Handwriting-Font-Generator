# src/Model-Training.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(X_train, y_train, X_test, y_test, model_save_path):
    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(y_train))
    
    model = build_cnn_model(input_shape, num_classes)
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    
    model.save(model_save_path)

if __name__ == "__main__":
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")
    
    train_model(X_train, y_train, X_test, y_test, "models/cnn_model.h5")
