# src/Font-Generation.py
import numpy as np
from tensorflow.keras.models import load_model

def generate_font(model_path, num_samples):
    model = load_model(model_path)
    noise = np.random.normal(0, 1, (num_samples, 100))
    generated_images = model.predict(noise)
    return generated_images

if __name__ == "__main__":
    generated_fonts = generate_font("models/gan_model.h5", 10)
    for i, font in enumerate(generated_fonts):
        plt.imshow(font.reshape(28, 28), cmap='gray')
        plt.savefig(f"generated_font_{i}.png")
