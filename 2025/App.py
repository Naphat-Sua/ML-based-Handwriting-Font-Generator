# App.py
from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from src.font_generation import generate_font

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    model_path = "models/gan_model.h5"
    num_samples = 1
    generated_fonts = generate_font(model_path, num_samples)
    generated_fonts = (generated_fonts * 255).astype(np.uint8)
    generated_fonts = generated_fonts.reshape(28, 28)
    return render_template('result.html', font_image=generated_fonts)

if __name__ == "__main__":
    app.run(debug=True)
