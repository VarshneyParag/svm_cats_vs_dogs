from flask import Flask, request, render_template
import os
import cv2
import numpy as np
import joblib

app = Flask(__name__)

# Load pre-trained model
try:
    model = joblib.load("svm_model.pkl")
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

def preprocess_image(img_path):
    try:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))
        return img.flatten().reshape(1, -1)
    except Exception as e:
        print("❌ Image preprocessing error:", e)
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "❌ Model not loaded. Please check logs."

    if 'image' not in request.files:
        return "❌ No file uploaded."

    file = request.files['image']
    if file.filename == '':
        return "❌ No image selected."

    filepath = os.path.join('static', file.filename)
    file.save(filepath)

    img_data = preprocess_image(filepath)
    if img_data is None:
        return "❌ Failed to process image."

    prediction = model.predict(img_data)[0]
    label = "Dog 🐶" if prediction == 1 else "Cat 🐱"

    return render_template('result.html', label=label, image=file.filename)

if __name__ == '__main__':
    app.run(debug=True)
