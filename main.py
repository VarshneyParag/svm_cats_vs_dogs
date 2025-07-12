from flask import Flask, request, render_template
import os
import cv2
import numpy as np
import joblib

app = Flask(__name__)

# ✅ Safe model loader
def load_model():
    try:
        model = joblib.load("svm_model.pkl")
        print("✅ Model loaded successfully.")
        return model
    except Exception as e:
        print("❌ Error loading model:", e)
        return None

model = load_model()

# ✅ Image preprocessing
def preprocess_image(img_path):
    try:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (64, 64))
        return img.flatten().reshape(1, -1)
    except Exception as e:
        print("❌ Error preprocessing image:", e)
        return None

# ✅ Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return "Model not loaded. Please try again later."

    if 'image' not in request.files:
        return "No file uploaded"

    file = request.files['image']
    if file.filename == '':
        return "No file selected"

    # Ensure static folder exists
    if not os.path.exists("static"):
        os.makedirs("static")

    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    img_data = preprocess_image(filepath)
    if img_data is None:
        return "Failed to process image."

    prediction = model.predict(img_data)
    label = "Dog" if prediction[0] == 1 else "Cat"

    return render_template('result.html', label=label, image=file.filename)

if __name__ == "__main__":
    app.run(debug=True)
