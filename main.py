from flask import Flask, request, render_template
import cv2
import numpy as np
import joblib
import os

app = Flask(__name__)
model = joblib.load("svm_model.pkl")

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (64, 64))
    return img.flatten().reshape(1, -1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded"

    file = request.files['image']
    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    img_data = preprocess_image(filepath)
    prediction = model.predict(img_data)
    label = "Dog" if prediction[0] == 1 else "Cat"

    return render_template('result.html', label=label, image=file.filename)

if __name__ == "__main__":
    app.run(debug=True)
