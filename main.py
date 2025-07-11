import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load images and labels
def load_data(folder_path, img_size=64):
    X, y = [], []
    for file in os.listdir(folder_path):
        if file.endswith(".jpg"):
            label = 0 if "cat" in file else 1
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_size, img_size))
            X.append(img.flatten())
            y.append(label)
    return np.array(X), np.array(y)

# Set paths and parameters
DATASET_PATH = "dataset/train"
IMAGE_SIZE = 64

print("Loading data...")
X, y = load_data(DATASET_PATH, IMAGE_SIZE)

print(f"Data loaded: {X.shape[0]} samples")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
print("Training SVM...")
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Save model
joblib.dump(svm, "svm_model.pkl")
print("Model saved as svm_model.pkl")

# Evaluate
y_pred = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))
