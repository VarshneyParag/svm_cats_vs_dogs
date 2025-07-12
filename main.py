import os
import zipfile
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ========== STEP 1: UNZIP DATASET ==========
def unzip_dataset(zip_file='dataset_small.zip', extract_dir='dataset_small'):
    if not os.path.exists(extract_dir):
        print("🔓 Unzipping dataset...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("✅ Dataset extracted to:", extract_dir)
    else:
        print("📁 Dataset already extracted.")

# Call unzip
unzip_dataset()

# ========== STEP 2: LOAD IMAGES ==========
def load_data(folder_path, img_size=64):
    X, y = [], []
    for file in os.listdir(folder_path):
        if file.endswith(".jpg"):
            label = 0 if "cat" in file.lower() else 1
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                X.append(img.flatten())
                y.append(label)
    return np.array(X), np.array(y)

# Define dataset path
dataset_path = "dataset_small/train"
print("📥 Loading image data...")
X, y = load_data(dataset_path)

# ========== STEP 3: TRAIN MODEL ==========
print("📊 Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🤖 Training SVM model...")
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# ========== STEP 4: SAVE MODEL ==========
joblib.dump(svm_model, "svm_model.pkl")
print("✅ Model saved as 'svm_model.pkl'")

# ========== STEP 5: EVALUATE ==========
print("🧪 Evaluating model...")
y_pred = svm_model.predict(X_test)
print("🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))
