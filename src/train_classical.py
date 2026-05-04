import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from src.preprocessing import extract_features

def train_rf():
    # 1. Setup paths
    train_path = r"C:\Users\Jermagne\forgery-detect\data\train"
    
    features = []
    labels = []
    
    print("Extracting features for Classical ML...")
    
    # 2. Extract features from training images
    # For Day 2, we assume images are labeled by folder or filename
    for file in os.listdir(train_path):
        if file.endswith(('.png', '.jpg')):
            img_path = os.path.join(train_path, file)
            feat = extract_features(img_path)
            features.append(feat)
            
            # Simple label logic: if 'forged' is in name, label 1, else 0
            labels.append(1 if "forged" in file.lower() else 0)
            
    X = np.array(features)
    y = np.array(labels)

    # 3. Train the Model
    # If your dataset is very small, we use a basic RF
    print(f"Training RandomForest on {len(X)} samples...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    # 4. Save the Deliverable
    os.makedirs('../models', exist_ok=True)
    joblib.dump(clf, '../models/random_forest.joblib')
    print("Success: Saved models/random_forest.joblib")

if __name__ == "__main__":
    train_rf()