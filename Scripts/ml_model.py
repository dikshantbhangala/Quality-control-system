import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv(r"V:\Quality Control\cleaned_defects.csv")


# 📌 Display available columns
print(f"📌 Available columns: {list(df.columns)}")

# Define expected features
expected_features = ['temperature', 'pressure', 'machine_speed', 'operator_experience']

# Check available features
available_features = [col for col in expected_features if col in df.columns]

# If expected features are missing, fallback to alternative numerical columns
if not available_features:
    print("⚠️ Warning: Expected features not found! Using alternative available numerical features.")
    alternative_features = ['severity', 'repair_cost']  # Modify based on dataset
    available_features = [col for col in alternative_features if col in df.columns]

# Ensure we have at least one valid feature
if not available_features:
    raise KeyError("❌ No valid feature columns found! Check dataset.")

# Extract features
X = df[available_features].copy()

# Convert categorical features to numerical using Label Encoding
label_encoders = {}  # Store encoders for future use in Streamlit app
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le  # Save encoder for Streamlit app

# Define target variable
target = 'defect_type'
if target in df.columns:
    y = LabelEncoder().fit_transform(df[target])
else:
    raise KeyError(f"❌ Target column '{target}' not found in dataset!")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical values (only for numeric columns)
numeric_cols = X.select_dtypes(include=['number']).columns
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2f}")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# ✅ Ensure models directory exists
models_dir = "models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# ✅ Save model, scaler, and label encoders
joblib.dump(model, os.path.join(models_dir, "defect_model.pkl"))
joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
joblib.dump(label_encoders, os.path.join(models_dir, "label_encoders.pkl"))

print("🎯 Model and preprocessing objects saved successfully!")
