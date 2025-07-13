import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# === Step 1: Load the dataset ===
try:
    data = pd.read_csv("quiz_dataset.csv")
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print("❌ Error: quiz_dataset.csv not found. Please make sure it exists in your project folder.")
    exit()

# === Step 2: Encode the personality labels ===
label_encoder = LabelEncoder()
data["Personality"] = label_encoder.fit_transform(data["Personality"])

# === Step 3: Prepare input features and target labels ===
X = data[["Q1", "Q2", "Q3", "Q4", "Q5"]]
y = data["Personality"]

# === Step 4: Train the Decision Tree model ===
model = DecisionTreeClassifier()
model.fit(X, y)

# === Step 5: Save the trained model and label encoder ===
joblib.dump(model, "quiz_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ Model training complete.")
print("📦 Saved:")
print(" - quiz_model.pkl (trained ML model)")
print(" - label_encoder.pkl (label converter)")
