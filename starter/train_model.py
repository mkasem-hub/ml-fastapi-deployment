# Script to train machine learning model.

import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

# Load the data
data = pd.read_csv("../data/census.csv")

# 🔧 Clean column names (remove leading/trailing whitespace)
data.columns = data.columns.str.strip()

# Split into train and test sets
train, test = train_test_split(data, test_size=0.20, random_state=42)

# Categorical features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process training data
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True,
)

# Process test data
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train model
model = train_model(X_train, y_train)

# Save model and encoders
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
joblib.dump(encoder, "model/encoder.pkl")
joblib.dump(lb, "model/lb.pkl")

# Evaluate model
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {fbeta:.3f}")