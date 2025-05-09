import pandas as pd
from model.process import preprocess_data
from model.train_model import train_model
from model.metrics import compute_model_metrics
import joblib

# Load the cleaned data
data = pd.read_csv("data/census_clean.csv")

# Preprocess the data
X_train, X_test, y_train, y_test, encoder, lb = preprocess_data(data)

# Train the model
model = train_model(X_train, y_train)

# Save the model and encoders
joblib.dump(model, "model/model.pkl")
joblib.dump(encoder, "model/encoder.pkl")
joblib.dump(lb, "model/lb.pkl")

# Evaluate the model
preds = model.predict(X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Precision: {precision}, Recall: {recall}, F1: {fbeta}")
