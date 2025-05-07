import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from ml.data import process_data
from ml.model import inference

# Load model and encoders
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "model", "model.pkl")
encoder_path = os.path.join(BASE_DIR, "model", "encoder.pkl")
lb_path = os.path.join(BASE_DIR, "model", "lb.pkl")

model = joblib.load(model_path)
encoder = joblib.load(encoder_path)
lb = joblib.load(lb_path)

# Initialize app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API is working!"}

class CensusData(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

@app.post("/inference")
def run_inference(data: CensusData):
    input_df = pd.DataFrame([data.dict()])

    cat_features = [
        "workclass", "education", "marital_status", "occupation",
        "relationship", "race", "sex", "native_country"
    ]

    X, _, _, _ = process_data(
        input_df,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb
    )

    pred = inference(model, X)
    label = lb.inverse_transform(pred)[0]

    return {"prediction": label}