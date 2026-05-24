from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import numpy as np
import joblib

app = FastAPI()

model = joblib.load(
    "models/xgboost_propensity_model.pkl"
)

scaler = joblib.load(
    "models/propensity_scaler.pkl"
)

class CustomerData(BaseModel):

    recency: float
    frequency: float
    monetary: float

@app.get("/")
def home():

    return {
        "message": "Customer Analytics API Running"
    }

@app.post("/predict")
def predict(data: CustomerData):

    input_data = pd.DataFrame({
        'Recency': [data.recency],
        'Frequency': [data.frequency],
        'Monetary': [data.monetary]
    })

    input_log = np.log1p(input_data)

    input_scaled = scaler.transform(
        input_log
    )

    probability = model.predict_proba(
        input_scaled
    )[0][1]

    return {
        "purchase_probability": float(probability)
    }