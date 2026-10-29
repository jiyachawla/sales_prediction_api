from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Create FastAPI application
app = FastAPI(title="Sales Prediction API")

# Load the trained ML model
model = joblib.load("model.pkl")


# Define input format
class SalesInput(BaseModel):
    budget: float


# Home API
@app.get("/")
def home():
    return {
        "message": "Sales Prediction API is working"
    }


# Prediction API
@app.post("/predict")
def predict_sales(data: SalesInput):
    prediction = model.predict([[data.budget]])

    return {
        "AdvertisingBudget": data.budget,
        "PredictedSales": round(float(prediction[0]), 2)
    }