# main.py

import os
import joblib
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import numpy as np
import pandas as pd

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_NAME = "GradientBoosting_best_model.pkl"  # Choose best performing model
model = joblib.load(MODEL_DIR / MODEL_NAME)

# Static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(
    request: Request,
    Contract: str = Form(...),
    InternetService: str = Form(...),
    OnlineSecurity: str = Form(...),
    OnlineBackup: str = Form(...),
    DeviceProtection: str = Form(...),
    TechSupport: str = Form(...),
    StreamingTV: str = Form(...),
    PaperlessBilling: str = Form(...),
    PaymentMethod: str = Form(...),
    SeniorCitizen: int = Form(...),
    tenure: float = Form(...),
    MonthlyCharges: float = Form(...)
):
    TotalCharges = tenure * MonthlyCharges
    input_data = pd.DataFrame([{
        "Contract": Contract,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "SeniorCitizen": SeniorCitizen,
        "TotalCharges": TotalCharges
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": "Yes" if prediction == 1 else "No",
        "probability": f"{probability * 100:.2f}%"
    })

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
