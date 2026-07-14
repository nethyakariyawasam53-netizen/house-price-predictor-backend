# IMPORTS
               
from fastapi import FastAPI, HTTPException
from pathlib import Path
import pickle
from pydantic import BaseModel
import numpy as np

# INPUT SCHEMA
class HousePricePredictionInput(BaseModel):
    total_sqft: float
    bath: int  
    balcony: int
    bhk: int
    area_type: str
    location: str

# FASTAPI APP

app = FastAPI(
    title="House Price Prediction API",
    description="Predict Bengaluru house prices using ML model",
    version="1.0.0"
)

# LOAD MODEL

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "bengaluru_house_price_linear_regression_model.pickle"
)

try:
    with open(MODEL_PATH, "rb") as file:
        model_package = pickle.load(file)

    model = model_package["model"]
    feature_columns = model_package["feature_columns"]

except FileNotFoundError:
    raise RuntimeError(f"Model file not found at: {MODEL_PATH}")

except KeyError:
    raise RuntimeError(
        "Pickle file must contain 'model' and 'feature_columns'"
    )

except Exception as e:
    raise RuntimeError(f"Unexpected error: {e}")

# FEATURE GROUPS

NUMERICAL_TYPE_FEATURES = [
    "total_sqft",
    "bath",
    "balcony",
    "bhk"
]

AREA_TYPE_FEATURES = [
    "Built-up  Area",
    "Carpet  Area",
    "Plot  Area",
    "Super built-up  Area"
]

LOCATION_TYPE_FEATURES = [
    feature
    for feature in feature_columns
    if feature not in NUMERICAL_TYPE_FEATURES
    and feature not in AREA_TYPE_FEATURES
]

# CREATE INPUT VECTOR

def create_input_row(data: HousePricePredictionInput):

    input_row = np.zeros(len(feature_columns))

    # Numerical Features
    numerical_features = {
        "total_sqft": data.total_sqft,
        "bath": data.bath,
        "balcony": data.balcony,
        "bhk": data.bhk
    }

    for feature, value in numerical_features.items():
        if feature in feature_columns:
            idx = feature_columns.index(feature)
            input_row[idx] = value

    # Area Type Validation
    if data.area_type not in AREA_TYPE_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid area type: {data.area_type}"
        )

    if data.area_type in feature_columns:
        idx = feature_columns.index(data.area_type)
        input_row[idx] = 1

    # Location Validation
    if data.location not in LOCATION_TYPE_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid location: {data.location}"
        )

    if data.location in feature_columns:
        idx = feature_columns.index(data.location)
        input_row[idx] = 1

    return input_row

# ROUTES

@app.get("/")
def root():
    return {
        "message": "House Price Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.get("/model-info")
def model_info():
    return {
        "model_type": type(model).__name__,
        "no_of_feature_columns": len(feature_columns),
        "sample_feature_columns": feature_columns[:10]
    }


@app.get("/options")
def options():
    return {
        "area_type_features": AREA_TYPE_FEATURES,
        "location_type_features": LOCATION_TYPE_FEATURES
    }


@app.post("/predict")
def predict(data: HousePricePredictionInput):

    input_row = create_input_row(data)

    prediction = model.predict([input_row])[0]

    return {
        "prediction": round(float(prediction), 2),
        "input_data": {
            "total_sqft": data.total_sqft,
            "bath": data.bath,
            "balcony": data.balcony,
            "bhk": data.bhk,
            "area_type": data.area_type,
            "location": data.location
        }
    }