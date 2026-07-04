from pathlib import Path
import pickle
import sys
import numpy as np


MODEL_PATH = Path("app/models/bengaluru_house_price_linear_regression_model.pickle")
with open(MODEL_PATH, "rb") as file:
    model_package = pickle.load(file)

    model = model_package["model"]
    feature_columns = model_package["feature_columns"]

print("Has predict:", hasattr(model, "predict"))
print("Model type:", type(model).__name__)
print("Features expected:", model.n_features_in_)

dummy_input = np.zeros(len(feature_columns))
prediction_result = model.predict([dummy_input])

print("Prediction result:", prediction_result)
print("Prediction result type:", type(prediction_result))
print("Prediction result shape:", prediction_result.shape)
print("First prediction:", prediction_result[0])