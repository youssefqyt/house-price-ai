"""Prediction from an explicitly supplied listing or a useful demo listing."""

import joblib
import pandas as pd

from src.config import FEATURES, MODEL_PATH

DEMO_HOUSE = {
    "area_m2": 140, "rooms": 3, "bathrooms": 2, "age": 5, "floor": 2,
    "latitude": 36.8065, "longitude": 10.1815, "distance_to_tunis_km": 0,
    "garage": 1, "garden": 0, "pool": 0, "beach_view": 0,
    "mountain_view": 0, "elevator": 1, "furnished": 0,
    "equipped_kitchen": 1, "central_heating": 1, "air_conditioning": 1,
    "governorate": "Tunis", "city": "Tunis", "neighborhood": "Centre Ville",
    "property_type": "Apartment", "state": "good",
}


def predict_house(features=None, model_path=MODEL_PATH):
    values = dict(DEMO_HOUSE if features is None else features)
    missing = sorted(set(FEATURES) - set(values))
    extra = sorted(set(values) - set(FEATURES))
    if missing or extra:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("unexpected: " + ", ".join(extra))
        raise ValueError("Invalid listing fields (" + "; ".join(details) + ")")
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found at {model_path}; run 'python main.py train' first")
    model = joblib.load(model_path)
    prediction = float(model.predict(pd.DataFrame([values], columns=FEATURES))[0])
    return max(0.0, prediction)


if __name__ == "__main__":
    print(f"Estimated demo price: {predict_house():,.0f} TND")
