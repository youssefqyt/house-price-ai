"""Paths and model schema shared by the command line workflows."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "tunisia_housing_500_synthetic.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "house_price_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
TARGET_COLUMN = "price_tnd"
ID_COLUMN = "id"
TEST_SIZE = 0.20
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "area_m2", "rooms", "bathrooms", "age", "floor", "latitude",
    "longitude", "distance_to_tunis_km", "garage", "garden", "pool",
    "beach_view", "mountain_view", "elevator", "furnished",
    "equipped_kitchen", "central_heating", "air_conditioning",
]
CATEGORICAL_FEATURES = [
    "governorate", "city", "neighborhood", "property_type", "state",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
