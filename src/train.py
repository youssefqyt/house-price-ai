"""Train and persist the complete preprocessing plus regression pipeline."""

import joblib
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    CATEGORICAL_FEATURES, FEATURES, MODEL_DIR, MODEL_PATH, RANDOM_STATE,
    TARGET_COLUMN, TEST_SIZE,
)
from src.load_data import load_data
from src.preprocessing import create_preprocessor


def train_model(data_path=None):
    data = load_data(data_path) if data_path is not None else load_data()
    x = data[FEATURES]
    y = data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    pipeline = Pipeline([
        ("preprocessing", create_preprocessor(
            [name for name in FEATURES if name not in CATEGORICAL_FEATURES],
            CATEGORICAL_FEATURES,
        )),
        ("model", Ridge(alpha=1.0)),
    ])
    pipeline.fit(x_train, y_train)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline, x_test, y_test


if __name__ == "__main__":
    train_model()
    print(f"Model saved to {MODEL_PATH}")
