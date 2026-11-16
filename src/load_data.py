"""Dataset loading and fail-fast validation."""

import pandas as pd

from src.config import DATA_PATH, FEATURES, ID_COLUMN, TARGET_COLUMN


def load_data(path=DATA_PATH):
    """Load CSV and validate required fields and target values."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    required = [*FEATURES, TARGET_COLUMN]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Dataset contains no rows")
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError(f"Target column '{TARGET_COLUMN}' contains missing values")
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    if frame[TARGET_COLUMN].isna().any() or (frame[TARGET_COLUMN] <= 0).any():
        raise ValueError(f"Target column '{TARGET_COLUMN}' must contain positive numbers")
    if ID_COLUMN in frame and frame[ID_COLUMN].duplicated().any():
        raise ValueError(f"Identifier column '{ID_COLUMN}' contains duplicates")
    return frame


if __name__ == "__main__":
    data = load_data()
    print(f"Validated {len(data)} rows and {len(data.columns)} columns")
