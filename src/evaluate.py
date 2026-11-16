"""Holdout evaluation with a machine-readable metrics artifact."""

import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import METRICS_PATH
from src.train import train_model


def evaluate_model(model=None, x_test=None, y_test=None):
    if model is None or x_test is None or y_test is None:
        model, x_test, y_test = train_model()
    predictions = model.predict(x_test)
    metrics = {
        "mae_tnd": float(mean_absolute_error(y_test, predictions)),
        "rmse_tnd": float(mean_squared_error(y_test, predictions) ** 0.5),
        "r2": float(r2_score(y_test, predictions)),
        "holdout_rows": int(len(y_test)),
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


if __name__ == "__main__":
    report = evaluate_model()
    print("Holdout evaluation (synthetic dataset)")
    print(f"MAE:  {report['mae_tnd']:,.0f} TND")
    print(f"RMSE: {report['rmse_tnd']:,.0f} TND")
    print(f"R²:   {report['r2']:.3f}")
