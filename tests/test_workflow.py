"""Fast checks for data contracts and the end-to-end model workflow."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.config import FEATURES, TARGET_COLUMN
from src.load_data import load_data
from src.predict import DEMO_HOUSE, predict_house
from src.train import train_model


class DataValidationTests(unittest.TestCase):
    def test_repository_dataset_has_required_schema_and_positive_target(self):
        data = load_data()
        self.assertEqual(len(data), 500)
        self.assertTrue(set(FEATURES + [TARGET_COLUMN]).issubset(data.columns))
        self.assertTrue((data[TARGET_COLUMN] > 0).all())

    def test_rejects_missing_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            pd.DataFrame({TARGET_COLUMN: [100]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                load_data(path)


class WorkflowTests(unittest.TestCase):
    def test_training_and_saved_artifact_support_prediction(self):
        model, x_test, y_test = train_model()
        self.assertEqual(len(x_test), len(y_test))
        self.assertEqual(len(x_test.columns), len(FEATURES))
        estimate = predict_house(DEMO_HOUSE)
        self.assertGreater(estimate, 0)
        self.assertLess(estimate, 10_000_000)

    def test_prediction_rejects_incomplete_listing(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            predict_house({"area_m2": 100})


if __name__ == "__main__":
    unittest.main()
