# Tunisia House Price Predictor

An offline machine learning project that estimates Tunisian residential listing prices in TND from property, location, and amenity features. It provides a reproducible training and holdout evaluation workflow, a serialized preprocessing/model pipeline, and single-listing predictions through a command line interface.

> **Data limitation:** the included 500-row dataset is synthetic. Scores and predictions are for demonstrating the engineering workflow; they are not evidence of real market accuracy and must not be used for property valuation or financial decisions.

## Capabilities

- Validates required columns, target values, and duplicate identifiers before training.
- Keeps preprocessing inside a scikit-learn pipeline; numeric values are imputed and scaled, categorical values are imputed and one-hot encoded.
- Excludes the row identifier from model inputs to avoid spurious signal.
- Uses a deterministic 80/20 holdout and Ridge regression baseline.
- Reports MAE, RMSE, R², and holdout row count as JSON under `models/metrics.json`.
- Predicts from a JSON listing and rejects missing or unexpected input fields.
- Runs locally or in a non-root Docker container; GitHub Actions checks tests and a training/evaluation run.

## Architecture

```text
CSV dataset -> schema validation -> fixed train/holdout split
            -> imputation + scaling / one-hot encoding -> Ridge model
            -> models/house_price_model.joblib + models/metrics.json
JSON listing -> field validation -> saved pipeline -> estimated price (TND)
```

The project is a batch/offline CLI. It has no database, user accounts, network API, or production market data integration. Those should only be added when a real serving and data governance requirement exists.

## Requirements and setup

Python 3.12 is used by CI. From a fresh checkout:

```bash
python -m venv .venv
# Activate the environment (Windows PowerShell): .venv\Scripts\Activate.ps1
# Activate the environment (macOS/Linux): source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Commands

```bash
python main.py train       # fit and save the model
python main.py evaluate    # fit, evaluate on held-out rows, and save metrics
python main.py predict     # predict the illustrative built-in listing
python main.py predict --input examples/listing.json
python -m unittest discover -s tests -v
```

The JSON file must contain all model input fields. See [`examples/listing.json`](examples/listing.json). Prediction requires a model artifact; run `train` or `evaluate` first. Generated model and metric artifacts are ignored by Git because they can be recreated from the versioned dataset and code.

## Dataset

`data/tunisia_housing_500_synthetic.csv` contains 500 generated examples. `data/data_dictionary.csv` documents the columns. `id` is retained for data integrity checks but excluded from model features. Missing feature values are imputed by the fitted training pipeline; missing, non-numeric, or non-positive target prices are rejected.

## Project layout

```text
main.py                 CLI entry point
src/config.py           Paths and feature schema
src/load_data.py        Dataset loading and validation
src/preprocessing.py    Leakage-safe transformations
src/train.py            Split, fit, and model persistence
src/evaluate.py         Holdout metrics and JSON report
src/predict.py          Listing validation and inference
data/                   Synthetic sample dataset and dictionary
tests/                  Workflow and contract tests
```

## Docker

Build and run the default evaluation workflow:

```bash
docker build -t tunisia-house-price-ai .
docker run --rm tunisia-house-price-ai
```

The default container command evaluates and writes artifacts to its writable container filesystem. To keep artifacts on the host, mount a writable `models` directory and run `docker run --rm -v "${PWD}/models:/app/models" tunisia-house-price-ai evaluate` (PowerShell users can replace `${PWD}` with `${PWD.Path}`).

## CI and security

GitHub Actions installs bounded dependencies, runs Bandit source checks and `pip-audit`, creates a CycloneDX dependency SBOM artifact, runs unit/workflow tests, trains and evaluates the model, and builds the container. The workflow uses read-only repository permissions. No credentials or external services are required. The image runs as an unprivileged user. Dependency ranges are bounded by major version; review and update them regularly as part of maintenance.

## Model lifecycle and current limits

The preprocessing and estimator are serialized together with joblib. The split seed and ratio are explicit in configuration; evaluation metrics are generated from the holdout split. There is no registry, experiment tracker, drift monitor, automated retraining, or model promotion because a single local synthetic dataset cannot support those workflows meaningfully. Do not load model artifacts from untrusted sources.

Before any real use, replace the synthetic dataset with licensed, representative, quality-checked transactions or listings; define geographic/time-based validation to prevent location or temporal leakage; compare candidate models and calibrate uncertainty; establish privacy and retention controls; and validate performance across regions and price bands. The current random holdout measures only performance on this synthetic sample.
