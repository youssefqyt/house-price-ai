"""Command line entry point for the Tunisia housing price workflow."""

import argparse
import json

from src.evaluate import evaluate_model
from src.predict import DEMO_HOUSE, predict_house
from src.train import train_model


def main():
    parser = argparse.ArgumentParser(description="Tunisia house price research workflow")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train", help="Train the model and save its artifact")
    commands.add_parser("evaluate", help="Train and evaluate against a fixed holdout")
    predict_parser = commands.add_parser("predict", help="Predict one listing from JSON")
    predict_parser.add_argument("--input", help="Path to JSON object; defaults to a demo listing")
    args = parser.parse_args()

    if args.command == "train":
        train_model()
        print("Model trained and saved.")
    elif args.command == "evaluate":
        model, x_test, y_test = train_model()
        metrics = evaluate_model(model, x_test, y_test)
        print(json.dumps(metrics, indent=2))
    else:
        if args.input:
            with open(args.input, encoding="utf-8") as listing_file:
                listing = json.load(listing_file)
            if not isinstance(listing, dict):
                parser.error("--input must contain a JSON object")
        else:
            listing = DEMO_HOUSE
        price = predict_house(listing)
        print(f"Estimated price: {price:,.0f} TND")


if __name__ == "__main__":
    main()
