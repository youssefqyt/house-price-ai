import matplotlib.pyplot as plt

from src.train import train_model


def plot_predictions():

    model, X_test, y_test = train_model()

    predictions = model.predict(
        X_test
    )


    plt.figure(
        figsize=(8, 6)
    )


    plt.scatter(
        y_test,
        predictions,
        alpha=0.7
    )


    # Perfect prediction line

    minimum = min(
        y_test.min(),
        predictions.min()
    )

    maximum = max(
        y_test.max(),
        predictions.max()
    )


    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )


    plt.xlabel(
        "Actual Price (TND)"
    )

    plt.ylabel(
        "Predicted Price (TND)"
    )

    plt.title(
        "Actual vs Predicted House Prices"
    )


    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    plot_predictions()