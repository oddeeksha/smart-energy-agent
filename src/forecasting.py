"""
P2 — Forecasting Model
Owns: this file. Contract: predict(model, input_df) -> df with a 'predicted' column.
DO NOT change this function's signature — P3, P4, P5, P7 all import it.
"""

import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.config import MODEL_FEATURE_COLUMNS, MODEL_PATH, FEATURES_CSV_PATH


def _create_model(**params) -> XGBRegressor:
    """Create an XGBoost regression model."""

    default_params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "objective": "reg:squarederror",
        "random_state": 42,
    }

    default_params.update(params)

    return XGBRegressor(**default_params)


def train_model(train_df: pd.DataFrame) -> XGBRegressor:
    """
    Train the XGBoost forecasting model.

    If a split column is present:
    - train rows are used for training
    - validation rows are used for hyperparameter selection
    - test rows are never used

    If no split column is present, all supplied rows are used for training.
    """

    required_columns = MODEL_FEATURE_COLUMNS + ["demand"]

    missing = [
        column
        for column in required_columns
        if column not in train_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # ---------------------------------------------------------
    # Separate TRAIN and VALIDATION data
    # ---------------------------------------------------------

    if "split" in train_df.columns:

        train_rows = train_df[
            train_df["split"] == "train"
        ].copy()

        val_rows = train_df[
            train_df["split"] == "val"
        ].copy()

        if train_rows.empty:
            raise ValueError(
                "No rows found with split == 'train'."
            )

    else:
        # Keeps compatibility with the existing unit tests.
        train_rows = train_df.copy()
        val_rows = pd.DataFrame()

    X_train = train_rows[MODEL_FEATURE_COLUMNS]
    y_train = train_rows["demand"]

    # ---------------------------------------------------------
    # Hyperparameter tuning using VALIDATION data only
    # ---------------------------------------------------------

    parameter_sets = [
        {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
        },
        {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
        },
        {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.03,
        },
    ]

    best_params = parameter_sets[0]

    if not val_rows.empty:

        X_val = val_rows[MODEL_FEATURE_COLUMNS]
        y_val = val_rows["demand"]

        best_rmse = float("inf")

        for params in parameter_sets:

            model = _create_model(**params)

            model.fit(
                X_train,
                y_train,
                verbose=False
            )

            predictions = model.predict(X_val)

            rmse = np.sqrt(
                np.mean(
                    (y_val.to_numpy() - predictions) ** 2
                )
            )

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params

    # ---------------------------------------------------------
    # Train final model using TRAIN data only
    # ---------------------------------------------------------

    final_model = _create_model(**best_params)

    final_model.fit(
        X_train,
        y_train,
        verbose=False
    )

    return final_model


def predict(model: XGBRegressor, input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes any dataframe with the required feature columns.

    Returns input_df with an added 'predicted' column.
    Does not mutate input_df in place.
    """

    missing = [
        column
        for column in MODEL_FEATURE_COLUMNS
        if column not in input_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required prediction columns: {missing}"
        )

    out = input_df.copy()

    X = out[MODEL_FEATURE_COLUMNS]

    out["predicted"] = model.predict(X)

    return out


def evaluate(actual: pd.Series, predicted: pd.Series) -> dict:
    """
    Returns:
        {
            'mape': float,
            'rmse': float,
            'mae': float
        }
    """

    actual = pd.Series(actual).astype(float)
    predicted = pd.Series(predicted).astype(float)

    errors = actual - predicted

    # Avoid division by zero in MAPE.
    non_zero = actual != 0

    if non_zero.any():
        mape = (
            errors[non_zero].abs()
            / actual[non_zero].abs()
        ).mean() * 100
    else:
        mape = 0.0

    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    mae = errors.abs().mean()

    return {
        "mape": float(mape),
        "rmse": float(rmse),
        "mae": float(mae),
    }


def evaluate_by_period(
    df: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
    peak_threshold_col: str
) -> dict:
    """
    Evaluate forecasting performance for peak and off-peak hours.
    """

    if peak_threshold_col not in df.columns:
        raise ValueError(
            f"Column '{peak_threshold_col}' not found."
        )

    peak_rows = df[
        df[peak_threshold_col] == True
    ]

    offpeak_rows = df[
        df[peak_threshold_col] == False
    ]

    if not peak_rows.empty:
        peak_metrics = evaluate(
            peak_rows[actual_col],
            peak_rows[predicted_col]
        )

        peak_bias = (
            peak_rows[actual_col]
            - peak_rows[predicted_col]
        ).mean()
    else:
        peak_metrics = {
            "mape": 0.0,
            "rmse": 0.0,
            "mae": 0.0,
        }

        peak_bias = 0.0

    if not offpeak_rows.empty:
        offpeak_metrics = evaluate(
            offpeak_rows[actual_col],
            offpeak_rows[predicted_col]
        )
    else:
        offpeak_metrics = {
            "mape": 0.0,
            "rmse": 0.0,
            "mae": 0.0,
        }

    return {
        "peak_hour_mape": float(
            peak_metrics["mape"]
        ),
        "offpeak_hour_mape": float(
            offpeak_metrics["mape"]
        ),
        "peak_hour_bias": float(
            peak_bias
        ),
    }


def run_weather_ablation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> dict:
    """
    Compare forecasting performance with and without temperature.

    Test data is used only for final evaluation.
    """

    if "temperature" not in MODEL_FEATURE_COLUMNS:
        raise ValueError(
            "'temperature' is not a model feature."
        )

    # ---------------------------------------------------------
    # WITH TEMPERATURE
    # ---------------------------------------------------------

    model_with_weather = train_model(train_df)

    predictions_with_weather = predict(
        model_with_weather,
        test_df
    )

    metrics_with_weather = evaluate(
        test_df["demand"],
        predictions_with_weather["predicted"]
    )

    # ---------------------------------------------------------
    # WITHOUT TEMPERATURE
    # ---------------------------------------------------------

    features_without_temperature = [
        column
        for column in MODEL_FEATURE_COLUMNS
        if column != "temperature"
    ]

    if "split" in train_df.columns:
        training_rows = train_df[
            train_df["split"] == "train"
        ].copy()
    else:
        training_rows = train_df.copy()

    model_without_weather = _create_model()

    model_without_weather.fit(
        training_rows[features_without_temperature],
        training_rows["demand"],
        verbose=False
    )

    predictions_without_weather = model_without_weather.predict(
        test_df[features_without_temperature]
    )

    metrics_without_weather = evaluate(
        test_df["demand"],
        pd.Series(
            predictions_without_weather,
            index=test_df.index
        )
    )

    return {
        "with_weather_mape": metrics_with_weather["mape"],
        "without_weather_mape": metrics_without_weather["mape"],
        "mape_difference": (
            metrics_without_weather["mape"]
            - metrics_with_weather["mape"]
        ),
    }


def save_model(
    model: XGBRegressor,
    path: str = MODEL_PATH
) -> None:
    """Save the trained model."""

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    joblib.dump(model, path)


def load_model(
    path: str = MODEL_PATH
) -> XGBRegressor:
    """Load the saved model."""

    return joblib.load(path)


if __name__ == "__main__":

    # ---------------------------------------------------------
    # Load features.csv
    # ---------------------------------------------------------

    if not os.path.exists(FEATURES_CSV_PATH):
        raise FileNotFoundError(
            f"Features file not found: {FEATURES_CSV_PATH}"
        )

    df = pd.read_csv(FEATURES_CSV_PATH)

    # ---------------------------------------------------------
    # TRAIN + VALIDATION
    # ---------------------------------------------------------

    train_val_df = df[
        df["split"].isin(["train", "val"])
    ].copy()

    # ---------------------------------------------------------
    # TEST
    # ---------------------------------------------------------

    test_df = df[
        df["split"] == "test"
    ].copy()

    if train_val_df.empty:
        raise ValueError(
            "No train/validation rows found."
        )

    if test_df.empty:
        raise ValueError(
            "No test rows found."
        )

    # ---------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------

    model = train_model(train_val_df)

    # ---------------------------------------------------------
    # PREDICT TEST DATA
    # ---------------------------------------------------------

    test_predictions = predict(
        model,
        test_df
    )

    # ---------------------------------------------------------
    # EVALUATE
    # ---------------------------------------------------------

    metrics = evaluate(
        test_predictions["demand"],
        test_predictions["predicted"]
    )

    print("\nP2 Forecasting Test Results")
    print("---------------------------")
    print(f"MAPE : {metrics['mape']:.4f}%")
    print(f"RMSE : {metrics['rmse']:.4f}")
    print(f"MAE  : {metrics['mae']:.4f}")

    # ---------------------------------------------------------
    # SAVE MODEL
    # ---------------------------------------------------------

    save_model(model)

    print(f"\nModel saved to: {MODEL_PATH}")