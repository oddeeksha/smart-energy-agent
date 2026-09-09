"""
P2 — Forecasting Model
Owns: this file. Contract: predict(model, input_df) -> df with a 'predicted' column.
DO NOT change this function's signature — P3, P4, P5, P7 all import it.
"""
import pandas as pd
import joblib
from xgboost import XGBRegressor

from src.config import MODEL_FEATURE_COLUMNS, MODEL_PATH


def train_model(train_df: pd.DataFrame) -> XGBRegressor:
    """Trains on rows where split == 'train'. Returns fitted model.

    TODO(P2): tune hyperparameters using the VAL split only, never test.
    """
    X_train = train_df[MODEL_FEATURE_COLUMNS]
    y_train = train_df["demand"]

    model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05)
    model.fit(X_train, y_train)
    return model


def predict(model: XGBRegressor, input_df: pd.DataFrame) -> pd.DataFrame:
    """Takes any dataframe with the required feature columns.
    Returns input_df with an added 'predicted' column. Does not mutate input_df in place.
    """
    out = input_df.copy()
    X = out[MODEL_FEATURE_COLUMNS]
    out["predicted"] = model.predict(X)
    return out


def evaluate(actual: pd.Series, predicted: pd.Series) -> dict:
    """Returns {'mape': float, 'rmse': float, 'mae': float}.
    Caller is responsible for filtering to the correct split (test only, for
    final reported numbers) before calling this.
    """
    import numpy as np
    errors = actual - predicted
    mape = (errors.abs() / actual.abs()).mean() * 100
    rmse = (errors ** 2).mean() ** 0.5
    mae = errors.abs().mean()
    return {"mape": mape, "rmse": rmse, "mae": mae}


def evaluate_by_period(df: pd.DataFrame, actual_col: str, predicted_col: str,
                        peak_threshold_col: str) -> dict:
    """Returns {'peak_hour_mape': float, 'offpeak_hour_mape': float,
    'peak_hour_bias': float}  # mean(actual - predicted) for peak hours

    TODO(P2): peak_threshold_col should be a boolean column indicating whether
    that row's actual demand exceeded the seasonal threshold (from P3).
    """
    peak_rows = df[df[peak_threshold_col]]
    offpeak_rows = df[~df[peak_threshold_col]]

    peak_metrics = evaluate(peak_rows[actual_col], peak_rows[predicted_col])
    offpeak_metrics = evaluate(offpeak_rows[actual_col], offpeak_rows[predicted_col])
    peak_bias = (peak_rows[actual_col] - peak_rows[predicted_col]).mean()

    return {
        "peak_hour_mape": peak_metrics["mape"],
        "offpeak_hour_mape": offpeak_metrics["mape"],
        "peak_hour_bias": peak_bias,
    }


def run_weather_ablation(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """TODO(P2): train with and without 'temperature' in MODEL_FEATURE_COLUMNS,
    compare MAPE, report both. Implement by temporarily dropping the column
    and re-running train_model/predict/evaluate."""
    raise NotImplementedError("TODO(P2): implement the ablation comparison")


def save_model(model: XGBRegressor, path: str = MODEL_PATH) -> None:
    joblib.dump(model, path)


def load_model(path: str = MODEL_PATH) -> XGBRegressor:
    return joblib.load(path)


if __name__ == "__main__":
    # TODO(P2): wire this up to load features.csv, filter to split=='train',
    # train, evaluate on split=='test', save the model.
    pass
