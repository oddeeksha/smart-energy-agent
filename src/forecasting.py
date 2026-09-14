"""
P2 — Forecasting Model
Owns: this file. Contract: predict(model, input_df) -> df with a 'predicted' column.
DO NOT change this function's signature — P3, P4, P5, P7 all import it.

FORECAST HORIZON SPECIFICATION:
-------------------------------
Forecast Horizon: 1 hour ahead / t+1
Input features at timestamp t:
    - temperature
    - hour
    - day_of_week
    - month
    - is_holiday
    - lag_1   (historical demand from t-1 hour)
    - lag_24  (historical demand from t-24 hours)
    - lag_168 (historical demand from t-168 hours)

Target:
    target_demand = demand.shift(-1)  [demand at timestamp t+1]

Meaning:
    Using information strictly available at or prior to timestamp t,
    the model predicts electricity demand for the upcoming hour t+1.

NO DATA LEAKAGE:
----------------
- target_demand is never included in MODEL_FEATURE_COLUMNS.
- Same-row demand is never included in MODEL_FEATURE_COLUMNS.
- lag_1, lag_24, lag_168 represent strictly historical observations.
- The original 'demand' column is preserved unchanged for downstream modules (P3/P4).
- The shifted target (target_demand) is used strictly for model training and evaluation.
- Chronological splitting is preserved: test data is never used for training or tuning.

PREDICTION SCHEMA:
------------------
Input:
    input_df: DataFrame containing at least the columns in MODEL_FEATURE_COLUMNS.
              May contain metadata columns (e.g. 'timestamp') which will be preserved.
Output:
    A copy of input_df with all original columns preserved, plus a new column:
        'predicted': float (one-hour-ahead demand forecast for timestamp t+1 relative to t).
    input_df is not mutated in place.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.config import (
    MODEL_FEATURE_COLUMNS,
    MODEL_PATH,
    FEATURES_CSV_PATH,
    FEATURE_IMPORTANCE_PATH,
    METRICS_PATH,
    FORECAST_HORIZON,
    TARGET_DEMAND_COL,
)


def _create_model(**params) -> XGBRegressor:
    """Create an XGBoost regression model with default baseline parameters."""

    default_params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "objective": "reg:squarederror",
        "random_state": 42,
    }

    default_params.update(params)

    return XGBRegressor(**default_params)


def check_timestamp_continuity(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp"
) -> dict:
    """
    Validates timestamp continuity and calculates unique consecutive time differences.

    Parameters:
        df: DataFrame containing the timestamp column.
        timestamp_col: Name of the timestamp column.

    Returns:
        Dictionary containing continuity statistics:
        {
            "is_strictly_hourly": bool,
            "unique_gaps": list of gap strings,
            "gap_counts": dict mapping gap string to count,
            "total_transitions": int,
            "min_timestamp": pd.Timestamp,
            "max_timestamp": pd.Timestamp
        }
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in dataframe.")

    ts = pd.to_datetime(df[timestamp_col])
    ts_sorted = ts.sort_values().reset_index(drop=True)

    diffs = ts_sorted.diff().dropna()
    gap_counts = {str(k): int(v) for k, v in diffs.value_counts().items()}
    unique_gaps = list(gap_counts.keys())

    one_hour_str = str(pd.Timedelta(hours=1))
    is_strictly_hourly = len(unique_gaps) == 1 and unique_gaps[0] == one_hour_str

    return {
        "is_strictly_hourly": is_strictly_hourly,
        "unique_gaps": unique_gaps,
        "gap_counts": gap_counts,
        "total_transitions": len(diffs),
        "min_timestamp": ts_sorted.iloc[0] if len(ts_sorted) > 0 else None,
        "max_timestamp": ts_sorted.iloc[-1] if len(ts_sorted) > 0 else None,
    }


def prepare_forecasting_target(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    demand_col: str = "demand",
    target_col: str = TARGET_DEMAND_COL
) -> pd.DataFrame:
    """
    Prepares the one-hour-ahead forecasting target (t+1) from chronological demand.

    Given observations at timestamp t:
        target_demand[t] = demand[t+1]

    Pipeline steps:
    1. Copies input dataframe (does not mutate input).
    2. Validates presence of required timestamp and demand columns.
    3. Chronologically sorts by timestamp.
    4. Validates timestamp continuity (computes diffs, records gaps).
    5. Creates shifted target: target_demand = demand.shift(-1).
    6. Preserves original demand column unchanged (for downstream P3/P4).
    7. Drops the final row(s) where target_demand is NaN due to shifting.
    8. Returns the prepared dataframe.
    """
    if demand_col not in df.columns:
        raise ValueError(f"Demand column '{demand_col}' not found in dataframe.")

    out = df.copy()

    if timestamp_col in out.columns:
        out[timestamp_col] = pd.to_datetime(out[timestamp_col])
        out = out.sort_values(timestamp_col).reset_index(drop=True)
        # Check continuity
        check_timestamp_continuity(out, timestamp_col=timestamp_col)

    # Shift demand by -1 to create target at t+1
    out[target_col] = out[demand_col].shift(-1)

    # Drop trailing NaN row(s) where t+1 does not exist
    out = out.dropna(subset=[target_col]).reset_index(drop=True)

    return out


def train_model(train_df: pd.DataFrame) -> XGBRegressor:
    """
    Train the XGBoost one-hour-ahead forecasting model.

    Expects train_df to already contain:
    - MODEL_FEATURE_COLUMNS: input features available at timestamp t
    - 'target_demand': shifted electricity demand target at timestamp t+1

    If a split column is present:
    - train rows are used for model training
    - validation rows are used for hyperparameter selection
    - test rows are never used

    If no split column is present, all supplied rows are used for training.
    """

    required_columns = MODEL_FEATURE_COLUMNS + [TARGET_DEMAND_COL]

    missing = [
        column
        for column in required_columns
        if column not in train_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns for training: {missing}. "
            f"Ensure prepare_forecasting_target() was called prior to train_model()."
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
        train_rows = train_df.copy()
        val_rows = pd.DataFrame()

    X_train = train_rows[MODEL_FEATURE_COLUMNS]
    y_train = train_rows[TARGET_DEMAND_COL]

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

    # Default to the primary project configuration
    best_params = parameter_sets[1]

    if not val_rows.empty:
        X_val = val_rows[MODEL_FEATURE_COLUMNS]
        y_val = val_rows[TARGET_DEMAND_COL]

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
    Takes any dataframe with the required model feature columns.

    Forecast Horizon:
        1 hour ahead / t+1.
        The returned 'predicted' column corresponds to the forecasted electricity
        demand at timestamp t+1 relative to feature observations at timestamp t.

    Guarantees:
        - Returns a copy of input_df (does NOT mutate input_df in place).
        - Preserves all original columns from input_df.
        - Adds exactly one new column: 'predicted'.

    Parameters:
        model: Trained XGBRegressor forecasting model.
        input_df: DataFrame containing at least MODEL_FEATURE_COLUMNS.

    Returns:
        DataFrame: Copy of input_df with the added 'predicted' column.
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
    Calculates forecast accuracy metrics (MAPE, RMSE, MAE).

    Parameters:
        actual: Ground truth demand values (e.g., test_df['target_demand']).
        predicted: Model predicted demand values.

    Returns:
        {
            'mape': float (percentage, e.g. 1.88 for 1.88%),
            'rmse': float,
            'mae': float
        }
    """

    actual = pd.Series(actual).astype(float)
    predicted = pd.Series(predicted).astype(float)

    errors = actual - predicted

    # Avoid division by zero in MAPE
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
    Evaluates against target_demand (t+1) if present, otherwise demand.
    Test data is used only for final evaluation.
    """

    if "temperature" not in MODEL_FEATURE_COLUMNS:
        raise ValueError(
            "'temperature' is not a model feature."
        )

    target_col = TARGET_DEMAND_COL if TARGET_DEMAND_COL in test_df.columns else "demand"

    # ---------------------------------------------------------
    # WITH TEMPERATURE
    # ---------------------------------------------------------

    model_with_weather = train_model(train_df)

    predictions_with_weather = predict(
        model_with_weather,
        test_df
    )

    metrics_with_weather = evaluate(
        test_df[target_col],
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

    train_target_col = TARGET_DEMAND_COL if TARGET_DEMAND_COL in training_rows.columns else "demand"

    model_without_weather = _create_model()

    model_without_weather.fit(
        training_rows[features_without_temperature],
        training_rows[train_target_col],
        verbose=False
    )

    predictions_without_weather = model_without_weather.predict(
        test_df[features_without_temperature]
    )

    metrics_without_weather = evaluate(
        test_df[target_col],
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


def export_feature_importance(
    model: XGBRegressor,
    feature_names: list = MODEL_FEATURE_COLUMNS,
    path: str = FEATURE_IMPORTANCE_PATH
) -> pd.DataFrame:
    """
    Exports feature importances from a trained XGBoost model to a CSV artifact.

    Parameters:
        model: Trained XGBRegressor instance.
        feature_names: List of feature column names matching model inputs.
        path: Destination path for the feature importance CSV artifact.

    Returns:
        DataFrame containing 'feature' and 'importance' columns, sorted descending.
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Provided model does not have 'feature_importances_' attribute.")

    importance_df = pd.DataFrame({
        "feature": list(feature_names),
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    importance_df.to_csv(path, index=False)
    return importance_df


def save_model(
    model: XGBRegressor,
    path: str = MODEL_PATH
) -> None:
    """Save the trained model artifact."""

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    joblib.dump(model, path)


def save_metrics(
    metrics: dict,
    test_df: pd.DataFrame,
    path: str = METRICS_PATH
) -> dict:
    """
    Saves forecasting evaluation metrics and test metadata to a JSON artifact.
    """
    test_start = str(test_df["timestamp"].min())
    test_end = str(test_df["timestamp"].max())
    test_rows = int(len(test_df))

    payload = {
        "mape": float(metrics["mape"]),
        "rmse": float(metrics["rmse"]),
        "mae": float(metrics["mae"]),
        "test_rows": test_rows,
        "test_start": test_start,
        "test_end": test_end,
    }

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

    return payload


def load_model(
    path: str = MODEL_PATH
) -> XGBRegressor:
    """Load the saved model artifact."""

    return joblib.load(path)


if __name__ == "__main__":

    print("==================================================")
    print("P2 FORECASTING PIPELINE — ONE-HOUR-AHEAD (t+1)")
    print("==================================================")

    # ---------------------------------------------------------
    # 1. Load features.csv
    # ---------------------------------------------------------

    if not os.path.exists(FEATURES_CSV_PATH):
        raise FileNotFoundError(
            f"Features file not found: {FEATURES_CSV_PATH}"
        )

    print(f"Loading feature dataset: {FEATURES_CSV_PATH}")
    raw_df = pd.read_csv(FEATURES_CSV_PATH)
    print(f"Loaded {len(raw_df)} rows, {raw_df.shape[1]} columns.")

    # ---------------------------------------------------------
    # 2. Chronological sorting & Timestamp validation
    # ---------------------------------------------------------

    continuity_stats = check_timestamp_continuity(raw_df, timestamp_col="timestamp")
    print("\nTimestamp Continuity Analysis:")
    print(f"  Strictly hourly : {continuity_stats['is_strictly_hourly']}")
    print(f"  Total transitions: {continuity_stats['total_transitions']}")
    print(f"  Unique time gaps : {continuity_stats['unique_gaps']}")
    print(f"  Gap counts       : {continuity_stats['gap_counts']}")

    # ---------------------------------------------------------
    # 3. Explicit Target Preparation (target_demand = demand.shift(-1))
    # ---------------------------------------------------------

    df = prepare_forecasting_target(
        raw_df,
        timestamp_col="timestamp",
        demand_col="demand",
        target_col=TARGET_DEMAND_COL
    )

    print(f"\nPrepared target_demand column for horizon: {FORECAST_HORIZON}")
    print(f"  Rows before target shift : {len(raw_df)}")
    print(f"  Rows after target shift  : {len(df)} (1 trailing row with NaN target dropped)")
    print(f"  Original 'demand' preserved: {'demand' in df.columns}")

    # ---------------------------------------------------------
    # 4. Report actual Train / Validation / Test Date Ranges
    # ---------------------------------------------------------

    print("\nDataset Split Ranges (Chronological):")
    for s in ["train", "val", "test"]:
        sub = df[df["split"] == s]
        min_ts = sub["timestamp"].min()
        max_ts = sub["timestamp"].max()
        print(f"  {s.upper():<10}: {min_ts} -> {max_ts} ({len(sub):,} rows)")

    # ---------------------------------------------------------
    # 5. Extract Train/Val and Test Data
    # ---------------------------------------------------------

    train_val_df = df[
        df["split"].isin(["train", "val"])
    ].copy()

    test_df = df[
        df["split"] == "test"
    ].copy()

    if train_val_df.empty:
        raise ValueError("No train/validation rows found.")

    if test_df.empty:
        raise ValueError("No test rows found.")

    # ---------------------------------------------------------
    # 6. Train XGBoost Model on target_demand
    # ---------------------------------------------------------

    print("\nTraining XGBoost forecasting model...")
    model = train_model(train_val_df)
    print("Model training complete.")

    # ---------------------------------------------------------
    # 7. Predict Held-out Test Set
    # ---------------------------------------------------------

    print("Generating predictions on held-out test set...")
    test_predictions = predict(model, test_df)

    # ---------------------------------------------------------
    # 8. Calculate NEW Test Metrics (Held-out Test Data against target_demand)
    # ---------------------------------------------------------

    metrics = evaluate(
        test_df[TARGET_DEMAND_COL],
        test_predictions["predicted"]
    )

    print("\n==================================================")
    print("NEW P2 Forecasting Test Results (1-Hour Ahead / t+1)")
    print("==================================================")
    print(f"  Test MAPE : {metrics['mape']:.4f}%")
    print(f"  Test RMSE : {metrics['rmse']:.4f}")
    print(f"  Test MAE  : {metrics['mae']:.4f}")
    print("==================================================")

    # ---------------------------------------------------------
    # 9. Save Metrics Artifact & Model Artifact
    # ---------------------------------------------------------

    save_metrics(metrics, test_df, METRICS_PATH)
    print(f"Saved metrics artifact: {METRICS_PATH}")

    save_model(model, MODEL_PATH)
    print(f"Model artifact saved to: {MODEL_PATH}")

    # ---------------------------------------------------------
    # 10. Export Feature Importance Artifact
    # ---------------------------------------------------------

    importance_df = export_feature_importance(model, MODEL_FEATURE_COLUMNS, FEATURE_IMPORTANCE_PATH)
    print(f"Feature importance exported to: {FEATURE_IMPORTANCE_PATH}")
    print("\nFeature Importances:")
    for idx, row in importance_df.iterrows():
        print(f"  {row['feature']:<15} : {row['importance']:.6f}")