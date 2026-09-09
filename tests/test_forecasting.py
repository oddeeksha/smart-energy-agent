import pandas as pd
import numpy as np
from src.forecasting import train_model, predict, evaluate
from src.config import MODEL_FEATURE_COLUMNS


def _fake_train_df(n=300):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "temperature": rng.uniform(20, 90, n),
        "hour": rng.integers(0, 24, n),
        "day_of_week": rng.integers(0, 7, n),
        "month": rng.integers(1, 13, n),
        "is_holiday": rng.choice([True, False], n),
        "lag_1": rng.uniform(1000, 2000, n),
        "lag_24": rng.uniform(1000, 2000, n),
        "lag_168": rng.uniform(1000, 2000, n),
    })
    df["demand"] = df["lag_1"] * 1.01 + rng.normal(0, 10, n)
    return df


def test_predict_returns_predicted_column_same_length():
    train_df = _fake_train_df()
    model = train_model(train_df)
    out = predict(model, train_df)
    assert "predicted" in out.columns
    assert len(out) == len(train_df)


def test_predict_does_not_mutate_input():
    train_df = _fake_train_df()
    model = train_model(train_df)
    original_cols = list(train_df.columns)
    predict(model, train_df)
    assert list(train_df.columns) == original_cols  # unchanged


def test_evaluate_returns_expected_keys():
    actual = pd.Series([100.0, 200.0, 300.0])
    predicted = pd.Series([110.0, 190.0, 305.0])
    result = evaluate(actual, predicted)
    assert set(result.keys()) == {"mape", "rmse", "mae"}
    assert result["mape"] > 0


# Example input/output:
#
# predict(model, df_with_columns=MODEL_FEATURE_COLUMNS)
# ->  same df + 'predicted': float column, e.g. 41890.2
