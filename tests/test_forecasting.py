import os
import tempfile
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

from src.forecasting import (
    train_model,
    predict,
    evaluate,
    prepare_forecasting_target,
    check_timestamp_continuity,
    export_feature_importance,
)
from src.config import MODEL_FEATURE_COLUMNS, TARGET_DEMAND_COL


def _fake_train_df(n=300):
    """Creates synthetic training data with valid chronological timestamps and target_demand."""
    rng = np.random.default_rng(0)
    timestamps = pd.date_range("2024-01-01 00:00:00", periods=n, freq="h")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "temperature": rng.uniform(20, 90, n),
        "hour": [ts.hour for ts in timestamps],
        "day_of_week": [ts.dayofweek for ts in timestamps],
        "month": [ts.month for ts in timestamps],
        "is_holiday": rng.choice([True, False], n),
        "lag_1": rng.uniform(1000, 2000, n),
        "lag_24": rng.uniform(1000, 2000, n),
        "lag_168": rng.uniform(1000, 2000, n),
    })
    df["demand"] = df["lag_1"] * 1.01 + rng.normal(0, 10, n)
    return prepare_forecasting_target(df)


# ---------------------------------------------------------------------------
# 1. Target Shifting Test (demand = [100, 200, 300] -> target_demand = [200, 300])
# ---------------------------------------------------------------------------
def test_target_shifting():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01 10:00:00", "2024-01-01 11:00:00", "2024-01-01 12:00:00"]),
        "demand": [100.0, 200.0, 300.0],
    })
    result = prepare_forecasting_target(df)
    assert len(result) == 2
    assert result["target_demand"].tolist() == [200.0, 300.0]


# ---------------------------------------------------------------------------
# 2. Original demand remains unchanged
# ---------------------------------------------------------------------------
def test_original_demand_remains_unchanged():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01 10:00:00", "2024-01-01 11:00:00", "2024-01-01 12:00:00"]),
        "demand": [100.0, 200.0, 300.0],
    })
    result = prepare_forecasting_target(df)
    assert result["demand"].tolist() == [100.0, 200.0]


# ---------------------------------------------------------------------------
# 3. demand is not in MODEL_FEATURE_COLUMNS
# ---------------------------------------------------------------------------
def test_demand_not_in_model_feature_columns():
    assert "demand" not in MODEL_FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# 4. target_demand is not in MODEL_FEATURE_COLUMNS
# ---------------------------------------------------------------------------
def test_target_demand_not_in_model_feature_columns():
    assert "target_demand" not in MODEL_FEATURE_COLUMNS
    assert TARGET_DEMAND_COL not in MODEL_FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# 5. predict() returns predicted column of same length
# ---------------------------------------------------------------------------
def test_predict_returns_predicted_column_same_length():
    train_df = _fake_train_df()
    model = train_model(train_df)
    out = predict(model, train_df)
    assert "predicted" in out.columns
    assert len(out) == len(train_df)


# ---------------------------------------------------------------------------
# 6. predict() does not mutate input
# ---------------------------------------------------------------------------
def test_predict_does_not_mutate_input():
    train_df = _fake_train_df()
    model = train_model(train_df)
    original_cols = list(train_df.columns)
    original_vals = train_df.copy().to_numpy()
    predict(model, train_df)
    assert list(train_df.columns) == original_cols  # unchanged
    np.testing.assert_array_equal(train_df.to_numpy(), original_vals)


# ---------------------------------------------------------------------------
# 7. train_model() trains correctly when target_demand exists (and raises if missing)
# ---------------------------------------------------------------------------
def test_train_model_trains_correctly_when_target_demand_exists():
    train_df = _fake_train_df()
    model = train_model(train_df)
    assert isinstance(model, XGBRegressor)

    # Verify train_model raises if target_demand is missing
    df_missing = train_df.drop(columns=[TARGET_DEMAND_COL])
    try:
        train_model(df_missing)
        assert False, "Should have raised ValueError for missing target_demand"
    except ValueError as e:
        assert "target_demand" in str(e)


# ---------------------------------------------------------------------------
# 8. Timestamp ordering is handled correctly
# ---------------------------------------------------------------------------
def test_timestamp_ordering_handled_correctly():
    # Pass timestamps out of order
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01 12:00:00", "2024-01-01 10:00:00", "2024-01-01 11:00:00"]),
        "demand": [300.0, 100.0, 200.0],
    })
    result = prepare_forecasting_target(df)
    # Sorted order should be 10:00 (demand 100 -> target 200), 11:00 (demand 200 -> target 300)
    assert result["timestamp"].iloc[0] == pd.Timestamp("2024-01-01 10:00:00")
    assert result["timestamp"].iloc[1] == pd.Timestamp("2024-01-01 11:00:00")
    assert result["target_demand"].tolist() == [200.0, 300.0]


# ---------------------------------------------------------------------------
# 9. Timestamp gap validation works correctly
# ---------------------------------------------------------------------------
def test_timestamp_gap_validation():
    # Strictly hourly dataframe
    hourly_df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="h")
    })
    stats = check_timestamp_continuity(hourly_df)
    assert stats["is_strictly_hourly"] is True
    assert stats["total_transitions"] == 4

    # Gapped dataframe (skip one hour)
    gapped_df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01 10:00:00", "2024-01-01 11:00:00", "2024-01-01 13:00:00"])
    })
    gapped_stats = check_timestamp_continuity(gapped_df)
    assert gapped_stats["is_strictly_hourly"] is False
    assert str(pd.Timedelta(hours=2)) in gapped_stats["unique_gaps"]


# ---------------------------------------------------------------------------
# 10. Feature importance export contains feature and importance columns
# ---------------------------------------------------------------------------
def test_feature_importance_export_schema():
    train_df = _fake_train_df()
    model = train_model(train_df)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_fi.csv")
        df_fi = export_feature_importance(model, MODEL_FEATURE_COLUMNS, out_path)
        assert os.path.exists(out_path)
        assert list(df_fi.columns) == ["feature", "importance"]
        assert len(df_fi) == len(MODEL_FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# 11. Feature importance values are non-negative
# ---------------------------------------------------------------------------
def test_feature_importance_values_non_negative():
    train_df = _fake_train_df()
    model = train_model(train_df)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_fi.csv")
        df_fi = export_feature_importance(model, MODEL_FEATURE_COLUMNS, out_path)
        assert (df_fi["importance"] >= 0).all()
        # Sum of importances should be approximately 1.0
        assert np.isclose(df_fi["importance"].sum(), 1.0, atol=1e-3)


# ---------------------------------------------------------------------------
# 12. No future target is included as a model feature
# ---------------------------------------------------------------------------
def test_no_future_target_as_model_feature():
    forbidden_features = {"demand", "target_demand", "future_demand", "target"}
    for f in MODEL_FEATURE_COLUMNS:
        assert f not in forbidden_features
        # Ensure lags are strictly historical (positive lag means shifted backward in time)
        if f.startswith("lag_"):
            lag_hours = int(f.split("_")[1])
            assert lag_hours > 0, f"Lag feature {f} must be strictly positive (historical)."


# ---------------------------------------------------------------------------
# 13. Evaluate returns expected keys and valid metrics
# ---------------------------------------------------------------------------
def test_evaluate_returns_expected_keys():
    actual = pd.Series([100.0, 200.0, 300.0])
    predicted = pd.Series([110.0, 190.0, 305.0])
    result = evaluate(actual, predicted)
    assert set(result.keys()) == {"mape", "rmse", "mae"}
    assert result["mape"] > 0
    assert result["rmse"] > 0
    assert result["mae"] > 0
