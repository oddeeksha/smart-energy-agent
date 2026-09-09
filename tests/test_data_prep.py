import pandas as pd
from src.data_prep import engineer_features, split_data


def _fake_raw_df(n=200):
    timestamps = pd.date_range("2023-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "timestamp": timestamps,
        "demand": [1000 + i for i in range(n)],
        "temperature": [50.0] * n,
    })


def test_engineer_features_no_nan():
    raw = _fake_raw_df()
    out = engineer_features(raw)
    assert out.isnull().sum().sum() == 0


def test_engineer_features_has_required_columns():
    raw = _fake_raw_df()
    out = engineer_features(raw)
    for col in ["hour", "day_of_week", "month", "season", "is_holiday",
                "lag_1", "lag_24", "lag_168"]:
        assert col in out.columns


def test_split_data_is_chronological_and_nonoverlapping():
    raw = _fake_raw_df(n=500)
    featured = engineer_features(raw)
    train_df, val_df, test_df = split_data(featured)

    assert train_df["timestamp"].max() <= val_df["timestamp"].min()
    assert val_df["timestamp"].max() <= test_df["timestamp"].min()
    assert set(train_df["split"]) == {"train"}
    assert set(val_df["split"]) == {"val"}
    assert set(test_df["split"]) == {"test"}


# Example input/output (for reference, not executed):
#
# Input (raw): timestamp, demand, temperature
#   2023-01-01 00:00, 1000, 32.1
#
# Output (features.csv row): timestamp, demand, temperature, hour, day_of_week,
#   month, season, is_holiday, lag_1, lag_24, lag_168, split
#   2023-01-08 00:00, 1168, 30.5, 0, 6, 1, "winter", False, 1167, 1144, 1000, "train"
