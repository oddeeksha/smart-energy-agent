"""
P1 — Data Acquisition & Feature Engineering
Owns: this file. Output contract: data/processed/features.csv with columns
defined in config.FEATURE_COLUMNS. Nobody downstream should need to touch this file.
"""
import pandas as pd
import holidays

from src.config import FEATURE_COLUMNS, RAW_DEMAND_PATH, RAW_WEATHER_PATH, FEATURES_PATH


def load_and_merge(demand_path: str, weather_path: str) -> pd.DataFrame:
    """Loads raw demand + weather, joins on timestamp. Returns merged, unfeatured df.

    TODO(P1): implement real loading. Watch for timezone mismatches between
    the two sources before joining — confirm both are in the same tz first.
    """
    demand_df = pd.read_csv(demand_path, parse_dates=["timestamp"])
    weather_df = pd.read_csv(weather_path, parse_dates=["timestamp"])

    # TODO(P1): confirm join keys/column names match your actual raw files
    merged = demand_df.merge(weather_df, on="timestamp", how="left")
    return merged


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds lag_1, lag_24, lag_168, hour, day_of_week, month, season,
    is_holiday. Drops rows with NaN from lagging.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    # TODO(P1): confirm this season mapping matches what P3 expects
    season_map = {12: "winter", 1: "winter", 2: "winter",
                  3: "spring", 4: "spring", 5: "spring",
                  6: "summer", 7: "summer", 8: "summer",
                  9: "fall", 10: "fall", 11: "fall"}
    df["season"] = df["month"].map(season_map)

    us_holidays = holidays.US()
    df["is_holiday"] = df["timestamp"].dt.date.apply(lambda d: d in us_holidays)

    df["lag_1"] = df["demand"].shift(1)
    df["lag_24"] = df["demand"].shift(24)
    df["lag_168"] = df["demand"].shift(168)

    df = df.dropna(subset=["lag_1", "lag_24", "lag_168"]).reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-based split. Returns (train_df, val_df, test_df).
    No shuffling. No overlap. TODO(P1): confirm split proportions with the team
    (e.g. 70/15/15 by time, chronological order preserved).
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    return train_df, val_df, test_df


def run_pipeline() -> pd.DataFrame:
    """Convenience entrypoint: load -> engineer -> split -> save. Run this to produce
    the final features.csv that everyone else reads from."""
    merged = load_and_merge(RAW_DEMAND_PATH, RAW_WEATHER_PATH)
    featured = engineer_features(merged)
    train_df, val_df, test_df = split_data(featured)
    full = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # sanity check against the agreed contract before saving
    missing_cols = set(FEATURE_COLUMNS) - set(full.columns)
    assert not missing_cols, f"Missing required columns: {missing_cols}"

    full.to_csv(FEATURES_PATH, index=False)
    return full


if __name__ == "__main__":
    run_pipeline()
    print(f"Saved features to {FEATURES_PATH}")
