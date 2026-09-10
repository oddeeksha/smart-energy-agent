"""
P1 — Data Acquisition & Feature Engineering
Data preprocessing & feature engineering pipeline module.
Provides load_and_merge(), engineer_features(), split_data(), run_pipeline(),
process_pjme_data(), clean_data(), preprocess_data(), and load_data() function signatures.
"""

import os
import sys
import pandas as pd
import numpy as np
import holidays

# Add project root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.config as config


def load_and_merge(demand_path: str = config.RAW_DEMAND_PATH, weather_path: str = config.RAW_WEATHER_PATH) -> pd.DataFrame:
    """Loads raw demand + weather, joins on timestamp. Returns merged df."""
    if not os.path.exists(demand_path) or not os.path.exists(weather_path):
        if os.path.exists(config.PJME_RAW_PATH):
            return pd.read_csv(config.PJME_RAW_PATH)
    demand_df = pd.read_csv(demand_path, parse_dates=["timestamp"])
    weather_df = pd.read_csv(weather_path, parse_dates=["timestamp"])
    merged = demand_df.merge(weather_df, on="timestamp", how="left")
    return merged


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds lag_1, lag_24, lag_168, hour, day_of_week, month, season, is_holiday, etc.
    Drops rows with NaN from lagging.
    """
    df = df.copy()

    # Rename columns if needed
    rename_dict = {}
    if 'Datetime' in df.columns and config.TIMESTAMP_COL not in df.columns:
        rename_dict['Datetime'] = config.TIMESTAMP_COL
    if 'PJME_MW' in df.columns and config.TARGET_COL not in df.columns:
        rename_dict['PJME_MW'] = config.TARGET_COL
    if 'temperature_2m' in df.columns and 'temperature' not in df.columns:
        rename_dict['temperature_2m'] = 'temperature'
    if rename_dict:
        df = df.rename(columns=rename_dict)

    if config.TIMESTAMP_COL in df.columns:
        df[config.TIMESTAMP_COL] = pd.to_datetime(df[config.TIMESTAMP_COL])
        df = df.sort_values(config.TIMESTAMP_COL).reset_index(drop=True)
        dt = df[config.TIMESTAMP_COL]

        if 'hour' not in df.columns:
            df['hour'] = dt.dt.hour
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = dt.dt.dayofweek
        if 'month' not in df.columns:
            df['month'] = dt.dt.month

        season_map = {
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        }
        if 'season' not in df.columns:
            df['season'] = df['month'].map(season_map)

        if 'is_holiday' not in df.columns:
            min_y, max_y = dt.dt.year.min(), dt.dt.year.max()
            us_hols = holidays.US(years=range(min_y, max_y + 1))
            df['is_holiday'] = dt.dt.date.apply(lambda d: d in us_hols).astype(bool)

    if config.TARGET_COL in df.columns:
        y = df[config.TARGET_COL]
        if 'lag_1' not in df.columns:
            df['lag_1'] = y.shift(1)
        if 'lag_24' not in df.columns:
            df['lag_24'] = y.shift(24)
        if 'lag_168' not in df.columns:
            df['lag_168'] = y.shift(168)

    # Drop rows with NaN from lagging
    lag_cols = [c for c in ['lag_1', 'lag_24', 'lag_168'] if c in df.columns]
    if lag_cols:
        df = df.dropna(subset=lag_cols).reset_index(drop=True)

    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-based split. Returns (train_df, val_df, test_df). No shuffling."""
    df = df.sort_values(config.TIMESTAMP_COL).reset_index(drop=True)
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


def clean_data(df):
    """Cleans input dataframe by sorting timestamps and resetting index."""
    df = df.copy()
    if config.TIMESTAMP_COL in df.columns:
        df[config.TIMESTAMP_COL] = pd.to_datetime(df[config.TIMESTAMP_COL])
        df = df.sort_values(config.TIMESTAMP_COL).reset_index(drop=True)
    return df


def preprocess_data(df, train_split_demand=None, peak_percentile=config.PEAK_PERCENTILE):
    """Main preprocessing wrapper for input dataframe."""
    df = clean_data(df)
    df = engineer_features(df)
    return df


def load_data(filepath=config.FINAL_DATASET_PATH):
    """Loads preprocessed PJME dataset."""
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    elif os.path.exists(config.FEATURES_PATH):
        return pd.read_csv(config.FEATURES_PATH)
    elif os.path.exists(config.FEATURES_CSV_PATH):
        return pd.read_csv(config.FEATURES_CSV_PATH)
    else:
        raise FileNotFoundError(f"Dataset not found at {filepath}")


def process_pjme_data(
    input_path=config.PJME_RAW_PATH,
    output_path=config.FINAL_DATASET_PATH
):
    """
    Executes final P1 preprocessing pipeline for PJME energy + weather forecasting dataset.
    Preserves all 46 existing features, performs requested column renames, adds season,
    is_holiday, drops initial 168 rows, and adds chronological 70/15/15 train/val/test split flags.
    """
    print("==========================================")
    print("RUNNING P1 DATA PREPROCESSING PIPELINE")
    print("==========================================")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    print(f"Loaded input dataset: {input_path} (Shape: {df.shape})")

    # 1. Rename requested columns
    rename_dict = {
        'Datetime': config.TIMESTAMP_COL,
        'PJME_MW': config.TARGET_COL,
        'temperature_2m': 'temperature'
    }
    df = df.rename(columns=rename_dict)
    print("[OK] Renamed columns: Datetime -> timestamp, PJME_MW -> demand, temperature_2m -> temperature")

    # 2. Parse timestamps & Ensure chronological order
    df[config.TIMESTAMP_COL] = pd.to_datetime(df[config.TIMESTAMP_COL])
    df = df.sort_values(config.TIMESTAMP_COL).reset_index(drop=True)
    print("[OK] Timestamp parsed and chronologically sorted.")

    # 3. Add season column
    season_map = {
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'fall', 10: 'fall', 11: 'fall'
    }
    df['season'] = df['month'].map(season_map)
    print("[OK] Added 'season' column (winter, spring, summer, fall).")

    # 4. Add is_holiday column using US holidays
    min_year = df[config.TIMESTAMP_COL].dt.year.min()
    max_year = df[config.TIMESTAMP_COL].dt.year.max()
    us_holidays = holidays.US(years=range(min_year, max_year + 1))

    df['is_holiday'] = df[config.TIMESTAMP_COL].dt.date.apply(lambda d: d in us_holidays).astype(int)
    print(f"[OK] Added 'is_holiday' column using US holidays ({df['is_holiday'].sum()} total holiday hours flagged).")

    # 5. Drop initial 168 rows containing lag/rolling initialization NaNs
    initial_rows_before = len(df)
    df = df.iloc[168:].reset_index(drop=True)
    print(f"[OK] Dropped first 168 initialization NaN rows (Rows: {initial_rows_before} -> {len(df)}).")

    # 6. Add chronological 70% / 15% / 15% split
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    splits = np.empty(n, dtype=object)
    splits[:train_end] = 'train'
    splits[train_end:val_end] = 'val'
    splits[val_end:] = 'test'
    df[config.SPLIT_COL] = splits

    train_cnt = (df[config.SPLIT_COL] == 'train').sum()
    val_cnt = (df[config.SPLIT_COL] == 'val').sum()
    test_cnt = (df[config.SPLIT_COL] == 'test').sum()
    print(f"[OK] Added 'split' column: train={train_cnt} ({train_cnt/n:.1%}), val={val_cnt} ({val_cnt/n:.1%}), test={test_cnt} ({test_cnt/n:.1%})")

    # 7. Train-only Peak Threshold Calculation (Zero Leakage)
    train_demand = df.loc[df[config.SPLIT_COL] == 'train', config.TARGET_COL]
    train_peak_thresh = train_demand.quantile(config.PEAK_PERCENTILE)
    df['peak_threshold'] = train_peak_thresh
    df['is_peak'] = (df[config.TARGET_COL] > train_peak_thresh).astype(int)
    print(f"[OK] Computed train-only peak_threshold: {train_peak_thresh:.2f} MW (85th percentile of training demand).")

    # Format timestamp as string for output
    df[config.TIMESTAMP_COL] = df[config.TIMESTAMP_COL].dt.strftime('%Y-%m-%d %H:%M:%S')

    # 8. Reorder and Validate Final 49 Columns
    missing_cols = [c for c in config.FINAL_COLUMN_LIST if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Pipeline error! Missing expected final columns: {missing_cols}")

    df = df[config.FINAL_COLUMN_LIST]

    # Save reproducible dataset to all output paths expected by P1-P8
    output_paths = [
        output_path,
        config.FEATURES_CSV_PATH,
        config.FEATURES_PATH,
        "data/features/features.csv",
        "data/processed/features.csv"
    ]
    for target_p in output_paths:
        os.makedirs(os.path.dirname(target_p), exist_ok=True)
        import time
        saved = False
        for attempt in range(10):
            try:
                df.to_csv(target_p, index=False)
                saved = True
                break
            except PermissionError:
                time.sleep(0.5)
        if not saved:
            raise PermissionError(f"Could not write to {target_p} after 10 attempts due to OS file lock.")
        print(f"[OK] Saved 49-column dataset to: {target_p} (Shape: {df.shape})")

    return df


def run_pipeline() -> pd.DataFrame:
    """Convenience entrypoint expected by starter architecture."""
    if os.path.exists(config.PJME_RAW_PATH):
        return process_pjme_data()
    merged = load_and_merge(config.RAW_DEMAND_PATH, config.RAW_WEATHER_PATH)
    featured = engineer_features(merged)
    train_df, val_df, test_df = split_data(featured)
    full = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full.to_csv(config.FEATURES_PATH, index=False)
    return full


if __name__ == "__main__":
    process_pjme_data()
