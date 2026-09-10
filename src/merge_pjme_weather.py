import os
import pandas as pd
import numpy as np

def merge_pjme_weather(
    pjme_path="data/features/PJME_hourly_features.csv",
    weather_path="weather/PJME_weather_2002_2018.csv",
    output_features_path="data/features/PJME_hourly_features_weather.csv",
    output_report_path="reports/PJME_weather_merge_report.csv"
):
    print("==========================================")
    print("MERGING PJME FEATURES WITH HISTORICAL WEATHER")
    print("==========================================")

    # Ensure directories exist
    os.makedirs(os.path.dirname(output_features_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)

    # 1. Load PJME Features Dataset
    print(f"Loading PJME features dataset from: {pjme_path}")
    df_pjme = pd.read_csv(pjme_path)
    energy_rows = len(df_pjme)
    orig_cols = list(df_pjme.columns)
    print(f"  PJME Rows: {energy_rows}")
    print(f"  PJME Columns: {len(orig_cols)}")
    print(f"  PJME Datetime Range: {df_pjme['Datetime'].min()} to {df_pjme['Datetime'].max()}")

    # 2. Load Weather Dataset
    print(f"\nLoading Weather dataset from: {weather_path}")
    # Skip metadata rows (lines 1-3 contain latitude/longitude/timezone metadata)
    df_weather = pd.read_csv(weather_path, skiprows=3)
    weather_rows = len(df_weather)
    print(f"  Weather Raw Rows: {weather_rows}")
    print(f"  Weather Raw Columns: {list(df_weather.columns)}")
    print(f"  Weather Time Range: {df_weather['time'].min()} to {df_weather['time'].max()}")

    # 3. Clean and Standardize Weather Columns
    rename_map = {}
    for col in df_weather.columns:
        if 'temperature' in col:
            rename_map[col] = 'temperature_2m'
        elif 'relative_humidity' in col:
            rename_map[col] = 'relative_humidity_2m'
        elif 'precipitation' in col:
            rename_map[col] = 'precipitation'
        elif 'cloud_cover' in col:
            rename_map[col] = 'cloud_cover'
        elif 'wind_speed' in col:
            rename_map[col] = 'wind_speed_10m'

    df_weather = df_weather.rename(columns=rename_map)
    weather_features_added = ['temperature_2m', 'relative_humidity_2m', 'precipitation', 'cloud_cover', 'wind_speed_10m']

    # Select only time and target weather variables (exclude metadata like lat/lon/timezone)
    df_weather_sub = df_weather[['time'] + weather_features_added].copy()

    # Parse datetimes for safe alignment
    df_pjme['dt_key'] = pd.to_datetime(df_pjme['Datetime'])
    df_weather_sub['dt_key'] = pd.to_datetime(df_weather_sub['time'])

    # Drop raw 'time' string from weather dataframe to prevent duplicate time key
    df_weather_sub = df_weather_sub.drop(columns=['time'])

    # 4. Perform Left Join to preserve PJME source timeline & exact row count
    print("\nAligning energy and weather timestamps via Left Join...")
    df_merged = pd.merge(df_pjme, df_weather_sub, on='dt_key', how='left')

    # Clean up temporary merge key
    df_merged = df_merged.drop(columns=['dt_key'])
    df_pjme = df_pjme.drop(columns=['dt_key'])

    final_rows = len(df_merged)
    final_cols = list(df_merged.columns)

    # 5. Validation Checks
    missing_weather_sum = df_merged[weather_features_added].isna().sum().sum()
    dup_datetime_cnt = df_merged['Datetime'].duplicated().sum()

    row_count_passed = (final_rows == energy_rows)
    all_orig_cols_present = all(col in final_cols for col in orig_cols)
    all_weather_added = all(col in final_cols for col in weather_features_added)
    no_extra_duplicates = (dup_datetime_cnt == df_pjme['Datetime'].duplicated().sum())

    merge_status = "SUCCESS" if (row_count_passed and all_orig_cols_present and all_weather_added and no_extra_duplicates and missing_weather_sum == 0) else "FAILED"

    print(f"\nMerge Validation Status: {merge_status}")
    print(f"  Final Dataset Shape: {df_merged.shape}")
    print(f"  Original PJME Rows: {energy_rows} | Final Merged Rows: {final_rows}")
    print(f"  Original Columns Count: {len(orig_cols)} | Final Columns Count: {len(final_cols)}")
    print(f"  Weather Features Added ({len(weather_features_added)}): {weather_features_added}")
    print(f"  Missing Weather Values: {missing_weather_sum}")
    print(f"  Duplicate Datetime Count: {dup_datetime_cnt}")

    # 6. Save Merged Dataset
    df_merged.to_csv(output_features_path, index=False)
    print(f"\nSaved integrated dataset to: {output_features_path}")

    # 7. Save Merge Report
    report_data = [{
        'energy_rows': energy_rows,
        'weather_rows': weather_rows,
        'final_rows': final_rows,
        'original_columns': len(orig_cols),
        'final_columns': len(final_cols),
        'weather_columns_added': ", ".join(weather_features_added),
        'missing_weather_values': missing_weather_sum,
        'duplicate_datetime_count': dup_datetime_cnt,
        'merge_status': merge_status
    }]
    df_report = pd.DataFrame(report_data)
    df_report.to_csv(output_report_path, index=False)
    print(f"Saved merge report to: {output_report_path}")

if __name__ == "__main__":
    merge_pjme_weather()
