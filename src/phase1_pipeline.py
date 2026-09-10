import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure directories exist
DIRS = [
    'data/raw',
    'data/processed',
    'data/features',
    'reports',
    'visualizations'
]
for d in DIRS:
    os.makedirs(d, exist_ok=True)

# List of target CSV files
CSV_FILES = [
    'AEP_hourly.csv',
    'COMED_hourly.csv',
    'DAYTON_hourly.csv',
    'DEOK_hourly.csv',
    'DOM_hourly.csv',
    'DUQ_hourly.csv',
    'EKPC_hourly.csv',
    'FE_hourly.csv',
    'NI_hourly.csv',
    'PJME_hourly.csv',
    'PJMW_hourly.csv',
    'PJM_Load_hourly.csv',
    'pjm_hourly_est.csv'
]

print(f"Found {len(CSV_FILES)} CSV files to process.")

summary_records = []
quality_records = []
dst_records = []

# Matplotlib styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 150

for fname in sorted(CSV_FILES):
    print(f"\n==========================================")
    print(f"Processing Dataset: {fname}")
    print(f"==========================================")

    filepath = fname
    df_raw = pd.read_csv(filepath)

    dataset_name = os.path.splitext(fname)[0]
    num_rows_raw = len(df_raw)
    num_cols = len(df_raw.columns)
    col_names = list(df_raw.columns)
    dtypes_dict = {c: str(df_raw[c].dtype) for c in col_names}

    # Task 1 & 2: Inspection & Data Quality Audit
    duplicate_rows = df_raw.duplicated().sum()

    # Datetime handling
    raw_dt = df_raw['Datetime']
    parsed_dt = pd.to_datetime(raw_dt, errors='coerce')
    invalid_dates_count = parsed_dt.isna().sum()

    duplicate_timestamps_count = parsed_dt.duplicated().sum()

    # Duplicate timestamp details
    dup_mask = parsed_dt.duplicated(keep=False)
    dup_df = df_raw[dup_mask].copy()
    if not dup_df.empty:
        print(f"  [Notice] Found {duplicate_timestamps_count} duplicate Datetime observations.")
        dup_df['parsed_dt'] = pd.to_datetime(dup_df['Datetime'], errors='coerce')
        dst_months = dup_df['parsed_dt'].dt.month.isin([3, 10, 11]).sum()
        print(f"  [DST Audit] {dst_months} of {len(dup_df)} duplicate timestamps occur in DST transition months (March/Oct/Nov).")
        dst_records.append({
            'filename': fname,
            'duplicate_count': duplicate_timestamps_count,
            'dst_transition_count': dst_months,
            'notes': 'Matches Autumn DST Fallback (2:00 AM clock turn back) or duplicate logging.'
        })

    # Process dataset
    df_clean = df_raw.copy()
    df_clean['Datetime'] = pd.to_datetime(df_clean['Datetime'], errors='coerce')
    df_clean = df_clean.sort_values('Datetime').reset_index(drop=True)

    min_dt = df_clean['Datetime'].min()
    max_dt = df_clean['Datetime'].max()

    # Identify energy columns (all columns except Datetime)
    energy_cols = [c for c in col_names if c != 'Datetime']

    for e_col in energy_cols:
        series = df_clean[e_col]
        missing_cnt = series.isna().sum()
        missing_pct = (missing_cnt / num_rows_raw) * 100.0
        zero_cnt = (series == 0).sum()
        neg_cnt = (series < 0).sum()

        min_val = series.min()
        max_val = series.max()
        mean_val = series.mean()
        median_val = series.median()

        summary_records.append({
            'filename': fname,
            'dataset_name': dataset_name,
            'target_column': e_col,
            'total_rows': num_rows_raw,
            'total_cols': num_cols,
            'column_names': ", ".join(col_names),
            'min_datetime': min_dt,
            'max_datetime': max_dt,
            'min_energy': min_val,
            'max_energy': max_val,
            'mean_energy': round(mean_val, 2),
            'median_energy': round(median_val, 2)
        })

        quality_records.append({
            'filename': fname,
            'dataset_name': dataset_name,
            'target_column': e_col,
            'data_type': dtypes_dict[e_col],
            'missing_values': missing_cnt,
            'missing_pct': round(missing_pct, 4),
            'duplicate_rows': duplicate_rows,
            'duplicate_timestamps': duplicate_timestamps_count,
            'invalid_dates': invalid_dates_count,
            'zero_values': zero_cnt,
            'negative_values': neg_cnt
        })

    # Save cleaned file (Preserving raw observations, sorted chronologically)
    cleaned_path = f"data/processed/{dataset_name}_cleaned.csv"
    df_clean.to_csv(cleaned_path, index=False)
    print(f"  [Saved Cleaned] {cleaned_path}")

    # Task 3 & 4: Feature Engineering
    df_feat = df_clean.copy()

    # Calendar features
    dt = df_feat['Datetime']
    df_feat['hour'] = dt.dt.hour
    df_feat['day'] = dt.dt.day
    df_feat['day_of_week'] = dt.dt.dayofweek  # 0=Mon, 6=Sun
    df_feat['day_of_month'] = dt.dt.day
    df_feat['week_of_year'] = dt.dt.isocalendar().week.astype(int)
    df_feat['month'] = dt.dt.month
    df_feat['quarter'] = dt.dt.quarter
    df_feat['year'] = dt.dt.year
    df_feat['is_weekend'] = df_feat['day_of_week'].isin([5, 6]).astype(int)
    df_feat['is_weekday'] = (~df_feat['day_of_week'].isin([5, 6])).astype(int)
    df_feat['is_month_start'] = dt.dt.is_month_start.astype(int)
    df_feat['is_month_end'] = dt.dt.is_month_end.astype(int)
    df_feat['is_year_start'] = dt.dt.is_year_start.astype(int)
    df_feat['is_year_end'] = dt.dt.is_year_end.astype(int)

    # Cyclical features
    df_feat['sin_hour'] = np.sin(2 * np.pi * df_feat['hour'] / 24.0)
    df_feat['cos_hour'] = np.cos(2 * np.pi * df_feat['hour'] / 24.0)
    df_feat['sin_day_of_week'] = np.sin(2 * np.pi * df_feat['day_of_week'] / 7.0)
    df_feat['cos_day_of_week'] = np.cos(2 * np.pi * df_feat['day_of_week'] / 7.0)
    df_feat['sin_month'] = np.sin(2 * np.pi * (df_feat['month'] - 1) / 12.0)
    df_feat['cos_month'] = np.cos(2 * np.pi * (df_feat['month'] - 1) / 12.0)

    lags = [1, 2, 3, 6, 12, 24, 48, 72, 168]

    for e_col in energy_cols:
        col_prefix = f"{e_col}_" if len(energy_cols) > 1 else ""
        y = df_feat[e_col]

        # Lag features (Strictly previous observations)
        for L in lags:
            df_feat[f"{col_prefix}lag_{L}"] = y.shift(L)

        # Rolling features (Strictly past observations using y.shift(1))
        y_hist = y.shift(1)
        df_feat[f"{col_prefix}rolling_mean_6"] = y_hist.rolling(6).mean()
        df_feat[f"{col_prefix}rolling_mean_12"] = y_hist.rolling(12).mean()
        df_feat[f"{col_prefix}rolling_mean_24"] = y_hist.rolling(24).mean()
        df_feat[f"{col_prefix}rolling_mean_48"] = y_hist.rolling(48).mean()
        df_feat[f"{col_prefix}rolling_mean_168"] = y_hist.rolling(168).mean()
        df_feat[f"{col_prefix}rolling_std_24"] = y_hist.rolling(24).std()
        df_feat[f"{col_prefix}rolling_min_24"] = y_hist.rolling(24).min()
        df_feat[f"{col_prefix}rolling_max_24"] = y_hist.rolling(24).max()

        # Task 4: Peak Feature (85th percentile threshold)
        p_thresh = y.quantile(0.85)
        df_feat[f"{col_prefix}peak_threshold"] = p_thresh
        df_feat[f"{col_prefix}is_peak"] = (y > p_thresh).astype(int)

    features_path = f"data/features/{dataset_name}_features.csv"
    df_feat.to_csv(features_path, index=False)
    print(f"  [Saved Features] {features_path} (Cols: {len(df_feat.columns)})")

    # Task 5: Basic EDA & Visualizations
    primary_target = energy_cols[0]

    # 1. Energy consumption over time
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_clean['Datetime'], df_clean[primary_target], color='#1f77b4', linewidth=0.5, alpha=0.85)
    ax.set_title(f"{dataset_name}: Energy Consumption Over Time ({primary_target})", fontsize=12, fontweight='bold')
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Energy (MW)")
    plt.tight_layout()
    plt.savefig(f"visualizations/{dataset_name}_time_series.png")
    plt.close()

    # 2. Average energy by hour
    fig, ax = plt.subplots(figsize=(8, 4))
    hourly_avg = df_feat.groupby('hour')[primary_target].mean()
    ax.plot(hourly_avg.index, hourly_avg.values, marker='o', color='#ff7f0e', linewidth=2)
    ax.set_title(f"{dataset_name}: Average Energy by Hour of Day", fontsize=12, fontweight='bold')
    ax.set_xlabel("Hour of Day (0-23)")
    ax.set_ylabel("Mean Energy (MW)")
    ax.set_xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(f"visualizations/{dataset_name}_hourly_profile.png")
    plt.close()

    # 3. Average energy by day of week
    fig, ax = plt.subplots(figsize=(7, 4))
    dow_avg = df_feat.groupby('day_of_week')[primary_target].mean()
    days_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    ax.bar(days_labels, dow_avg.values, color='#2ca02c', alpha=0.85)
    ax.set_title(f"{dataset_name}: Average Energy by Day of Week", fontsize=12, fontweight='bold')
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Mean Energy (MW)")
    plt.tight_layout()
    plt.savefig(f"visualizations/{dataset_name}_dayofweek_profile.png")
    plt.close()

    # 4. Average energy by month
    fig, ax = plt.subplots(figsize=(8, 4))
    monthly_avg = df_feat.groupby('month')[primary_target].mean()
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.plot(month_labels, monthly_avg.values, marker='s', color='#d62728', linewidth=2)
    ax.set_title(f"{dataset_name}: Average Energy by Month", fontsize=12, fontweight='bold')
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean Energy (MW)")
    plt.tight_layout()
    plt.savefig(f"visualizations/{dataset_name}_monthly_profile.png")
    plt.close()

    # 5. Energy consumption distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df_clean[primary_target].dropna(), kde=True, ax=ax, color='#9467bd', bins=50)
    ax.set_title(f"{dataset_name}: Energy Consumption Distribution", fontsize=12, fontweight='bold')
    ax.set_xlabel("Energy (MW)")
    ax.set_ylabel("Frequency / Count")
    plt.tight_layout()
    plt.savefig(f"visualizations/{dataset_name}_distribution.png")
    plt.close()

# Export Task 6 Reports
df_summary = pd.DataFrame(summary_records)
df_summary.to_csv("reports/dataset_summary.csv", index=False)
print(f"\n[Saved Report] reports/dataset_summary.csv")

df_quality = pd.DataFrame(quality_records)
df_quality.to_csv("reports/data_quality_report.csv", index=False)
print(f"[Saved Report] reports/data_quality_report.csv")

print("\nPipeline Execution Complete!")
