"""
Configuration settings for Smart Energy Optimization PJME Forecasting Pipeline.
Defines column schemas, feature lists, data paths, split settings, and shared P2-P8 starter constants.
"""

import os

# --- Paths ---
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
FEATURES_DATA_DIR = "data/features"
REPORTS_DIR = "reports"

RAW_DEMAND_PATH = "data/raw/demand.csv"
RAW_WEATHER_PATH = "data/raw/weather.csv"
FEATURES_PATH = "data/processed/features.csv"
FEATURES_CSV_PATH = "data/features/features.csv"
MODEL_PATH = "data/artifacts/model.pkl"
THRESHOLDS_PATH = "data/artifacts/peak_thresholds.csv"
RATE_CONFIG_PATH = "data/artifacts/impact_rate_config.json"
FEATURE_IMPORTANCE_PATH = "data/artifacts/feature_importance.csv"

# --- Forecast Horizon & Targets ---
FORECAST_HORIZON = "1 hour ahead / t+1"
FORECAST_HORIZON_HOURS = 1
TARGET_DEMAND_COL = "target_demand"

PJME_RAW_PATH = os.path.join(FEATURES_DATA_DIR, "PJME_hourly_features_weather.csv")
FINAL_DATASET_PATH = os.path.join(FEATURES_DATA_DIR, "PJME_hourly_features_weather_final.csv")
FINAL_REPORT_PATH = os.path.join(REPORTS_DIR, "PJME_final_preprocessing_report.csv")

# --- Anomaly detection & Hyperparameters ---
ROLLING_WINDOW_HOURS = 720  # 30 days, strictly trailing
K_THRESHOLD = 3.0

# --- Severity bands ---
ANOMALY_SEVERITY_LOW_MAX = 2.0
ANOMALY_SEVERITY_MEDIUM_MAX = 3.5

PEAK_SEVERITY_LOW_MAX = 0.038654
PEAK_SEVERITY_MEDIUM_MAX = 0.106595

# --- Peak detection ---
PEAK_PERCENTILE = 0.95

# --- Key Column Identifiers ---
TIMESTAMP_COL = "timestamp"
TIMESTAMP_COLUMN = "timestamp"
TARGET_COL = "demand"
TARGET_COLUMN = "demand"
SPLIT_COL = "split"
SPLIT_COLUMN = "split"

# --- Target-derived Columns (Must not be used as model input features) ---
TARGET_DERIVED_COLS = ["peak_threshold", "is_peak"]
TARGET_DERIVED_COLUMNS = ["peak_threshold", "is_peak"]
PEAK_THRESHOLD_COL = "peak_threshold"
IS_PEAK_COL = "is_peak"

# --- Weather Columns ---
WEATHER_COLS = ["temperature", "relative_humidity_2m", "precipitation", "cloud_cover", "wind_speed_10m"]
WEATHER_COLUMNS = WEATHER_COLS

# --- Starter Feature Columns List ---
FEATURE_COLUMNS = [
    "timestamp", "demand", "temperature", "hour", "day_of_week",
    "month", "season", "is_holiday", "lag_1", "lag_24", "lag_168", "split",
]

MODEL_FEATURE_COLUMNS = [
    "temperature", "hour", "day_of_week", "month", "is_holiday",
    "lag_1", "lag_24", "lag_168",
]

# --- 49 Complete Output Dataset Columns (Ordered) ---
FINAL_COLUMN_LIST = [
    # Metadata & Target
    "timestamp",
    "demand",

    # Calendar Features (14)
    "hour",
    "day",
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month",
    "quarter",
    "year",
    "is_weekend",
    "is_weekday",
    "is_month_start",
    "is_month_end",
    "is_year_start",
    "is_year_end",

    # Cyclical Features (6)
    "sin_hour",
    "cos_hour",
    "sin_day_of_week",
    "cos_day_of_week",
    "sin_month",
    "cos_month",

    # Lag Features (9)
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "lag_24",
    "lag_48",
    "lag_72",
    "lag_168",

    # Rolling Features (8)
    "rolling_mean_6",
    "rolling_mean_12",
    "rolling_mean_24",
    "rolling_mean_48",
    "rolling_mean_168",
    "rolling_std_24",
    "rolling_min_24",
    "rolling_max_24",

    # Peak Target Indicators (2)
    "peak_threshold",
    "is_peak",

    # Weather Features (5)
    "temperature",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "wind_speed_10m",

    # Pipeline Additions (3)
    "season",
    "is_holiday",
    "split"
]

# Predictor Features (44 model inputs)
MODEL_FEATURE_COLS = [
    c for c in FINAL_COLUMN_LIST
    if c not in [TIMESTAMP_COL, TARGET_COL, SPLIT_COL] + TARGET_DERIVED_COLS
]

MODEL_FEATURES = MODEL_FEATURE_COLS
FEATURE_COLS = MODEL_FEATURE_COLS
ALL_FEATURES = MODEL_FEATURE_COLS
