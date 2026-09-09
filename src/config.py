"""
Shared constants. Import these — never hardcode magic numbers inline.
Fill in real values as they're finalized (region, rate source, etc.).
"""

# --- Paths ---
RAW_DEMAND_PATH = "data/raw/demand.csv"
RAW_WEATHER_PATH = "data/raw/weather.csv"
FEATURES_PATH = "data/processed/features.csv"
MODEL_PATH = "data/artifacts/model.pkl"
THRESHOLDS_PATH = "data/artifacts/peak_thresholds.csv"
RATE_CONFIG_PATH = "data/artifacts/impact_rate_config.json"

# --- Anomaly detection ---
ROLLING_WINDOW_HOURS = 720  # 30 days, strictly trailing
K_THRESHOLD = 3.0  # TODO(P4): confirm this is the value locked after validation-split tuning

# --- Severity bands ---
# Anomaly severity is a z-score-like scale (roughly 0-5).
ANOMALY_SEVERITY_LOW_MAX = 2.0
ANOMALY_SEVERITY_MEDIUM_MAX = 3.5

# Peak severity is (forecast - threshold) / threshold, a *small percentage* scale.
# TODO(P3/P4): these are placeholders — replace with real cutoffs derived from
# the actual distribution of peak_severity_score in your data. Using the anomaly
# cutoffs here would be WRONG (see engineering spec critical blocker note) —
# every peak would register as "Low" forever.
PEAK_SEVERITY_LOW_MAX = 0.03
PEAK_SEVERITY_MEDIUM_MAX = 0.08

# --- Peak detection ---
PEAK_PERCENTILE = 0.95

# --- Feature columns expected in features.csv (P1's output contract) ---
FEATURE_COLUMNS = [
    "timestamp", "demand", "temperature", "hour", "day_of_week",
    "month", "season", "is_holiday", "lag_1", "lag_24", "lag_168", "split",
]

MODEL_FEATURE_COLUMNS = [
    "temperature", "hour", "day_of_week", "month", "is_holiday",
    "lag_1", "lag_24", "lag_168",
]
