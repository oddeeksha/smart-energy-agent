import pandas as pd
from src.peak_detection import compute_thresholds, detect_peak, peak_severity, severity_label


def _fake_train_df(n=500):
    import numpy as np
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "season": rng.choice(["winter", "spring", "summer", "fall"], n),
        "hour": rng.integers(0, 24, n),
        "demand": rng.uniform(1000, 2000, n),
    })


def test_compute_thresholds_has_global_fallback():
    train_df = _fake_train_df()
    thresholds = compute_thresholds(train_df)
    assert ("__global__", None) in thresholds


def test_detect_peak_below_threshold_returns_not_peak():
    thresholds = {("summer", 18): 2000.0, ("__global__", None): 2000.0}
    result = detect_peak(forecast=1000.0, season="summer", hour=18, thresholds=thresholds)
    assert result.is_peak is False
    assert result.severity is None


def test_detect_peak_above_threshold_returns_peak_with_severity():
    thresholds = {("summer", 18): 1000.0, ("__global__", None): 1000.0}
    result = detect_peak(forecast=1200.0, season="summer", hour=18, thresholds=thresholds)
    assert result.is_peak is True
    assert result.severity in {"Low", "Medium", "High"}
    assert result.reasoning is not None


def test_detect_peak_missing_bucket_uses_global_fallback():
    thresholds = {("__global__", None): 1500.0}  # no ("winter", 3) entry
    result = detect_peak(forecast=1600.0, season="winter", hour=3, thresholds=thresholds)
    assert result.threshold == 1500.0  # used the fallback, didn't crash


# Example input/output:
#
# detect_peak(forecast=41890.2, season="summer", hour=18, thresholds={...})
# -> PeakResult(is_peak=True, threshold=41000.0, severity="Medium",
#               severity_score=0.0217, reasoning="Forecasted demand ...")
