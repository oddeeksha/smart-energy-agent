import pandas as pd
from src.anomaly_detection import compute_residual, rolling_residual_stats, detect_anomaly, anomaly_severity
from src.recommendations import get_recommendation, estimate_impact


def test_compute_residual():
    assert compute_residual(actual=110.0, predicted=100.0) == 10.0


def test_rolling_stats_is_trailing_not_lookahead():
    # Construct a series where a lookahead bug would be detectable:
    # a big spike at the END should NOT affect the rolling stat computed
    # at an EARLIER index.
    residuals = pd.Series([0.0] * 50 + [1000.0])  # spike only at the very last row
    rolling_mean, rolling_std = rolling_residual_stats(residuals, window=10)
    # rolling stat at index 20 (well before the spike) must be unaffected by it
    assert rolling_mean.iloc[20] == 0.0


def test_detect_anomaly_flags_large_deviation():
    result = detect_anomaly(residual=100.0, rolling_mean=0.0, rolling_std=10.0, k=3.0)
    assert result.is_anomaly is True
    assert result.direction == "high"
    assert result.severity in {"Low", "Medium", "High"}


def test_detect_anomaly_handles_cold_start_nan_gracefully():
    result = detect_anomaly(residual=100.0, rolling_mean=float("nan"), rolling_std=float("nan"), k=3.0)
    assert result.is_anomaly is False  # must not raise


def test_estimate_impact_never_negative():
    impact = estimate_impact(predicted_demand=900.0, threshold=1000.0,
                              peak_rate=0.14, offpeak_rate=0.09)
    assert impact == 0.0  # below threshold -> no reduction target -> zero impact


def test_get_recommendation_returns_expected_trigger_type():
    rec = get_recommendation("peak", "Medium")
    assert rec.trigger_type == "peak"
    assert "demand-response" in rec.action_text.lower()


# Example input/output:
#
# detect_anomaly(residual=850.0, rolling_mean=0.0, rolling_std=200.0, k=3.0)
# -> AnomalyResult(is_anomaly=True, direction="high", zscore=4.25,
#                   severity="High", severity_score=4.25, reasoning="...")
