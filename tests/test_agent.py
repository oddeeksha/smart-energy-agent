import pandas as pd
from src.agent import run_agent_step, run_agent_over_range
from src.explainability import explain_peak, explain_anomaly, feature_importance_chart_data


def _thresholds():
    return {("summer", 18): 1000.0, ("__global__", None): 1000.0}


def _rate_config():
    return {"peak_rate": 0.14, "offpeak_rate": 0.09, "source": "test fixture"}


def test_run_agent_step_no_trigger():
    entry = run_agent_step(
        timestamp=pd.Timestamp("2024-07-15 18:00"),
        actual=500.0, forecast=500.0, season="summer", hour=18,
        thresholds=_thresholds(), rolling_mean=0.0, rolling_std=10.0,
        k=3.0, rate_config=_rate_config(),
    )
    assert entry.trigger_type is None


def test_run_agent_step_peak_only():
    entry = run_agent_step(
        timestamp=pd.Timestamp("2024-07-15 18:00"),
        actual=1200.0, forecast=1200.0, season="summer", hour=18,
        thresholds=_thresholds(), rolling_mean=0.0, rolling_std=1000.0,  # huge std -> not anomalous
        k=3.0, rate_config=_rate_config(),
    )
    assert entry.trigger_type == "peak"


def test_run_agent_step_combined_trigger_does_not_crash():
    """CRITICAL: this is the combined peak+anomaly case flagged in the
    engineering spec review. Must produce a 'combined' entry, not crash,
    and must include both reasoning strings concatenated."""
    entry = run_agent_step(
        timestamp=pd.Timestamp("2024-07-15 18:00"),
        actual=1500.0, forecast=1200.0, season="summer", hour=18,
        thresholds=_thresholds(),  # forecast 1200 > threshold 1000 -> peak
        rolling_mean=0.0, rolling_std=10.0,  # residual=300, zscore=30 -> anomaly
        k=3.0, rate_config=_rate_config(),
    )
    assert entry.trigger_type == "combined"
    assert entry.reasoning is not None
    assert len(entry.reasoning) > 0


def test_run_agent_over_range_returns_dataframe_with_expected_columns():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-07-15", periods=3, freq="h"),
        "actual": [500.0, 1200.0, 1500.0],
        "predicted": [500.0, 1200.0, 1200.0],
        "season": ["summer"] * 3,
        "hour": [16, 17, 18],
        "rolling_mean": [0.0, 0.0, 0.0],
        "rolling_std": [10.0, 10.0, 10.0],
    })
    log_df = run_agent_over_range(df, _thresholds(), k=3.0, rate_config=_rate_config())
    assert "trigger_type" in log_df.columns
    assert len(log_df) == 3


def test_explain_peak_returns_nonempty_string():
    result = explain_peak(forecast=1200.0, threshold=1000.0, severity="Medium")
    assert isinstance(result, str) and len(result) > 0
