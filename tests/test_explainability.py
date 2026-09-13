from src.explainability import (
    explain_peak,
    explain_anomaly,
    feature_importance_chart_data,
)
from src.forecasting import load_model


def test_explain_peak_returns_expected_message():
    result = explain_peak(500, 450, "High")

    assert isinstance(result, str)
    assert "500" in result
    assert "450" in result
    assert "High severity peak event" in result


def test_explain_anomaly_returns_expected_message():
    result = explain_anomaly(30, 3.2, "high", "Medium")

    assert isinstance(result, str)
    assert "3.2 standard deviations above expected" in result
    assert "Medium severity anomaly" in result


def test_feature_importance_chart_data_with_trained_model():
    model = load_model()
    result = feature_importance_chart_data(model)

    assert list(result.columns) == ["feature", "importance"]
    assert len(result) == 8
    assert result["importance"].ge(0).all()
    assert result["importance"].is_monotonic_decreasing