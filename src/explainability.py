"""
P6 — Explainability
Owns: this file. Contract: explain_peak/explain_anomaly return template strings,
feature_importance_chart_data returns a dataframe for the UI.

NOTE: peak_detection.py and anomaly_detection.py already build reasoning strings
inline via these same templates conceptually — if the team decides P3/P4's
inline reasoning is sufficient, this file's explain_* functions can just be
thin wrappers or left unused. Coordinate with P3/P4 to avoid duplicate logic.
"""
import pandas as pd


def explain_peak(forecast: float, threshold: float, severity: str) -> str:
    return (
        f"Forecasted demand ({forecast:.0f}) exceeds the seasonal 95th-percentile "
        f"threshold ({threshold:.0f}) — {severity} severity peak event."
    )


def explain_anomaly(residual: float, zscore: float, direction: str, severity: str) -> str:
    verb = "above" if direction == "high" else "below"
    return (
        f"Actual demand deviated {abs(zscore):.1f} standard deviations {verb} "
        f"expected — {severity} severity anomaly."
    )


def feature_importance_chart_data(model) -> pd.DataFrame:
    """Returns {feature, importance} sorted descending, from model.feature_importances_.

    TODO(P6): import MODEL_FEATURE_COLUMNS from config to label the importances
    correctly — the order must match what was passed into model.fit().
    """
    from src.config import MODEL_FEATURE_COLUMNS
    importances = model.feature_importances_
    df = pd.DataFrame({
        "feature": MODEL_FEATURE_COLUMNS,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return df


if __name__ == "__main__":
    # TODO(P6): load the trained model, call feature_importance_chart_data,
    # print/save the result for the deck.
    pass
