"""
P4 (part 1/2) — Anomaly Detection
Owns: this file. Contract: detect_anomaly(residual, rolling_mean, rolling_std, k) -> AnomalyResult
DO NOT change this signature — P5 depends on it directly.

CRITICAL: rolling_residual_stats MUST be strictly trailing (no lookahead).
Using pandas .rolling(window) with default settings (no center=True) is correct.
"""
import pandas as pd

from src.schemas import AnomalyResult
from src.config import ROLLING_WINDOW_HOURS, K_THRESHOLD, ANOMALY_SEVERITY_LOW_MAX, ANOMALY_SEVERITY_MEDIUM_MAX


def compute_residual(actual: float, predicted: float) -> float:
    return actual - predicted


def rolling_residual_stats(residuals: pd.Series, window: int = ROLLING_WINDOW_HOURS) -> tuple[pd.Series, pd.Series]:
    """STRICTLY TRAILING. First `window` rows will be NaN — this is expected
    (cold-start period). Do NOT pass center=True, that would be lookahead leakage.
    """
    rolling_mean = residuals.rolling(window=window, center=False).mean()
    rolling_std = residuals.rolling(window=window, center=False).std()
    return rolling_mean, rolling_std


def detect_anomaly(residual: float, rolling_mean: float, rolling_std: float,
                    k: float = K_THRESHOLD) -> AnomalyResult:
    """Returns AnomalyResult. direction is 'high' if residual > 0, 'low' if
    residual < 0, only meaningful when is_anomaly is True.

    For cold-start periods (NaN rolling statistics) or zero rolling
standard deviation, returns not-anomaly rather than raising.
    """
    if pd.isna(rolling_mean) or pd.isna(rolling_std) or rolling_std == 0:
        return AnomalyResult(is_anomaly=False, direction=None, zscore=0.0,
                              severity=None, severity_score=None, reasoning=None)

    zscore = (residual - rolling_mean) / rolling_std
    is_anomaly = abs(zscore) > k

    if not is_anomaly:
        return AnomalyResult(is_anomaly=False, direction=None, zscore=zscore,
                              severity=None, severity_score=None, reasoning=None)

    direction = "high" if residual > 0 else "low"
    score = anomaly_severity(residual, rolling_std)
    label = severity_label(score)
    reasoning = (
        f"Actual demand deviated {abs(zscore):.1f} standard deviations "
        f"{'above' if direction == 'high' else 'below'} expected — "
        f"{label} severity anomaly."
    )
    return AnomalyResult(is_anomaly=True, direction=direction, zscore=zscore,
                          severity=label, severity_score=score, reasoning=reasoning)


def anomaly_severity(residual: float, rolling_std: float) -> float:
    return abs(residual) / rolling_std


def severity_label(score: float) -> str:
    """This is the anomaly-scale severity label — z-score-like magnitude.
    Do NOT reuse these cutoffs for peak_severity_score (see peak_detection.py)."""
    if score < ANOMALY_SEVERITY_LOW_MAX:
        return "Low"
    elif score < ANOMALY_SEVERITY_MEDIUM_MAX:
        return "Medium"
    else:
        return "High"


if __name__ == "__main__":
    # TODO(P4): wire up to load features.csv + predictions, compute residuals
    # across the full range, run rolling_residual_stats, spot-check known
    # extreme-weather dates against detect_anomaly output.
    pass
