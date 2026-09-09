"""
P3 — Peak Detection
Owns: this file. Contract: detect_peak(forecast, season, hour, thresholds) -> PeakResult
DO NOT change this signature — P5 depends on it directly.

CRITICAL: peak_severity_score is on a DIFFERENT numeric scale than anomaly
severity (see config.py comments). Do not reuse the anomaly cutoffs.
"""
import pandas as pd

from src.schemas import PeakResult
from src.config import (
    PEAK_PERCENTILE,
    PEAK_SEVERITY_LOW_MAX,
    PEAK_SEVERITY_MEDIUM_MAX,
    THRESHOLDS_PATH,
)


def compute_thresholds(train_df: pd.DataFrame) -> dict:
    """Groups by (season, hour), computes the 95th percentile of 'demand'.
    MUST be called on split == 'train' rows only — this is the leakage
    guard flagged in the engineering spec. Caller is responsible for
    filtering to train before calling this.

    Returns {(season, hour): threshold_value}, plus a global fallback
    under key ('__global__', None).
    """
    grouped = train_df.groupby(["season", "hour"])["demand"].quantile(PEAK_PERCENTILE)
    thresholds = {(season, hour): value for (season, hour), value in grouped.items()}
    thresholds[("__global__", None)] = train_df["demand"].quantile(PEAK_PERCENTILE)
    return thresholds


def detect_peak(forecast: float, season: str, hour: int, thresholds: dict) -> PeakResult:
    """Looks up threshold for (season, hour). Falls back to global threshold
    if that specific bucket is missing (e.g. sparse data for some combo).
    """
    key = (season, hour)
    threshold = thresholds.get(key, thresholds.get(("__global__", None)))

    is_peak = forecast > threshold
    if not is_peak:
        return PeakResult(is_peak=False, threshold=threshold, severity=None,
                           severity_score=None, reasoning=None)

    score = peak_severity(forecast, threshold)
    label = severity_label(score)
    reasoning = (
        f"Forecasted demand ({forecast:.0f}) exceeds the seasonal "
        f"95th-percentile threshold ({threshold:.0f}) — {label} severity peak event."
    )
    return PeakResult(is_peak=True, threshold=threshold, severity=label,
                       severity_score=score, reasoning=reasoning)


def peak_severity(forecast: float, threshold: float) -> float:
    """Returns (forecast - threshold) / threshold.
    Can be negative if below threshold — caller checks is_peak before treating
    this as meaningful.
    """
    return (forecast - threshold) / threshold


def severity_label(score: float) -> str:
    """TODO(P3 + P4, agree together): these cutoffs are placeholders.
    Derive real cutoffs from the actual distribution of peak_severity_score
    in your data before locking this. Do NOT reuse the anomaly z-score cutoffs.
    """
    if score < PEAK_SEVERITY_LOW_MAX:
        return "Low"
    elif score < PEAK_SEVERITY_MEDIUM_MAX:
        return "Medium"
    else:
        return "High"


def save_thresholds(thresholds: dict, path: str = THRESHOLDS_PATH) -> None:
    rows = [{"season": s, "hour": h, "threshold": v} for (s, h), v in thresholds.items()]
    pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    # TODO(P3): load features.csv, filter to split=='train', compute_thresholds,
    # save to THRESHOLDS_PATH.
    pass
