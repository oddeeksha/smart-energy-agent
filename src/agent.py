"""
P5 — Agent Orchestration Loop
Owns: this file. Contract: run_agent_step(...) -> LogEntry, run_agent_over_range(...) -> pd.DataFrame
DO NOT change these signatures — P7/P8 depend on run_agent_over_range's output shape.

Implements the mandatory combined-trigger rule from the engineering spec.
"""
import pandas as pd

from src.schemas import LogEntry, severity_rank
from src.peak_detection import detect_peak
from src.anomaly_detection import compute_residual, detect_anomaly
from src.recommendations import get_recommendation, estimate_impact


def run_agent_step(
    timestamp: pd.Timestamp,
    actual: float,
    forecast: float,
    season: str,
    hour: int,
    thresholds: dict,
    rolling_mean: float,
    rolling_std: float,
    k: float,
    rate_config: dict,
) -> LogEntry:
    """The full perceive-evaluate-recommend-log cycle for one timestep."""

    # --- Perceive + Evaluate ---
    peak_result = detect_peak(forecast, season, hour, thresholds)
    residual = compute_residual(actual, forecast)
    anomaly_result = detect_anomaly(residual, rolling_mean, rolling_std, k)

    # --- Act (mandatory combined-trigger rule) ---
    if peak_result.is_peak and anomaly_result.is_anomaly:
        trigger_type = "combined"
        severity = (peak_result.severity if severity_rank(peak_result.severity) >= severity_rank(anomaly_result.severity)
                    else anomaly_result.severity)
        reasoning = f"{peak_result.reasoning} {anomaly_result.reasoning}"
        recommendation = get_recommendation("combined", severity)
        impact = estimate_impact(forecast, peak_result.threshold,
                                  rate_config.get("peak_rate", 0.0), rate_config.get("offpeak_rate", 0.0))

    elif peak_result.is_peak:
        trigger_type = "peak"
        severity = peak_result.severity
        reasoning = peak_result.reasoning
        recommendation = get_recommendation("peak", severity)
        impact = estimate_impact(forecast, peak_result.threshold,
                                  rate_config.get("peak_rate", 0.0), rate_config.get("offpeak_rate", 0.0))

    elif anomaly_result.is_anomaly:
        trigger_type = f"{anomaly_result.direction}_anomaly"
        severity = anomaly_result.severity
        reasoning = anomaly_result.reasoning
        recommendation = get_recommendation(trigger_type, severity)
        impact = 0.0  # TODO(P4/P5): decide if anomaly-only triggers get a nonzero impact estimate

    else:
        trigger_type = None
        severity = None
        reasoning = None
        recommendation = None
        impact = None

    return LogEntry(
        timestamp=str(timestamp),
        trigger_type=trigger_type,
        severity=severity,
        reasoning=reasoning,
        recommendation=recommendation.action_text if recommendation else None,
        estimated_impact=impact,
    )


def run_agent_over_range(df: pd.DataFrame, thresholds: dict, k: float,
                          rate_config: dict) -> pd.DataFrame:
    """Runs run_agent_step across every row in df (must be time-ordered,
    must already have 'actual', 'predicted', 'season', 'hour', 'rolling_mean',
    'rolling_std' columns computed). Returns a dataframe of LogEntry rows.
    """
    log_entries = []
    for _, row in df.iterrows():
        entry = run_agent_step(
            timestamp=row["timestamp"],
            actual=row["actual"],
            forecast=row["predicted"],
            season=row["season"],
            hour=row["hour"],
            thresholds=thresholds,
            rolling_mean=row["rolling_mean"],
            rolling_std=row["rolling_std"],
            k=k,
            rate_config=rate_config,
        )
        log_entries.append(entry.__dict__)
    return pd.DataFrame(log_entries)


if __name__ == "__main__":
    # TODO(P5): wire up to load features.csv + predictions + rolling stats,
    # load thresholds and rate_config, run over the full test range, save log.
    pass
