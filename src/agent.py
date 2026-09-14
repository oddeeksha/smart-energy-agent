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
    import os
    from src.config import FEATURES_CSV_PATH, MODEL_PATH, K_THRESHOLD, ROLLING_WINDOW_HOURS, REPORTS_DIR
    from src.forecasting import load_model, predict
    from src.peak_detection import compute_thresholds
    from src.anomaly_detection import compute_residual, rolling_residual_stats
    from src.recommendations import load_rate_config

    print("==================================================")
    print("RUNNING P5 AGENT ORCHESTRATION PIPELINE")
    print("==================================================")

    # 1. Load feature dataset & set actual demand alias
    print(f"Loading feature dataset: {FEATURES_CSV_PATH}")
    df = pd.read_csv(FEATURES_CSV_PATH)
    df["actual"] = df["demand"]

    # 2. Load model & predict across full timeline
    print(f"Loading forecasting model: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    df = predict(model, df)

    # 3. Compute rolling residual statistics (strictly trailing)
    print(f"Computing rolling residual stats (window={ROLLING_WINDOW_HOURS}h)...")
    residuals = compute_residual(df["actual"], df["predicted"])
    df["rolling_mean"], df["rolling_std"] = rolling_residual_stats(residuals, window=ROLLING_WINDOW_HOURS)

    # 4. Compute peak thresholds from training split
    train_df = df[df["split"] == "train"]
    thresholds = compute_thresholds(train_df)

    # 5. Load rate config for financial impact estimation
    rate_config = load_rate_config()

    # 6. Run agent over held-out test split
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    print(f"Running agent over test set ({len(test_df):,} timesteps)...")
    log_df = run_agent_over_range(test_df, thresholds, k=K_THRESHOLD, rate_config=rate_config)

    # 7. Save agent log artifact & pipeline outputs artifact
    os.makedirs(REPORTS_DIR, exist_ok=True)
    output_log_path = os.path.join(REPORTS_DIR, "agent_execution_log.csv")
    log_df.to_csv(output_log_path, index=False)

    # 8. Extract complete P2/P3/P4/P5 metrics to create comprehensive pipeline_outputs.csv
    detailed_records = []
    source_note = rate_config.get("source", "")

    for _, row in test_df.iterrows():
        ts_str = str(row["timestamp"])
        forecast_val = row["predicted"]
        actual_val = row["actual"]
        season_val = row["season"]
        hour_val = row["hour"]
        rmean = row["rolling_mean"]
        rstd = row["rolling_std"]

        # Re-evaluate P3 peak & P4 anomaly outputs to capture individual metrics
        pr = detect_peak(forecast_val, season_val, hour_val, thresholds)
        res = compute_residual(actual_val, forecast_val)
        ar = detect_anomaly(res, rmean, rstd, K_THRESHOLD)

        detailed_records.append({
            "timestamp": ts_str,
            "season": season_val,
            "hour": hour_val,
            "actual": actual_val,
            "predicted": forecast_val,
            "residual": res,
            "rolling_mean": rmean,
            "rolling_std": rstd,
            "z_score": ar.zscore,
            "is_anomaly": ar.is_anomaly,
            "anomaly_direction": ar.direction,
            "anomaly_severity": ar.severity,
            "anomaly_severity_score": ar.severity_score,
            "peak_threshold": pr.threshold,
            "is_peak": pr.is_peak,
            "peak_severity": pr.severity,
            "peak_severity_score": pr.severity_score,
            "impact_source_note": source_note,
        })

    detailed_df = pd.DataFrame(detailed_records)
    log_cols = ["trigger_type", "severity", "reasoning", "recommendation", "estimated_impact"]
    pipeline_outputs_df = pd.concat([detailed_df, log_df[log_cols]], axis=1)

    ordered_cols = [
        "timestamp",
        "season",
        "hour",
        "actual",
        "predicted",
        "residual",
        "rolling_mean",
        "rolling_std",
        "z_score",
        "is_anomaly",
        "anomaly_direction",
        "anomaly_severity",
        "anomaly_severity_score",
        "peak_threshold",
        "is_peak",
        "peak_severity",
        "peak_severity_score",
        "trigger_type",
        "severity",
        "reasoning",
        "recommendation",
        "estimated_impact",
        "impact_source_note",
    ]
    pipeline_outputs_df = pipeline_outputs_df[ordered_cols]

    output_pipeline_path = os.path.join(REPORTS_DIR, "pipeline_outputs.csv")
    pipeline_outputs_df.to_csv(output_pipeline_path, index=False)

    print(f"\n[OK] Agent execution complete.")
    print(f"Log saved to: {output_log_path}")
    print(f"Comprehensive pipeline outputs saved to: {output_pipeline_path}")
    print(f"Trigger Summary across test set:")
    print(log_df["trigger_type"].value_counts(dropna=False))

