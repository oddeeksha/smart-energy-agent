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
    temperature: float = None,
    is_holiday=None,
) -> LogEntry:
    """The full perceive-evaluate-recommend-log cycle for one timestep."""

    # --- Perceive + Evaluate ---
    peak_result = detect_peak(forecast, season, hour, thresholds, temperature=temperature, is_holiday=is_holiday)
    residual = compute_residual(actual, forecast)
    anomaly_result = detect_anomaly(residual, rolling_mean, rolling_std, k)

    # --- Act (mandatory combined-trigger rule) ---
    if peak_result.is_peak and anomaly_result.is_anomaly:
        trigger_type = "combined"
        severity = (peak_result.severity if severity_rank(peak_result.severity) >= severity_rank(anomaly_result.severity)
                    else anomaly_result.severity)
        reasoning = f"{peak_result.reasoning} {anomaly_result.reasoning}"
        impact = estimate_impact(forecast, peak_result.threshold,
                                  rate_config.get("peak_rate", 0.0), rate_config.get("offpeak_rate", 0.0))
        recommendation = get_recommendation("combined", severity, forecast=forecast, threshold=peak_result.threshold, estimated_impact=impact, temperature=temperature, season=season, hour=hour)

    elif peak_result.is_peak:
        trigger_type = "peak"
        severity = peak_result.severity
        reasoning = peak_result.reasoning
        impact = estimate_impact(forecast, peak_result.threshold,
                                  rate_config.get("peak_rate", 0.0), rate_config.get("offpeak_rate", 0.0))
        recommendation = get_recommendation("peak", severity, forecast=forecast, threshold=peak_result.threshold, estimated_impact=impact, temperature=temperature, season=season, hour=hour)

    elif anomaly_result.is_anomaly:
        trigger_type = f"{anomaly_result.direction}_anomaly"
        severity = anomaly_result.severity
        reasoning = anomaly_result.reasoning
        impact = 0.0
        recommendation = get_recommendation(trigger_type, severity, forecast=forecast, threshold=0.0, estimated_impact=0.0, temperature=temperature, season=season, hour=hour)


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
        peak_threshold=peak_result.threshold,
    )


def run_agent_over_range(df: pd.DataFrame, thresholds: dict, k: float,
                          rate_config: dict) -> pd.DataFrame:
    """Runs run_agent_step across every row in df (must be time-ordered,
    must already have 'actual', 'predicted', 'season', 'hour', 'rolling_mean',
    'rolling_std' columns computed). Returns a dataframe of LogEntry rows.
    """
    log_entries = []
    has_temp = "temperature" in df.columns
    has_hol = "is_holiday" in df.columns

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
            temperature=row["temperature"] if has_temp else None,
            is_holiday=row["is_holiday"] if has_hol else None,
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

    # 7. Save agent log artifact and full pipeline outputs
    os.makedirs(REPORTS_DIR, exist_ok=True)
    output_log_path = os.path.join(REPORTS_DIR, "agent_execution_log.csv")
    log_df.to_csv(output_log_path, index=False)

    pipeline_outputs_path = os.path.join(REPORTS_DIR, "pipeline_outputs.csv")
    # Join log entry attributes back to test_df for pipeline_outputs.csv
    for col in ["trigger_type", "severity", "reasoning", "recommendation", "estimated_impact", "peak_threshold"]:
        test_df[col] = log_df[col]
    test_df.to_csv(pipeline_outputs_path, index=False)

    print(f"\n[OK] Agent execution complete. Log saved to: {output_log_path}")
    print(f"[OK] Pipeline outputs saved to: {pipeline_outputs_path}")
    print(f"Trigger Summary across test set:")
    print(log_df["trigger_type"].value_counts(dropna=False))


