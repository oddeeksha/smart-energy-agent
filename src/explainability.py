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


def build_context_explanation(temperature: float = None, is_holiday=None) -> str:
    """Builds a specific, operator-focused contextual explanation string incorporating actual feature values.
    Returns empty string for moderate temperatures on non-holidays.
    """
    has_temp = temperature is not None and pd.notna(temperature)
    has_hol = is_holiday is not None and pd.notna(is_holiday)

    if not has_temp and not has_hol:
        return ""

    temp_val = float(temperature) if has_temp else None
    
    if has_hol:
        if isinstance(is_holiday, (int, float)):
            hol_bool = (int(is_holiday) == 1)
        elif isinstance(is_holiday, str):
            hol_bool = is_holiday.strip().lower() in ("1", "true", "yes")
        else:
            hol_bool = bool(is_holiday)
    else:
        hol_bool = False

    is_hot = temp_val is not None and temp_val >= 24.0
    is_cold = temp_val is not None and temp_val <= 5.0

    # Combined high temp + holiday
    if is_hot and hol_bool:
        return f" Ambient temperature is {temp_val:.1f}°C on a public holiday, which may be driving abnormal cooling demand."
    # Combined low temp + holiday
    elif is_cold and hol_bool:
        return f" Low temperature conditions ({temp_val:.1f}°C) combined with a public holiday may be driving abnormal demand shifts."
    # High temp only
    elif is_hot:
        return f" Ambient temperature is {temp_val:.1f}°C, which may be increasing cooling-related electricity demand."
    # Low temp only
    elif is_cold:
        return f" Low temperature conditions ({temp_val:.1f}°C) may be contributing to increased heating demand."
    # Holiday only (moderate temp)
    elif hol_bool:
        return " The event occurred on a public holiday, which may have altered normal consumption behavior."

    # Moderate temperature and not a holiday -> return empty string
    return ""


def explain_peak(
    forecast: float,
    threshold: float,
    severity: str,
    temperature: float = None,
    is_holiday=None,
) -> str:
    excess = forecast - threshold
    percent_excess = (excess / threshold * 100.0) if threshold > 0 else 0.0
    sev_upper = severity.upper()
    base_msg = (
        f"Forecast demand {forecast:,.0f} MW exceeds the seasonal peak threshold {threshold:,.0f} MW "
        f"by {excess:,.0f} MW (+{percent_excess:.1f}%). Peak risk classified as {sev_upper}."
    )
    return base_msg + build_context_explanation(temperature=temperature, is_holiday=is_holiday)


def explain_anomaly(
    residual: float,
    zscore: float,
    direction: str,
    severity: str,
    temperature: float = None,
    is_holiday=None,
) -> str:
    verb = "above" if direction == "high" else "below"
    base_msg = (
        f"Actual demand deviated {abs(zscore):.1f} standard deviations {verb} "
        f"expected — {severity} severity anomaly."
    )
    return base_msg + build_context_explanation(temperature=temperature, is_holiday=is_holiday)


def feature_importance_chart_data(model) -> pd.DataFrame:
   
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
