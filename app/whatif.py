"""
P8 (part 2/2) — What-If Simulator
Owns: this file. Contract: render_whatif_simulator(current_forecast, current_threshold, rate_config)
Reuses P4's estimate_impact() — does NOT re-run the forecasting model.
The UI copy must make this explicit (flagged as a judge-question risk in review).
"""
import streamlit as st

from src.recommendations import estimate_impact


def render_whatif_simulator(current_forecast: float, current_threshold: float,
                             rate_config: dict) -> None:
    """User inputs a hypothetical MW reduction via st.slider.
    Recomputes peak status and impact using the SAME formula as the live
    pipeline — does not re-forecast.
    """
    st.caption(
        "This simulator recomputes the impact estimate for a hypothetical "
        "demand reduction using the existing forecast — it does not re-run "
        "the forecasting model."
    )

    excess = current_forecast - current_threshold
    if excess > 0:
        st.info(f"⚡ **Active Peak Event**: Demand exceeds threshold by **{excess:,.0f} MW**.")
    else:
        st.caption(f"🟢 **Grid Normal**: Forecast is **{abs(excess):,.0f} MW** below peak threshold.")

    reduction = st.slider("Hypothetical demand reduction (MW)", 0, 5000, 0, step=100)
    adjusted_demand = current_forecast - reduction
    new_is_peak = adjusted_demand > current_threshold

    new_impact = estimate_impact(
        adjusted_demand, current_threshold,
        rate_config.get("peak_rate", 0.0), rate_config.get("offpeak_rate", 0.0),
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Peak status after reduction", "Still peaking" if new_is_peak else "Peak avoided")
    with col2:
        st.metric("Estimated impact after reduction", f"${new_impact:,.0f}")



if __name__ == "__main__":
    # TODO(P8): standalone smoke test with placeholder forecast/threshold/rate_config values.
    pass
