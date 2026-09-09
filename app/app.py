"""
P7 — App Shell, Chart, Replay Control
Owns: this file. Contract: load_pipeline_outputs(), render_chart(), render_replay_control()
Run with: streamlit run app/app.py
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.forecasting import load_model, predict
from src.peak_detection import detect_peak
from src.agent import run_agent_over_range
from src.config import FEATURES_PATH, THRESHOLDS_PATH, RATE_CONFIG_PATH, K_THRESHOLD


def load_pipeline_outputs() -> pd.DataFrame:
    """Loads features.csv + runs (or loads a cached) full agent pipeline run.
    Returns one merged dataframe: timestamp, actual, predicted, is_peak,
    is_anomaly, severity, trigger_type.

    TODO(P7): this is currently a placeholder that returns FAKE data so the
    app runs end-to-end immediately. Replace with real loading once P1/P2/P5
    outputs are available:
      1. load features.csv, filter split == 'test'
      2. load model, call predict()
      3. compute rolling residual stats
      4. load thresholds
      5. call run_agent_over_range()
      6. merge everything into one dataframe for the chart/log/panels
    """
    n = 200
    timestamps = pd.date_range("2024-07-01", periods=n, freq="h")
    import numpy as np
    rng = np.random.default_rng(42)
    actual = 1000 + rng.normal(0, 50, n).cumsum() * 0.1 + 200 * np.sin(np.arange(n) / 24 * 2 * 3.14)
    predicted = actual + rng.normal(0, 30, n)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "actual": actual,
        "predicted": predicted,
    })
    # FAKE trigger flags for now — TODO(P7): replace with real agent log merge
    df["trigger_type"] = None
    df.loc[df.index % 47 == 0, "trigger_type"] = "peak"
    df.loc[df.index % 61 == 0, "trigger_type"] = "high_anomaly"
    df["severity"] = df["trigger_type"].map({"peak": "Medium", "high_anomaly": "High"})
    return df


def render_chart(df: pd.DataFrame, current_step: int) -> None:
    """Renders actual-vs-predicted line chart up to current_step,
    with markers for peak/anomaly rows."""
    visible = df.iloc[:current_step + 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=visible["timestamp"], y=visible["actual"],
                              name="Actual", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=visible["timestamp"], y=visible["predicted"],
                              name="Forecast", line=dict(color="#ff7f0e", dash="dot")))

    peaks = visible[visible["trigger_type"] == "peak"]
    fig.add_trace(go.Scatter(x=peaks["timestamp"], y=peaks["actual"], mode="markers",
                              name="Peak", marker=dict(color="orange", size=10, symbol="triangle-up")))

    anomalies = visible[visible["trigger_type"].isin(["high_anomaly", "low_anomaly"])]
    fig.add_trace(go.Scatter(x=anomalies["timestamp"], y=anomalies["actual"], mode="markers",
                              name="Anomaly", marker=dict(color="red", size=10, symbol="x")))

    combined = visible[visible["trigger_type"] == "combined"]
    fig.add_trace(go.Scatter(x=combined["timestamp"], y=combined["actual"], mode="markers",
                              name="Combined (peak+anomaly)",
                              marker=dict(color="purple", size=12, symbol="star")))

    fig.update_layout(height=450, margin=dict(l=20, r=20, t=30, b=20),
                       xaxis_title="Time", yaxis_title="Demand")
    st.plotly_chart(fig, use_container_width=True)


def render_replay_control(max_steps: int) -> int:
    """Renders the slider, returns current_step."""
    return st.slider("Replay progress", 0, max_steps - 1, 0)


def main():
    st.set_page_config(layout="wide", page_title="Smart Energy Consumption Optimization Agent")
    st.title("Smart Energy Consumption Optimization Agent")
    st.caption("Grid-operator decision-support system — PJM regional demand")

    df = load_pipeline_outputs()

    col1, col2 = st.columns([2, 1])
    with col1:
        current_step = render_replay_control(len(df))
        render_chart(df, current_step)
    with col2:
        # TODO(P8): import and call render_log_panel, render_recommendation_panel here
        st.subheader("Agent Activity Log")
        st.info("TODO(P8): wire up ui_panels.render_log_panel(log_df, current_step)")
        st.subheader("Recommendations")
        st.info("TODO(P8): wire up ui_panels.render_recommendation_panel(log_df, current_step)")

    st.divider()
    st.subheader("What-If Simulator")
    st.info("TODO(P8): wire up whatif.render_whatif_simulator(...)")


if __name__ == "__main__":
    main()
