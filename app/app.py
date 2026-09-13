"""
P7 — App Shell, Chart, Replay Control
Owns: this file.
Contract: load_pipeline_outputs(), render_chart(), render_replay_control()

Run with:
streamlit run app/app.py
"""

import json

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.forecasting import load_model, predict
from src.agent import run_agent_over_range
from src.anomaly_detection import rolling_residual_stats
from src.config import (
    FEATURES_CSV_PATH,
    MODEL_PATH,
    THRESHOLDS_PATH,
    RATE_CONFIG_PATH,
    K_THRESHOLD,
)


@st.cache_data
def load_pipeline_outputs() -> pd.DataFrame:
    """Load real test data, generate forecasts, run the P5 agent,
    and return the combined output for the Streamlit app.
    """

    # 1. Load feature data
    df = pd.read_csv(FEATURES_CSV_PATH)

    # 2. Keep only test split
    test_df = df[df["split"] == "test"].copy()
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])

    # 3. Generate forecasts using trained model
    model = load_model(MODEL_PATH)
    test_df = predict(model, test_df)

    # 4. Calculate residual and rolling statistics
    test_df["residual"] = test_df["demand"] - test_df["predicted"]

    test_df["rolling_mean"], test_df["rolling_std"] = rolling_residual_stats(
        test_df["residual"]
    )

    # 5. Load peak thresholds
    thresholds_df = pd.read_csv(THRESHOLDS_PATH)

    thresholds = {
        (row["season"], int(row["hour"])): row["threshold"]
        for _, row in thresholds_df.dropna(subset=["hour"]).iterrows()
    }

    # 6. Load impact-rate configuration
    with open(RATE_CONFIG_PATH, "r") as f:
        rate_config = json.load(f)

    # 7. Prepare input for P5 agent
    agent_input = test_df[
        [
            "timestamp",
            "demand",
            "predicted",
            "season",
            "hour",
            "rolling_mean",
            "rolling_std",
        ]
    ].rename(columns={"demand": "actual"})

    # 8. Run P5 agent
    log_df = run_agent_over_range(
        agent_input,
        thresholds,
        K_THRESHOLD,
        rate_config,
    )

    log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])

    # 9. Combine actual, forecast and agent output
    result = test_df[["timestamp", "demand", "predicted"]].rename(
        columns={"demand": "actual"}
    )

    result = result.merge(log_df, on="timestamp", how="left")

    return result


def render_chart(df: pd.DataFrame, current_step: int) -> None:
    """Render actual vs forecast with peak and anomaly markers."""

    visible = df.iloc[: current_step + 1]

    fig = go.Figure()

    # Actual demand
    fig.add_trace(
        go.Scatter(
            x=visible["timestamp"],
            y=visible["actual"],
            name="Actual",
        )
    )

    # Forecast
    fig.add_trace(
        go.Scatter(
            x=visible["timestamp"],
            y=visible["predicted"],
            name="Forecast",
            line=dict(dash="dot"),
        )
    )

    # Peak events
    peaks = visible[visible["trigger_type"] == "peak"]

    fig.add_trace(
        go.Scatter(
            x=peaks["timestamp"],
            y=peaks["actual"],
            mode="markers",
            name="Peak",
            marker=dict(size=10, symbol="triangle-up"),
        )
    )

    # Anomaly events
    anomalies = visible[
        visible["trigger_type"].isin(
            ["high_anomaly", "low_anomaly"]
        )
    ]

    fig.add_trace(
        go.Scatter(
            x=anomalies["timestamp"],
            y=anomalies["actual"],
            mode="markers",
            name="Anomaly",
            marker=dict(size=10, symbol="x"),
        )
    )

    # Combined events
    combined = visible[
        visible["trigger_type"] == "combined"
    ]

    fig.add_trace(
        go.Scatter(
            x=combined["timestamp"],
            y=combined["actual"],
            mode="markers",
            name="Combined",
            marker=dict(size=12, symbol="star"),
        )
    )

    fig.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Time",
        yaxis_title="Demand",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_replay_control(max_steps: int) -> int:
    """Render replay slider and return current step."""

    return st.slider(
        "Replay progress",
        0,
        max_steps - 1,
        0,
    )


def main():
    st.set_page_config(
        layout="wide",
        page_title="Smart Energy Consumption Optimization Agent",
    )

    st.title("Smart Energy Consumption Optimization Agent")

    st.caption(
        "Grid-operator decision-support system — PJM regional demand"
    )

    # Load real P3 → P4 → P5 pipeline output
    df = load_pipeline_outputs()

    col1, col2 = st.columns([2, 1])

    with col1:
        current_step = render_replay_control(len(df))
        render_chart(df, current_step)

    with col2:
        # TODO(P8): wire up ui_panels.render_log_panel(...)
        st.subheader("Agent Activity Log")
        st.info(
            "TODO(P8): wire up "
            "ui_panels.render_log_panel(log_df, current_step)"
        )

        # TODO(P8): wire up ui_panels.render_recommendation_panel(...)
        st.subheader("Recommendations")
        st.info(
            "TODO(P8): wire up "
            "ui_panels.render_recommendation_panel(log_df, current_step)"
        )

    st.divider()

    st.subheader("What-If Simulator")

    # TODO(P8): wire up whatif.render_whatif_simulator(...)
    st.info(
        "TODO(P8): wire up whatif.render_whatif_simulator(...)"
    )


if __name__ == "__main__":
    main()