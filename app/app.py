"""
P7 — App Shell, Chart, Replay Control
Owns: this file.
Contract: load_pipeline_outputs(), render_chart(), render_replay_control()

Run with:
streamlit run app/app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go


@st.cache_data
def load_pipeline_outputs() -> pd.DataFrame:
    """Load the generated P5 pipeline output for the Streamlit app."""

    result = pd.read_csv("reports/pipeline_outputs.csv")

    result["timestamp"] = pd.to_datetime(result["timestamp"])

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

    # Load generated P5 pipeline output
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