"""
P7 — App Shell, Chart, Replay Control
Owns: this file.
Contract: load_pipeline_outputs(), render_chart(), render_replay_control()

Run with:
streamlit run app/app.py
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from ui_panels import render_log_panel, render_event_inspector


@st.cache_data
def _load_pipeline_outputs_cached(mtime: float) -> pd.DataFrame:
    result = pd.read_csv("reports/pipeline_outputs.csv")
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    return result


def load_pipeline_outputs() -> pd.DataFrame:
    """Load the generated P5 pipeline output for the Streamlit app, invalidating cache when file updates on disk."""
    csv_path = "reports/pipeline_outputs.csv"
    mtime = os.path.getmtime(csv_path) if os.path.exists(csv_path) else 0.0
    return _load_pipeline_outputs_cached(mtime)



def render_chart(df: pd.DataFrame, current_step: int) -> None:
    """Render actual vs forecast with peak and anomaly markers, ensuring all timestamps emit click events."""

    st.info(
        "ℹ️ **Forecast Horizon (t+1)**: The dotted Forecast line represents a 1-hour-ahead "
        "prediction generated at the displayed timestamp (e.g., the forecast shown at 05:00 is "
        "the model's prediction for demand at 06:00). Peak alerts and recommendations are "
        "therefore issued in advance to support proactive grid operations."
    )

    visible = df.iloc[: current_step + 1]

    fig = go.Figure()

    # Actual demand (lines + markers for 100% clickability across all timesteps)
    fig.add_trace(
        go.Scatter(
            x=visible["timestamp"],
            y=visible["actual"],
            name="Actual",
            mode="lines+markers",
            marker=dict(size=4, opacity=0.7),
            hovertemplate="<b>Actual Demand</b><br>Time: %{x}<br>Demand: %{y:,.0f} MW<extra></extra>",
        )
    )

    # Forecast demand (lines + markers for 100% clickability across all timesteps)
    fig.add_trace(
        go.Scatter(
            x=visible["timestamp"],
            y=visible["predicted"],
            name="Forecast",
            mode="lines+markers",
            line=dict(dash="dot"),
            marker=dict(size=4, opacity=0.7),
            hovertemplate="<b>Forecast Demand</b><br>Time: %{x}<br>Forecast: %{y:,.0f} MW<extra></extra>",
        )
    )

    # Peak events
    peaks = visible[visible["trigger_type"] == "peak"]
    if not peaks.empty:
        fig.add_trace(
            go.Scatter(
                x=peaks["timestamp"],
                y=peaks["actual"],
                mode="markers",
                name="Peak",
                marker=dict(size=10, symbol="triangle-up"),
                hovertemplate="<b>Peak Event</b><br>Time: %{x}<br>Actual: %{y:,.0f} MW<extra></extra>",
            )
        )

    # Anomaly events
    anomalies = visible[
        visible["trigger_type"].isin(
            ["high_anomaly", "low_anomaly"]
        )
    ]
    if not anomalies.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies["timestamp"],
                y=anomalies["actual"],
                mode="markers",
                name="Anomaly",
                marker=dict(size=10, symbol="x"),
                hovertemplate="<b>Anomaly Event</b><br>Time: %{x}<br>Actual: %{y:,.0f} MW<extra></extra>",
            )
        )

    # Combined events
    combined = visible[
        visible["trigger_type"] == "combined"
    ]
    if not combined.empty:
        fig.add_trace(
            go.Scatter(
                x=combined["timestamp"],
                y=combined["actual"],
                mode="markers",
                name="Combined",
                marker=dict(size=12, symbol="star"),
                hovertemplate="<b>Combined Event</b><br>Time: %{x}<br>Actual: %{y:,.0f} MW<extra></extra>",
            )
        )

    fig.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Time",
        yaxis_title="Demand",
    )

    event_data = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
    )

    if event_data and isinstance(event_data, dict) and "selection" in event_data:
        points = event_data["selection"].get("points", [])
        if points:
            clicked_x = points[0].get("x")
            if clicked_x:
                new_ts = str(clicked_x)
                if st.session_state.get("selected_inspector_ts") != new_ts:
                    st.session_state["selected_inspector_ts"] = new_ts
                    st.rerun()


def render_event_navigator(df: pd.DataFrame) -> None:
    """Render sidebar Event Navigator for 1-click event jumping."""

    if "current_step" not in st.session_state:
        st.session_state["current_step"] = 0

    st.sidebar.header("🎯 Event Navigator")

    # Quick Filters
    filter_options = [
        "All Events",
        "High Peaks",
        "Medium Peaks",
        "Low Peaks",
        "Anomalies",
        "Combined Events",
    ]
    selected_filter = st.sidebar.selectbox("Filter Events", filter_options)

    # Filter events
    events_df = df[df["trigger_type"].notna()].copy()

    if selected_filter == "High Peaks":
        filtered = events_df[
            (events_df["trigger_type"] == "peak") & (events_df["severity"] == "High")
        ]
    elif selected_filter == "Medium Peaks":
        filtered = events_df[
            (events_df["trigger_type"] == "peak") & (events_df["severity"] == "Medium")
        ]
    elif selected_filter == "Low Peaks":
        filtered = events_df[
            (events_df["trigger_type"] == "peak") & (events_df["severity"] == "Low")
        ]
    elif selected_filter == "Anomalies":
        filtered = events_df[events_df["trigger_type"].str.contains("anomaly", na=False)]
    elif selected_filter == "Combined Events":
        filtered = events_df[events_df["trigger_type"] == "combined"]
    else:
        filtered = events_df

    if filtered.empty:
        st.sidebar.info("No events match the selected filter.")
        return

    event_indices = filtered.index.tolist()

    def format_label(idx):
        row = filtered.loc[idx]
        ts_str = pd.to_datetime(row["timestamp"]).strftime("%Y-%m-%d %H:%M")
        sev = str(row.get("severity", "")).upper()
        trig = str(row.get("trigger_type", "")).replace("_", " ").upper()
        return f"{ts_str} | {sev} {trig}".strip()

    labels = [format_label(idx) for idx in event_indices]
    label_to_index = dict(zip(labels, event_indices))

    cur_step = st.session_state["current_step"]

    # Prev / Next event buttons
    prev_candidates = [idx for idx in event_indices if idx < cur_step]
    next_candidates = [idx for idx in event_indices if idx > cur_step]

    col_prev, col_next = st.sidebar.columns(2)
    with col_prev:
        if st.button(
            "◀ Prev Event",
            use_container_width=True,
            disabled=len(prev_candidates) == 0,
        ):
            st.session_state["current_step"] = prev_candidates[-1]
            st.rerun()

    with col_next:
        if st.button(
            "Next Event ▶",
            use_container_width=True,
            disabled=len(next_candidates) == 0,
        ):
            st.session_state["current_step"] = next_candidates[0]
            st.rerun()

    # One-shot action jump selectbox
    placeholder = "-- Select event to jump --"
    dropdown_options = [placeholder] + labels

    def on_jump_select():
        chosen = st.session_state.event_jump_selectbox
        if chosen != placeholder and chosen in label_to_index:
            st.session_state["current_step"] = label_to_index[chosen]

    st.sidebar.selectbox(
        "Jump to Event",
        dropdown_options,
        index=0,
        key="event_jump_selectbox",
        on_change=on_jump_select,
    )

    st.sidebar.caption(
        f"📍 **Active Replay Step**: `{cur_step}` / `{len(df) - 1}`"
    )




def render_replay_control(df: pd.DataFrame) -> int:
    """Render timestamp display and replay slider synchronized with session_state."""

    if "current_step" not in st.session_state:
        st.session_state["current_step"] = 0

    cur_step = st.session_state["current_step"]
    cur_ts = pd.to_datetime(df.iloc[cur_step]["timestamp"]).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    st.markdown(
        f"⏱️ **Current Replay Timestamp**: `{cur_ts}` &nbsp;|&nbsp; "
        f"**Step**: `{cur_step} / {len(df) - 1}`"
    )

    return st.slider(
        "Replay progress",
        0,
        len(df) - 1,
        key="current_step",
    )


def main():
    st.set_page_config(
        layout="wide",
        page_title="GridPulse: AI-Powered Energy Demand Forecasting, Peak Detection, and Grid Monitoring",
    )

    st.title("GridPulse: AI-Powered Energy Demand Forecasting, Peak Detection, and Grid Monitoring")

    st.caption(
        "Grid-operator decision-support system — PJM regional demand"
    )

    # Load generated P5 pipeline output
    df = load_pipeline_outputs()

    # Render Event Navigator in sidebar
    render_event_navigator(df)

    col1, col2 = st.columns([1.6, 1.0])

    with col1:
        current_step = render_replay_control(df)
        render_chart(df, current_step)

    with col2:
        render_event_inspector(df)
        render_log_panel(df, current_step)


if __name__ == "__main__":
    main()