"""
P8 — UI Panels
Owns: this file. Contract: render_log_panel()
Called from app/app.py — coordinate with P7 on layout/columns.
"""
import streamlit as st
import pandas as pd

_SEVERITY_COLOR = {"Low": "blue", "Medium": "orange", "High": "red"}


def render_log_panel(log_df: pd.DataFrame, current_step: int) -> None:
    """Shows log entries up to current_step, most recent first,
    color-coded by severity inside a fixed-height scroll container.
    """
    st.subheader("Agent Activity Log")

    visible = log_df.iloc[:current_step + 1]
    triggered = visible[visible["trigger_type"].notna()]

    if triggered.empty:
        st.write("No events triggered yet.")
        return

    with st.container(height=520):
        for _, row in triggered[::-1].iterrows():
            color = _SEVERITY_COLOR.get(row.get("severity"), "gray")
            st.markdown(
                f"**{row['timestamp']}** — "
                f":{color}[{row.get('severity', 'N/A')}] {row.get('reasoning', '')}"
            )


def _metric_card(label: str, value: str) -> None:
    """Renders a compact, non-truncating metric card with text wrapping."""
    st.markdown(
        f"""
        <div style="
            background-color: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 8px;
            padding: 8px 12px;
            margin-bottom: 8px;
        ">
            <div style="
                font-size: 0.78rem;
                font-weight: 500;
                color: #b0b8c0;
                line-height: 1.25;
                word-wrap: break-word;
                white-space: normal;
            ">{label}</div>
            <div style="
                font-size: 1.1rem;
                font-weight: 700;
                color: #f0f4f8;
                margin-top: 3px;
                word-wrap: break-word;
                white-space: normal;
            ">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_event_inspector(df: pd.DataFrame) -> None:
    """Renders the Event Inspector panel for the user-selected point on the chart."""
    st.subheader("🔍 Event Inspector")

    selected_ts = st.session_state.get("selected_inspector_ts")

    if not selected_ts:
        st.info("Select a point on the chart to inspect event details.")
        return

    # Match row in df
    selected_dt = pd.to_datetime(selected_ts)
    match = df[df["timestamp"] == selected_dt]

    if match.empty:
        match = df[df["timestamp"].astype(str) == str(selected_ts)]

    if match.empty:
        st.info("Select a point on the chart to inspect event details.")
        return

    row = match.iloc[0]

    ts_str = pd.to_datetime(row["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
    trig_raw = row.get("trigger_type")
    trig_type = str(trig_raw).replace("_", " ").upper() if pd.notna(trig_raw) else "NORMAL"
    sev_raw = row.get("severity")
    severity = str(sev_raw).upper() if pd.notna(sev_raw) else "NORMAL"

    # Status callout banner
    if severity == "HIGH":
        st.error(f"🚨 **[{severity} SEVERITY] {trig_type} DETECTED**")
    elif severity == "MEDIUM":
        st.warning(f"⚠️ **[{severity} SEVERITY] {trig_type} DETECTED**")
    elif severity == "LOW":
        st.info(f"ℹ️ **[{severity} SEVERITY] {trig_type} DETECTED**")
    else:
        st.success("🟢 **GRID OPERATING NORMALLY**")

    st.markdown(f"📍 **Inspected Timestamp**: `{ts_str}`")

    # Metrics computation
    actual = row.get("actual")
    pred = row.get("predicted")
    thresh = row.get("peak_threshold")

    excess_mw = (pred - thresh) if (pd.notna(pred) and pd.notna(thresh) and thresh > 0 and pred > thresh) else 0.0
    pct_exceed = (excess_mw / thresh * 100.0) if (thresh and thresh > 0 and excess_mw > 0) else 0.0

    temp = row.get("temperature")
    hol = row.get("is_holiday")
    hol_str = "Public Holiday" if (pd.notna(hol) and (hol == 1 or hol is True)) else "Normal Day"

    impact = row.get("estimated_impact")
    impact_str = f"${impact:,.0f}" if (pd.notna(impact) and impact > 0) else "$0"

    actual_str = f"{actual:,.0f} MW" if pd.notna(actual) else "N/A"
    pred_str = f"{pred:,.0f} MW" if pd.notna(pred) else "N/A"
    thresh_str = f"{thresh:,.0f} MW" if (pd.notna(thresh) and thresh > 0) else "N/A"
    excess_str = f"{excess_mw:,.0f} MW" if excess_mw > 0 else "0 MW"
    pct_str = f"+{pct_exceed:.1f}%" if pct_exceed > 0 else "0.0%"
    temp_str = f"{temp:.1f}°C" if pd.notna(temp) else "N/A"

    # Preferred 2-column layout
    c1, c2 = st.columns(2)
    with c1:
        _metric_card("Actual Demand", actual_str)
        _metric_card("Forecast Demand", pred_str)
        _metric_card("Seasonal Threshold", thresh_str)
        _metric_card("Excess Demand", excess_str)
        _metric_card("Percentage Exceedance", pct_str)

    with c2:
        _metric_card("Temperature", temp_str)
        _metric_card("Holiday Status", hol_str)
        _metric_card("Severity", severity)
        _metric_card("Trigger Type", trig_type)
        _metric_card("Estimated Financial Impact", impact_str)

    # Dedicated Reasoning & Explanation Section
    reasoning = row.get("reasoning")
    if pd.notna(reasoning) and reasoning:
        st.markdown("#### **Reasoning & Explanation**")
        st.write(reasoning)

    if st.button("❌ Clear Inspector Selection", use_container_width=True):
        st.session_state["selected_inspector_ts"] = None
        st.rerun()


if __name__ == "__main__":
    pass
