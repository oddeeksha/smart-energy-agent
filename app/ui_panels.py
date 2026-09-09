"""
P8 (part 1/2) — UI Panels
Owns: this file. Contract: render_log_panel(), render_recommendation_panel()
Called from app/app.py — coordinate with P7 on layout/columns.
"""
import streamlit as st
import pandas as pd

_SEVERITY_COLOR = {"Low": "blue", "Medium": "orange", "High": "red"}


def render_log_panel(log_df: pd.DataFrame, current_step: int) -> None:
    """Shows log entries up to current_step, most recent first,
    color-coded by severity.

    TODO(P8): log_df is expected to have columns: timestamp, trigger_type,
    severity, reasoning, recommendation, estimated_impact — matching LogEntry.
    Currently accepts a placeholder df with the same shape for standalone testing.
    """
    st.subheader("Agent Activity Log")

    visible = log_df.iloc[:current_step + 1]
    triggered = visible[visible["trigger_type"].notna()]

    if triggered.empty:
        st.write("No events triggered yet.")
        return

    for _, row in triggered[::-1].iterrows():
        color = _SEVERITY_COLOR.get(row.get("severity"), "gray")
        st.markdown(
            f"**{row['timestamp']}** — "
            f":{color}[{row.get('severity', 'N/A')}] {row.get('reasoning', '')}"
        )


def render_recommendation_panel(log_df: pd.DataFrame, current_step: int) -> None:
    """Shows the most recent active recommendation with its estimated impact."""
    st.subheader("Recommendations")

    visible = log_df.iloc[:current_step + 1]
    triggered = visible[visible["trigger_type"].notna()]

    if triggered.empty:
        st.write("No active recommendations.")
        return

    latest = triggered.iloc[-1]
    st.write(latest.get("recommendation", "N/A"))
    impact = latest.get("estimated_impact")
    if impact is not None:
        st.metric("Estimated impact", f"${impact:,.0f}")


if __name__ == "__main__":
    # TODO(P8): standalone smoke test — build a small fake log_df and call
    # both functions inside a minimal st app to confirm rendering works
    # before full integration with P5/P7.
    pass
