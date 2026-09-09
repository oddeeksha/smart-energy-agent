"""
P4 (part 2/2) — Recommendations & Impact Estimation
Owns: this file. Contracts: get_recommendation(trigger_type, severity) -> Recommendation
                             estimate_impact(...) -> float
DO NOT change these signatures — P5 and P8 depend on them directly.
"""
import json

from src.schemas import Recommendation
from src.config import RATE_CONFIG_PATH


_TEMPLATES = {
    "peak": "Recommend initiating demand-response planning.",
    "high_anomaly": "Recommend operator investigation of unexpected high demand.",
    "low_anomaly": "Recommend outage / measurement verification for unexpected low demand.",
    "combined": "Recommend initiating demand-response planning and flagging for operator investigation.",
}


def get_recommendation(trigger_type: str, severity: str) -> Recommendation:
    """trigger_type in {'peak', 'high_anomaly', 'low_anomaly', 'combined'}.

    TODO(P4): estimated_impact and impact_source_note are placeholders here —
    the real values get filled in by the caller (agent.py) using estimate_impact()
    below, since this function alone doesn't have access to demand/threshold values.
    """
    action_text = _TEMPLATES.get(trigger_type, "No recommendation template for this trigger type.")
    rate_config = load_rate_config()
    return Recommendation(
        trigger_type=trigger_type,
        action_text=f"[{severity}] {action_text}",
        estimated_impact=0.0,  # caller overwrites with a real estimate_impact() call
        impact_source_note=rate_config.get("source", "TODO(P4): cite a real rate source"),
    )


def estimate_impact(predicted_demand: float, threshold: float,
                     peak_rate: float, offpeak_rate: float) -> float:
    """max(predicted_demand - threshold, 0) * (peak_rate - offpeak_rate)"""
    reduction_target = max(predicted_demand - threshold, 0)
    return reduction_target * (peak_rate - offpeak_rate)


def load_rate_config(path: str = RATE_CONFIG_PATH) -> dict:
    """TODO(P4): populate impact_rate_config.json with real cited values, e.g.:
    {"peak_rate": 0.14, "offpeak_rate": 0.09, "source": "PJM Market Monitor 2024, $X/MWh spread"}
    """
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"peak_rate": 0.0, "offpeak_rate": 0.0,
                "source": "TODO(P4): rate config file not found, using placeholder zeros"}


if __name__ == "__main__":
    # TODO(P4): create data/artifacts/impact_rate_config.json with real cited values.
    pass
