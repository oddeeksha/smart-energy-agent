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
    "peak": "Forecast exceeds seasonal 95th-percentile threshold. Initiate demand-response load shedding to prevent capacity charges.",
    "high_anomaly": "Unusually high residual detected above rolling 30-day baseline. Dispatch field verification for localized surges.",
    "low_anomaly": "Unusually low demand detected below rolling 30-day baseline. Verify telemetry and check for potential feeder outages.",
    "combined": "Critical event: Concurrent peak demand and high anomaly detected. Immediately deploy demand-response protocols and dispatch grid operations.",
}



def get_recommendation(
    trigger_type: str,
    severity: str,
    forecast: float = 0.0,
    threshold: float = 0.0,
    estimated_impact: float = 0.0,
    temperature: float = None,
    hour: int = None,
    season: str = None,
    anomaly_score: float = None,
    is_holiday=None,
) -> Recommendation:
    """Generates concise, severity-dependent operator action recommendations using pipeline-derived quantities.
    
    Avoids duplicating reasoning text (thresholds, percentages, severity labels, weather context) shown in the activity log.
    """
    excess_demand = max(forecast - threshold, 0.0) if (threshold > 0 and forecast > threshold) else 0.0

    if trigger_type == "peak":
        if severity == "High":
            action = "Initiate commercial demand-response contracts and prepare reserve generation capacity."
        elif severity == "Medium":
            action = "Pre-stage demand-response resources and issue voluntary conservation notifications."
        else:
            action = "Increase grid monitoring and prepare demand-response activation if conditions worsen."

        if excess_demand > 0 and estimated_impact > 0:
            action_text = f"{action} Excess demand: {excess_demand:,.0f} MW | Financial exposure: ${estimated_impact:,.0f}."
        elif excess_demand > 0:
            action_text = f"{action} Excess demand: {excess_demand:,.0f} MW."
        else:
            action_text = action

    elif trigger_type == "combined":
        action = "Activate emergency demand-response protocols and dispatch field verification teams."
        if excess_demand > 0 and estimated_impact > 0:
            action_text = f"{action} Excess demand: {excess_demand:,.0f} MW | Financial exposure: ${estimated_impact:,.0f}."
        elif excess_demand > 0:
            action_text = f"{action} Excess demand: {excess_demand:,.0f} MW."
        else:
            action_text = action

    elif trigger_type == "high_anomaly":
        if severity == "High":
            action_text = "Dispatch operations team for situational assessment and inspect localized load spikes."
        elif severity == "Medium":
            action_text = "Review feeder-level consumption and monitor for localized overload conditions."
        else:
            action_text = "Continue standard grid monitoring for minor demand deviation."

    elif trigger_type == "low_anomaly":
        action_text = "Verify telemetry integrity and inspect SCADA feeds for potential feeder outages."

    else:
        action_text = "Initiate standard grid monitoring protocols."

    rate_config = load_rate_config()

    return Recommendation(
        trigger_type=trigger_type,
        action_text=action_text,
        estimated_impact=estimated_impact,
        impact_source_note=rate_config.get("source", "PJM Market Monitor 2024"),
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
