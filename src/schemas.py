"""
Shared data models — Part D of the engineering spec.
DO NOT edit without a full-team conversation. Every module imports from here.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ForecastResult:
    timestamp: str          # ISO 8601 string
    actual: Optional[float] # None if this is a future/unobserved timestep
    predicted: float


@dataclass
class PeakResult:
    is_peak: bool
    threshold: float
    severity: Optional[str]        # "Low" / "Medium" / "High", None if not is_peak
    severity_score: Optional[float]
    reasoning: Optional[str]


@dataclass
class AnomalyResult:
    is_anomaly: bool
    direction: Optional[str]       # "high" or "low", None if not is_anomaly
    zscore: float
    severity: Optional[str]
    severity_score: Optional[float]
    reasoning: Optional[str]


@dataclass
class Recommendation:
    trigger_type: str              # "peak" | "high_anomaly" | "low_anomaly" | "combined"
    action_text: str
    estimated_impact: float
    impact_source_note: str        # citation string, e.g. "PJM Market Monitor 2024"


@dataclass
class LogEntry:
    timestamp: str
    trigger_type: Optional[str]    # None if no trigger fired this step
    severity: Optional[str]
    reasoning: Optional[str]
    recommendation: Optional[str]
    estimated_impact: Optional[float]


# Helper used by P5's combined-trigger logic (max severity comparison).
# Defined here since both P3 and P4's severities get compared in agent.py.
SEVERITY_RANK = {None: -1, "Low": 0, "Medium": 1, "High": 2}


def severity_rank(label: Optional[str]) -> int:
    return SEVERITY_RANK.get(label, -1)
