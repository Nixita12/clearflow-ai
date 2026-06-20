"""
confidence_engine.py  —  Module E: ML Confidence & Breakdown Risk Scoring
--------------------------------------------------------------------------
Produces interpretable confidence scores for:

  1. DETECTION CONFIDENCE  — how certain is YOLO that this is a vehicle
  2. VIOLATION CONFIDENCE  — how certain are we this is genuinely illegal parking
  3. BREAKDOWN RISK SCORE  — probability that congestion worsens to gridlock
                             (derived from CI trend + zone + time-of-day)

Outputs are shown in the dispatch UI alongside the impact score, so
operators can judge whether to dispatch or monitor.

Confidence bands:
  ≥ 85%  → HIGH confidence   (green)
  60–84% → MODERATE          (amber)
  < 60%  → LOW               (red) — human review recommended
"""

from dataclasses import dataclass
import math


@dataclass
class ConfidenceReport:
    detection_confidence_pct: float    # YOLO model raw conf * calibration
    violation_confidence_pct: float    # P(genuine illegal parking)
    breakdown_risk_pct: float          # P(congestion escalates to gridlock)
    detection_band: str                # HIGH / MODERATE / LOW
    violation_band: str
    breakdown_band: str
    confidence_narrative: str          # one-liner for dispatch UI


def _band(pct: float) -> str:
    if pct >= 85:
        return "HIGH"
    if pct >= 60:
        return "MODERATE"
    return "LOW"


def compute_confidence(
    yolo_conf: float,                  # 0.0–1.0 raw YOLO confidence
    frames_stationary: int,            # how many consecutive frames vehicle was still
    stationary_threshold_s: float,     # config threshold (e.g. 60s)
    duration_sec: float,               # actual parking duration
    congestion_index: float,           # current CI 0–100
    zone_weight: float,                # 0–1, zone risk weight
    time_of_day_weight: float,         # 0–1, from config
    congestion_trend: float = 0.0,     # positive = CI rising (pass delta CI/dt)
    vehicle_type: str = "CAR",
) -> ConfidenceReport:
    """
    Compute detection, violation, and breakdown confidence scores.

    All outputs are percentages (0–100).
    """

    # ── 1. Detection Confidence ───────────────────────────────────────────────
    # YOLO conf directly, with a slight penalty for small/occluded classes
    SIZE_PENALTY = {"SCOOTER": 0.05, "MOPED": 0.05, "MOTOR_CYCLE": 0.03}
    det_conf = yolo_conf * 100.0 - SIZE_PENALTY.get(vehicle_type, 0.0) * 100.0
    det_conf = round(max(0.0, min(100.0, det_conf)), 1)

    # ── 2. Violation Confidence ────────────────────────────────────────────────
    # Component 1: temporal evidence — how much longer than threshold?
    time_ratio = min(duration_sec / max(stationary_threshold_s, 1.0), 3.0)
    temporal_score = 40.0 * (1.0 - math.exp(-time_ratio))

    # Component 2: frame consistency (more stationary frames = more certain)
    frame_score = min(frames_stationary / 30.0, 1.0) * 30.0

    # Component 3: zone context (known hotspot = higher prior)
    zone_score = zone_weight * 30.0

    viol_conf = round(min(100.0, temporal_score + frame_score + zone_score), 1)

    # ── 3. Breakdown Risk Score ────────────────────────────────────────────────
    # Logistic model: higher CI, worse zone, peak hour → higher risk
    ci_component    = congestion_index / 100.0          # 0–1
    zone_component  = zone_weight                        # 0–1
    time_component  = time_of_day_weight                 # 0–1
    trend_component = max(0.0, min(1.0, congestion_trend / 20.0))  # normalise trend

    # Weighted logistic
    x = (
        ci_component   * 0.50 +
        zone_component * 0.25 +
        time_component * 0.15 +
        trend_component * 0.10
    )
    breakdown_risk = round(100.0 / (1.0 + math.exp(-8.0 * (x - 0.5))), 1)

    # ── Narrative ─────────────────────────────────────────────────────────────
    narrative = (
        f"Detection confidence: {det_conf}% | "
        f"Violation confidence: {viol_conf}% | "
        f"Breakdown risk: {breakdown_risk}%"
    )

    return ConfidenceReport(
        detection_confidence_pct = det_conf,
        violation_confidence_pct = viol_conf,
        breakdown_risk_pct       = breakdown_risk,
        detection_band           = _band(det_conf),
        violation_band           = _band(viol_conf),
        breakdown_band           = _band(breakdown_risk),
        confidence_narrative     = narrative,
    )
