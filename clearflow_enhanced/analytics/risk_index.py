"""
risk_index.py  —  Module L: Urban Mobility Risk Index (UMRI)
----------------------------------------------------------------
Single executive-level 0–100 score per zone combining:
  - Parking violation rate     (from hotspot_intelligence)
  - Current congestion index   (from flow_analyser)
  - Time-of-day risk weight    (from config)
  - Emergency corridor exposure (from emergency_corridor)
  - Historical chronic pattern (from hotspot_intelligence)

This is the single number a traffic commander glances at each morning:

    Zone           Risk
    Metro Hub      95
    Stadium        90
    Market Area    78

One metric → instant prioritisation across the whole city.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional

from analytics.hotspot_intelligence import DATASET_HOTSPOT_PROFILES
from analytics.emergency_corridor import EMERGENCY_FACILITIES, _haversine_km


@dataclass
class RiskScore:
    zone_id: str
    zone_name: str
    umri_score: float            # 0–100, higher = more urgent
    band: str                    # CRITICAL / HIGH / ELEVATED / MODERATE / LOW
    component_violation: float   # contribution from violation rate
    component_congestion: float  # contribution from live CI
    component_chronic: float     # contribution from chronic pattern
    component_emergency: float   # contribution from nearby emergency facilities
    component_time: float        # contribution from time-of-day
    top_driver: str              # which component dominates
    narrative: str


# ── Weights (sum to 1.0) — tuned so violation history + live CI dominate ──────
W_VIOLATION   = 0.30
W_CONGESTION  = 0.30
W_CHRONIC     = 0.15
W_EMERGENCY   = 0.15
W_TIME        = 0.10


def _normalise_violation_rate(daily_rate: float, max_rate: float = 110.0) -> float:
    return min(100.0, daily_rate / max_rate * 100.0)


def _emergency_exposure_score(lat: float, lon: float) -> float:
    """0-100: how much emergency-facility exposure this zone has."""
    if not EMERGENCY_FACILITIES:
        return 0.0
    nearest = min(
        EMERGENCY_FACILITIES,
        key=lambda f: _haversine_km(lat, lon, f.lat, f.lon)
    )
    dist = _haversine_km(lat, lon, nearest.lat, nearest.lon)
    if dist >= nearest.critical_radius_km:
        return 0.0
    return round((1.0 - dist / nearest.critical_radius_km) * 100.0, 1)


def compute_risk_index(
    zone_id: str,
    zone_name: str,
    lat: float,
    lon: float,
    live_ci: Optional[float] = None,
    time_of_day_weight: float = 0.5,
) -> RiskScore:
    """
    Compute the Urban Mobility Risk Index for a zone.
    live_ci: pass the current/latest congestion index if available,
             otherwise falls back to historical chronic score proxy.
    """
    profile = DATASET_HOTSPOT_PROFILES.get(zone_id)

    daily_rate    = profile.violations_per_day if profile else 30.0
    chronic_score = profile.chronic_score if profile else 0.5

    viol_component  = _normalise_violation_rate(daily_rate)
    cong_component  = live_ci if live_ci is not None else chronic_score * 60.0
    chronic_component = chronic_score * 100.0
    emerg_component = _emergency_exposure_score(lat, lon)
    time_component  = time_of_day_weight * 100.0

    umri = (
        viol_component   * W_VIOLATION +
        cong_component    * W_CONGESTION +
        chronic_component * W_CHRONIC +
        emerg_component   * W_EMERGENCY +
        time_component     * W_TIME
    )
    umri = round(min(100.0, umri), 1)

    if umri >= 85:   band = "CRITICAL"
    elif umri >= 70: band = "HIGH"
    elif umri >= 50: band = "ELEVATED"
    elif umri >= 30: band = "MODERATE"
    else:            band = "LOW"

    components = {
        "violation history": viol_component * W_VIOLATION,
        "live congestion":   cong_component * W_CONGESTION,
        "chronic pattern":   chronic_component * W_CHRONIC,
        "emergency exposure":emerg_component * W_EMERGENCY,
        "time-of-day":       time_component * W_TIME,
    }
    top_driver = max(components, key=components.get)

    narrative = (
        f"{zone_name}: UMRI {umri}/100 [{band}]. "
        f"Primary driver: {top_driver} "
        f"({round(components[top_driver],1)} of {umri} points)."
    )

    return RiskScore(
        zone_id=zone_id, zone_name=zone_name, umri_score=umri, band=band,
        component_violation=round(viol_component * W_VIOLATION, 1),
        component_congestion=round(cong_component * W_CONGESTION, 1),
        component_chronic=round(chronic_component * W_CHRONIC, 1),
        component_emergency=round(emerg_component * W_EMERGENCY, 1),
        component_time=round(time_component * W_TIME, 1),
        top_driver=top_driver, narrative=narrative,
    )


def compute_citywide_risk_index(
    zones: List[dict],     # list of {"id", "name", "lat", "lon"} from config
    live_ci_by_zone: Optional[Dict[str, float]] = None,
    time_of_day_weight: float = 0.5,
) -> List[RiskScore]:
    """Compute UMRI for all zones and return sorted by risk descending."""
    live_ci_by_zone = live_ci_by_zone or {}
    scores = []
    for z in zones:
        score = compute_risk_index(
            zone_id=z["id"], zone_name=z["name"],
            lat=z["lat"], lon=z["lon"],
            live_ci=live_ci_by_zone.get(z["id"]),
            time_of_day_weight=time_of_day_weight,
        )
        scores.append(score)
    return sorted(scores, key=lambda s: s.umri_score, reverse=True)


def print_risk_table(scores: List[RiskScore]):
    print(f"\n  ── URBAN MOBILITY RISK INDEX (UMRI) ──")
    print(f"  {'Zone':<28} {'UMRI':>6} {'Band':<10} {'Top Driver'}")
    print(f"  {'─'*28} {'─'*6} {'─'*10} {'─'*20}")
    for s in scores:
        print(f"  {s.zone_name:<28} {s.umri_score:>5.1f} {s.band:<10} {s.top_driver}")
