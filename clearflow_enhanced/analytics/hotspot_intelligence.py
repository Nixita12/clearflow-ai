"""
hotspot_intelligence.py  —  Module G: Hotspot Risk Intelligence
---------------------------------------------------------------
Answers: "Which zones need a PERMANENT enforcement post vs temporary patrol?"

Derives from historical dataset:
  - Violation velocity (violations per hour, trending up/down)
  - Chronic vs acute hotspot classification
  - Enforcement ROI score (impact per patrol unit deployed)
  - Recommended enforcement mode: FIXED_POST / PATROL / CAMERA / MONITOR

This directly addresses the problem statement gap:
  "No heatmap of parking violations vs congestion impact.
   Difficult to prioritize enforcement zones."
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class HotspotProfile:
    zone_id: str
    zone_name: str
    violation_count: int            # historical total
    violations_per_day: float       # avg daily rate
    peak_hour: int                  # hour with most violations (0–23)
    peak_day: str                   # day of week with most violations
    chronic_score: float            # 0–1: how consistently bad (vs spike-only)
    enforcement_roi: float          # impact_score / patrol_cost proxy
    recommended_mode: str           # FIXED_POST / MOBILE_PATROL / SPEED_CAMERA / MONITOR
    mode_rationale: str
    top_vehicle_type: str           # most common offending vehicle
    top_offence: str                # most common offence code


# Pre-computed from dataset (298,450 records, Jan–May 2024)
# These would normally be computed live from DatasetLoader but are
# embedded here for offline use / demo mode
DATASET_HOTSPOT_PROFILES: Dict[str, HotspotProfile] = {
    "BTP051": HotspotProfile(
        zone_id="BTP051", zone_name="Safina Plaza Junction",
        violation_count=15449, violations_per_day=102.3,
        peak_hour=5, peak_day="Wednesday",
        chronic_score=0.91,       # violations spread across ALL hours → chronic
        enforcement_roi=94.2,
        recommended_mode="FIXED_POST",
        mode_rationale="Highest volume (102/day), chronic pattern (91%), "
                       "peak ROI — dedicated post justified.",
        top_vehicle_type="SCOOTER", top_offence="NO_PARKING",
    ),
    "BTP082": HotspotProfile(
        zone_id="BTP082", zone_name="KR Market Junction",
        violation_count=11538, violations_per_day=76.4,
        peak_hour=5, peak_day="Thursday",
        chronic_score=0.84,
        enforcement_roi=78.6,
        recommended_mode="FIXED_POST",
        mode_rationale="Second highest volume, market area with all-day violations.",
        top_vehicle_type="SCOOTER", top_offence="NO_PARKING",
    ),
    "BTP040": HotspotProfile(
        zone_id="BTP040", zone_name="Elite Junction",
        violation_count=10718, violations_per_day=71.0,
        peak_hour=5, peak_day="Friday",
        chronic_score=0.79,
        enforcement_roi=72.1,
        recommended_mode="FIXED_POST",
        mode_rationale="Commercial zone, consistent weekday violations.",
        top_vehicle_type="CAR", top_offence="WRONG_PARKING",
    ),
    "BTP044": HotspotProfile(
        zone_id="BTP044", zone_name="Sagar Theatre Junction",
        violation_count=10549, violations_per_day=69.9,
        peak_hour=5, peak_day="Saturday",
        chronic_score=0.76,
        enforcement_roi=70.8,
        recommended_mode="MOBILE_PATROL",
        mode_rationale="Theatre area — weekend spike pattern, mobile patrol efficient.",
        top_vehicle_type="CAR", top_offence="WRONG_PARKING",
    ),
    "BTP211": HotspotProfile(
        zone_id="BTP211", zone_name="Central Street Junction",
        violation_count=5388, violations_per_day=35.7,
        peak_hour=6, peak_day="Wednesday",
        chronic_score=0.55,
        enforcement_roi=42.3,
        recommended_mode="MOBILE_PATROL",
        mode_rationale="Moderate volume, clustered peaks — patrol covers cost.",
        top_vehicle_type="SCOOTER", top_offence="NO_PARKING",
    ),
    "BTP058": HotspotProfile(
        zone_id="BTP058", zone_name="Subbanna Junction",
        violation_count=5189, violations_per_day=34.4,
        peak_hour=5, peak_day="Tuesday",
        chronic_score=0.52,
        enforcement_roi=40.1,
        recommended_mode="MOBILE_PATROL",
        mode_rationale="Similar to Central Street — shared patrol route feasible.",
        top_vehicle_type="SCOOTER", top_offence="NO_PARKING",
    ),
    "BTP027": HotspotProfile(
        zone_id="BTP027", zone_name="Modi Bridge Junction",
        violation_count=4584, violations_per_day=30.4,
        peak_hour=4, peak_day="Monday",
        chronic_score=0.44,
        enforcement_roi=33.7,
        recommended_mode="SPEED_CAMERA",
        mode_rationale="Bridge approach, low chronic score — ANPR camera cost-effective.",
        top_vehicle_type="TRUCK", top_offence="PARKING_MAIN_ROAD",
    ),
    "BTP020": HotspotProfile(
        zone_id="BTP020", zone_name="Hosahalli Metro Station",
        violation_count=4101, violations_per_day=27.2,
        peak_hour=7, peak_day="Monday",
        chronic_score=0.38,
        enforcement_roi=28.9,
        recommended_mode="SPEED_CAMERA",
        mode_rationale="Metro station — commuter peak only, camera covers off-hours.",
        top_vehicle_type="SCOOTER", top_offence="NO_PARKING",
    ),
}


def get_enforcement_plan(zone_ids: List[str] = None) -> List[HotspotProfile]:
    """
    Returns enforcement profiles sorted by ROI.
    If zone_ids is None, returns all 8 hotspots.
    """
    profiles = list(DATASET_HOTSPOT_PROFILES.values())
    if zone_ids:
        profiles = [p for p in profiles if p.zone_id in zone_ids]
    return sorted(profiles, key=lambda p: p.enforcement_roi, reverse=True)


def print_enforcement_plan():
    """Pretty-print the full enforcement recommendation table."""
    profiles = get_enforcement_plan()
    sep = "═" * 80
    print(f"\n{sep}")
    print("  ClearFlow AI — Zone Enforcement Intelligence Report")
    print(sep)
    print(f"  {'Zone':<30} {'Viol/Day':>8} {'Chronic':>8} {'ROI':>6} {'Mode':<16} {'Peak'}")
    print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*6} {'─'*16} {'─'*12}")
    for p in profiles:
        print(
            f"  {p.zone_name:<30} {p.violations_per_day:>7.1f} "
            f"{p.chronic_score*100:>7.0f}% {p.enforcement_roi:>5.1f} "
            f"{p.recommended_mode:<16} {p.peak_day} {p.peak_hour:02d}:00"
        )
    print(sep)
