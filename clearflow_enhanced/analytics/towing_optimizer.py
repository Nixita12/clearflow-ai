"""
towing_optimizer.py  —  Module P: AI-Powered Towing Priority Optimizer
---------------------------------------------------------------------------
Given a set of currently-active illegal-parking violations, decides
WHICH vehicles should be towed FIRST when tow-truck capacity is limited.

Priority is NOT just impact_score — it specifically weighs:
  - Lane blockage severity (full lane vs partial vs shoulder)
  - Congestion impact (from CongestionImpactRecord)
  - Emergency-route interference (from emergency_corridor)
  - Duration already parked (longer = lower marginal urgency to wait further,
    but also signals the owner isn't returning soon)

Output: a ranked tow dispatch queue with ETA-aware truck assignment.
"""

from dataclasses import dataclass
from typing import List, Optional

from analytics.emergency_corridor import check_emergency_corridor


# Lane blockage severity by vehicle type (heuristic — larger vehicles
# more likely to fully block a lane vs encroach on shoulder)
LANE_BLOCKAGE_SEVERITY = {
    "HGV": 1.0, "TANKER": 1.0, "LORRY": 1.0, "BUS": 1.0, "PRIVATE_BUS": 1.0,
    "LGV": 0.85, "MAXI_CAB": 0.75, "VAN": 0.65,
    "CAR": 0.55, "PASSENGER_AUTO": 0.45,
    "MOTOR_CYCLE": 0.20, "SCOOTER": 0.20, "MOPED": 0.15,
    "DEFAULT": 0.50,
}


@dataclass
class TowCandidate:
    event_id: str
    vehicle_type: str
    zone_name: str
    impact_score: float
    duration_seconds: float
    lane_blockage_score: float       # 0-1
    emergency_interference: bool
    emergency_delay_min: float
    tow_priority_score: float        # final ranking score
    tow_rank: int
    rationale: str


def compute_tow_priority(
    event_id: str,
    vehicle_type: str,
    zone_name: str,
    impact_score: float,
    duration_seconds: float,
    lat: float,
    lon: float,
    congestion_index: float,
    speed_penalty_kmh: float,
    zone_id: str = "",
) -> TowCandidate:
    """Compute a single vehicle's tow priority score (0-100)."""

    lane_score = LANE_BLOCKAGE_SEVERITY.get(
        vehicle_type.replace(" ", "_").replace("-", "_"),
        LANE_BLOCKAGE_SEVERITY["DEFAULT"],
    )

    # Emergency corridor check
    emerg_alerts = check_emergency_corridor(
        zone_id=zone_id, zone_name=zone_name, lat=lat, lon=lon,
        congestion_index=congestion_index, speed_penalty_kmh=speed_penalty_kmh,
    )
    has_emergency = len(emerg_alerts) > 0
    max_delay = max((a.estimated_delay_min for a in emerg_alerts), default=0.0)

    # Duration factor: vehicles parked very long are less likely to be
    # "just stepping out" — slightly raises tow urgency (capped)
    duration_factor = min(1.0, duration_seconds / 900.0)   # saturates at 15 min

    # ── Weighted priority score ──────────────────────────────────────────────
    score = (
        impact_score        * 0.35 +
        lane_score * 100     * 0.30 +
        (max_delay * 10)      * 0.25 +   # emergency delay heavily weighted, capped via clamp below
        duration_factor * 100 * 0.10
    )
    score = round(min(100.0, score), 1)

    rationale_parts = [f"Impact {impact_score:.0f}/100", f"lane blockage {lane_score*100:.0f}%"]
    if has_emergency:
        rationale_parts.append(f"⚠ emergency route delay +{max_delay:.1f}min")
    rationale_parts.append(f"parked {duration_seconds/60:.0f}min")

    return TowCandidate(
        event_id=event_id, vehicle_type=vehicle_type, zone_name=zone_name,
        impact_score=impact_score, duration_seconds=duration_seconds,
        lane_blockage_score=round(lane_score, 2),
        emergency_interference=has_emergency,
        emergency_delay_min=round(max_delay, 1),
        tow_priority_score=score, tow_rank=0,
        rationale=" | ".join(rationale_parts),
    )


def optimize_tow_queue(candidates: List[TowCandidate], n_trucks: int = 2) -> List[TowCandidate]:
    """
    Rank all candidates by tow priority and assign rank order.
    n_trucks informs how many can be dispatched in the first wave
    (used by caller for ETA display, not encoded into the score itself).
    """
    ranked = sorted(candidates, key=lambda c: c.tow_priority_score, reverse=True)
    for i, c in enumerate(ranked):
        c.tow_rank = i + 1
    return ranked


def print_tow_queue(candidates: List[TowCandidate], n_trucks: int = 2):
    ranked = optimize_tow_queue(candidates, n_trucks)
    print(f"\n  ── TOW DISPATCH QUEUE ({n_trucks} trucks available) ──")
    print(f"  {'#':<3} {'Event':<14} {'Vehicle':<14} {'Zone':<22} {'Score':>6} {'Wave'}")
    print(f"  {'─'*3} {'─'*14} {'─'*14} {'─'*22} {'─'*6} {'─'*6}")
    for c in ranked:
        wave = (c.tow_rank - 1) // n_trucks + 1
        flag = " ⚠" if c.emergency_interference else ""
        print(
            f"  {c.tow_rank:<3} {c.event_id:<14} {c.vehicle_type:<14} "
            f"{c.zone_name[:21]:<22} {c.tow_priority_score:>5.1f} wave-{wave}{flag}"
        )
        print(f"      {c.rationale}")
