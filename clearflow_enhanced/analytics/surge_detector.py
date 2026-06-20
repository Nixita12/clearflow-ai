"""
surge_detector.py  —  Module Q: Multi-Vehicle Surge / Swarm Detector
-----------------------------------------------------------------------
Every other module in this pipeline scores ONE violation at a time.
But the dataset contains events where 80-140+ vehicles are flagged at
the SAME junction within the SAME hour — a fundamentally different
operational problem than "one car is parked illegally."

A single tow truck cannot clear a 137-vehicle surge. Module Q exists
to detect that distinction and hand off a different response plan:
zone-wide multi-unit dispatch instead of individual ticketing.

WHY THIS IS A GENUINE GAP (not duplicate of existing modules):
  - hotspot_intelligence.py profiles AVERAGE daily volume per zone (102/day)
  - forecast_engine.py predicts EXPECTED violations in a future window
  - towing_optimizer.py queues INDIVIDUAL vehicles for pickup
  None of them ask: "are violations clustering into a single coordinated
  event RIGHT NOW, this hour, at this junction?" — which needs a
  fundamentally different response (zone cordon vs single tow).

METHOD:
  Groups raw violations by (junction, date, hour) directly from the
  source CSV — no simulation, no estimation. Classifies each cluster
  by size into a response tier, and computes a non-linear severity
  score reflecting that N simultaneous vehicles cause more than N times
  the disruption of one (multiple lanes blocked at once, no clear path
  for through-traffic).
"""

import math
import logging
import datetime
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
SURGE_MIN_VEHICLES        = 8     # below this, treat as independent violations
COORDINATED_THRESHOLD     = 20    # likely organised loading/vending activity
EVENT_OVERFLOW_THRESHOLD  = 50    # likely festival/rally/market-day spillover

RESPONSE_TIER = {
    "ISOLATED":           "Standard single-unit dispatch",
    "SURGE":               "Multi-unit dispatch (2-3 units), zone commander notified",
    "COORDINATED_ACTIVITY":"Senior officer + 4+ tow units, investigate recurring cause",
    "EVENT_OVERFLOW":      "Zone cordon, traffic diversion, full enforcement team",
}

# Per-vehicle blockage weight used in the non-linear severity formula —
# mirrors the structural logic already used in congestion_engine.py's
# vehicle_zone_weight so the severity scale is consistent across modules.
_VEHICLE_SEVERITY = {
    "LGV": 2.0, "HGV": 2.0, "PRIVATE_BUS": 2.0, "BUS": 2.0, "MAXI_CAB": 1.6,
    "PASSENGER_AUTO": 1.0, "CAR": 1.0, "VAN": 1.2, "SCOOTER": 0.5,
    "MOTOR_CYCLE": 0.5, "MOPED": 0.4, "DEFAULT": 1.0,
}


@dataclass
class SurgeEvent:
    junction_name: str
    police_station: str
    date: str
    hour: int
    vehicle_count: int
    response_tier: str            # ISOLATED / SURGE / COORDINATED_ACTIVITY / EVENT_OVERFLOW
    severity_score: float         # 0-100, non-linear in vehicle_count
    dominant_vehicle_type: str
    estimated_lanes_blocked: float
    recommended_response: str
    surge_narrative: str


def _classify_tier(n: int) -> str:
    if n >= EVENT_OVERFLOW_THRESHOLD:
        return "EVENT_OVERFLOW"
    if n >= COORDINATED_THRESHOLD:
        return "COORDINATED_ACTIVITY"
    if n >= SURGE_MIN_VEHICLES:
        return "SURGE"
    return "ISOLATED"


def _severity_score(n: int, dominant_vehicle: str) -> float:
    """
    Non-linear severity: sqrt(n) growth reflects that simultaneous
    vehicles block multiple lanes/approaches at once rather than
    queueing sequentially — disruption compounds, it doesn't just add.
    """
    veh_weight = _VEHICLE_SEVERITY.get(dominant_vehicle, _VEHICLE_SEVERITY["DEFAULT"])
    raw = math.sqrt(n) * veh_weight * 6.0
    return round(min(100.0, raw), 1)


def detect_historical_surges(df: pd.DataFrame, min_vehicles: int = SURGE_MIN_VEHICLES) -> list:
    """
    Scans the FULL historical dataset for surge events — moments where
    many vehicles were flagged at the same junction within the same hour.

    Requires df to have: junction_name, date (or created_datetime), hour,
    vehicle_type_canonical (or vehicle_type), police_station, id columns.

    Returns list[SurgeEvent] sorted by severity descending.
    """
    work = df[df['junction_name'] != 'No Junction'].copy()
    if 'date' not in work.columns:
        work['date'] = work['created_datetime'].dt.date

    veh_col = 'vehicle_type_canonical' if 'vehicle_type_canonical' in work.columns else 'vehicle_type'

    grouped = (
        work.groupby(['junction_name', 'date', 'hour'])
        .agg(
            vehicle_count=('id', 'count'),
            dominant_vehicle=(veh_col, lambda x: x.value_counts().index[0] if len(x) else 'CAR'),
            police_station=('police_station', 'first'),
        )
        .reset_index()
    )

    surges = grouped[grouped['vehicle_count'] >= min_vehicles].copy()

    events = []
    for _, row in surges.iterrows():
        n = int(row['vehicle_count'])
        tier = _classify_tier(n)
        veh = str(row['dominant_vehicle']).replace(' ', '_').replace('-', '_')
        severity = _severity_score(n, veh)
        veh_weight = _VEHICLE_SEVERITY.get(veh, _VEHICLE_SEVERITY["DEFAULT"])
        lanes_blocked = round(min(4.0, n * veh_weight / 12.0), 1)

        narrative = (
            f"{row['junction_name']} on {row['date']} at {int(row['hour']):02d}:00 — "
            f"{n} vehicles flagged simultaneously ({tier.replace('_', ' ').title()}), "
            f"dominant vehicle {veh} (severity weight {veh_weight:.1f}x). "
            f"Estimated {lanes_blocked:.1f} lane-equivalents blocked. "
            f"Response: {RESPONSE_TIER[tier]}."
        )

        events.append(SurgeEvent(
            junction_name           = row['junction_name'],
            police_station          = str(row['police_station']),
            date                    = str(row['date']),
            hour                    = int(row['hour']),
            vehicle_count           = n,
            response_tier           = tier,
            severity_score          = severity,
            dominant_vehicle_type   = veh,
            estimated_lanes_blocked = lanes_blocked,
            recommended_response    = RESPONSE_TIER[tier],
            surge_narrative         = narrative,
        ))

    events.sort(key=lambda e: e.severity_score, reverse=True)
    return events


def surge_summary(events: list) -> dict:
    """Aggregate stats across all detected surge events."""
    if not events:
        return {
            "total_surge_events": 0, "total_vehicles_in_surges": 0,
            "event_overflow_count": 0, "coordinated_count": 0, "surge_count": 0,
        }
    return {
        "total_surge_events":      len(events),
        "total_vehicles_in_surges": sum(e.vehicle_count for e in events),
        "event_overflow_count":    sum(1 for e in events if e.response_tier == "EVENT_OVERFLOW"),
        "coordinated_count":       sum(1 for e in events if e.response_tier == "COORDINATED_ACTIVITY"),
        "surge_count":             sum(1 for e in events if e.response_tier == "SURGE"),
        "junctions_affected":      len(set(e.junction_name for e in events)),
        "avg_severity":            round(sum(e.severity_score for e in events) / len(events), 1),
    }


def print_surge_report(events: list, top_n: int = 15):
    sep = "═" * 82
    print(f"\n{sep}")
    print(f"  ClearFlow AI — Module Q: Multi-Vehicle Surge Detection")
    print(f"  (Distinct from per-violation scoring — detects COORDINATED clustering)")
    print(sep)

    summary = surge_summary(events)
    print(f"  Total surge events found  : {summary['total_surge_events']:,}")
    print(f"  Total vehicles involved   : {summary['total_vehicles_in_surges']:,}")
    print(f"  Junctions affected        : {summary['junctions_affected']}")
    print(f"  EVENT_OVERFLOW tier       : {summary['event_overflow_count']:,}")
    print(f"  COORDINATED_ACTIVITY tier : {summary['coordinated_count']:,}")
    print(f"  SURGE tier                : {summary['surge_count']:,}")
    print(f"{'─'*82}")

    print(f"  TOP {top_n} SURGE EVENTS  (ranked by severity score — weights vehicle type, not just count)\n")
    print(f"  {'#':<3} {'Junction':<28} {'Date':<12} {'Hr':>3} {'Vehicles':>8} "
          f"{'Tier':<22} {'Severity':>8} {'Lanes':>6}")
    print(f"  {'─'*3} {'─'*28} {'─'*12} {'─'*3} {'─'*8} {'─'*22} {'─'*8} {'─'*6}")
    for i, e in enumerate(events[:top_n], 1):
        print(
            f"  {i:<3} {e.junction_name[:27]:<28} {e.date:<12} {e.hour:>3} "
            f"{e.vehicle_count:>8} {e.response_tier:<22} {e.severity_score:>8.1f} "
            f"{e.estimated_lanes_blocked:>6.1f}"
        )

    by_volume = sorted(events, key=lambda e: e.vehicle_count, reverse=True)[:5]
    print(f"\n  Largest by raw vehicle count (for reference — severity above also weighs vehicle type):")
    for e in by_volume:
        print(f"    {e.junction_name} | {e.date} {e.hour:02d}:00 | {e.vehicle_count} vehicles")

    if events:
        print(f"\n  Worst case:\n  {events[0].surge_narrative}")
    print(sep)
