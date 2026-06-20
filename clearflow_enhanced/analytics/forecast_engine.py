"""
forecast_engine.py  —  Module H: Violation & Congestion Forecasting
--------------------------------------------------------------------
Answers: "When and where will the NEXT congestion spike happen?"

Uses historical hourly + day-of-week distribution from the dataset
to forecast violation probability and expected congestion index
for any zone × hour × day combination.

This directly fills the problem statement gap:
  "Event impact is not quantified in advance.
   Resource deployment is experience-driven."

Outputs:
  - Violation probability for next N hours per zone
  - Expected congestion index forecast
  - Recommended pre-deployment windows
  - Weekly patrol schedule optimized by predicted load
"""

from dataclasses import dataclass
from typing import List, Dict
import datetime
import math


# ── Dataset-derived hourly violation distribution ─────────────────────────────
# Normalised from 298,450 records (hour → fraction of daily total)
HOURLY_DISTRIBUTION = {
    0: 0.064, 1: 0.050, 2: 0.073, 3: 0.075, 4: 0.085,
    5: 0.100, 6: 0.079, 7: 0.043, 8: 0.025, 9: 0.009,
    10: 0.002, 11: 0.002, 12: 0.001, 13: 0.000, 14: 0.000,
    15: 0.000, 16: 0.001, 17: 0.002, 18: 0.006, 19: 0.031,
    20: 0.035, 21: 0.058, 22: 0.067, 23: 0.067,
}

# Day-of-week multiplier (Mon=0 … Sun=6), derived from dataset
DOW_MULTIPLIER = {
    0: 1.05,   # Monday    — commercial delivery day
    1: 1.02,   # Tuesday
    2: 1.08,   # Wednesday — dataset peak
    3: 1.06,   # Thursday
    4: 1.10,   # Friday    — pre-weekend
    5: 0.95,   # Saturday  — lower commercial
    6: 0.74,   # Sunday    — lowest
}

# Zone daily violation rate (from dataset)
ZONE_DAILY_RATE = {
    "BTP051": 102.3, "BTP082": 76.4, "BTP040": 71.0,
    "BTP044": 69.9,  "BTP211": 35.7, "BTP058": 34.4,
    "BTP027": 30.4,  "BTP020": 27.2,
}


@dataclass
class HourForecast:
    zone_id: str
    hour: int
    day_name: str
    violation_probability: float   # 0–1
    expected_violations: float     # count
    expected_ci: float             # congestion index 0–100
    risk_level: str                # HIGH / MEDIUM / LOW
    deploy_recommended: bool


@dataclass
class PatrolSlot:
    zone_id: str
    zone_name: str
    start_hour: int
    end_hour: int
    day: str
    expected_violations: float
    priority_rank: int


def forecast_zone(
    zone_id: str,
    hours_ahead: int = 6,
    reference_dt: datetime.datetime = None,
) -> List[HourForecast]:
    """
    Forecast violation probability and CI for a zone over next N hours.
    """
    if reference_dt is None:
        reference_dt = datetime.datetime.now()

    daily_rate = ZONE_DAILY_RATE.get(zone_id, 30.0)
    forecasts = []

    for offset in range(hours_ahead):
        dt = reference_dt + datetime.timedelta(hours=offset)
        hour = dt.hour
        dow  = dt.weekday()

        hourly_frac = HOURLY_DISTRIBUTION.get(hour, 0.001)
        dow_mult    = DOW_MULTIPLIER.get(dow, 1.0)

        expected_violations = daily_rate * hourly_frac * dow_mult * 24
        # Clamp to realistic per-hour values
        expected_violations = min(expected_violations, daily_rate * 0.15)

        # Violation probability: sigmoid on expected rate
        viol_prob = 1.0 / (1.0 + math.exp(-0.5 * (expected_violations - 5)))

        # Expected CI: proportional to violation probability + zone weight
        from analytics.hotspot_intelligence import DATASET_HOTSPOT_PROFILES
        zone_chronic = DATASET_HOTSPOT_PROFILES.get(zone_id)
        chronic = zone_chronic.chronic_score if zone_chronic else 0.5
        expected_ci = viol_prob * chronic * 65.0   # max CI ~65 for a single zone

        risk = (
            "HIGH"   if expected_violations >= 8  else
            "MEDIUM" if expected_violations >= 3  else
            "LOW"
        )

        forecasts.append(HourForecast(
            zone_id=zone_id, hour=hour,
            day_name=dt.strftime("%A"),
            violation_probability=round(viol_prob, 3),
            expected_violations=round(expected_violations, 1),
            expected_ci=round(expected_ci, 1),
            risk_level=risk,
            deploy_recommended=(risk in ("HIGH", "MEDIUM")),
        ))

    return forecasts


def generate_weekly_patrol_schedule(
    zone_ids: List[str] = None,
) -> List[PatrolSlot]:
    """
    Generate an optimised weekly patrol schedule across all zones.
    Groups high-risk hours into patrol slots and ranks by expected load.
    """
    if zone_ids is None:
        zone_ids = list(ZONE_DAILY_RATE.keys())

    from analytics.hotspot_intelligence import DATASET_HOTSPOT_PROFILES

    ZONE_NAMES = {
        zid: DATASET_HOTSPOT_PROFILES[zid].zone_name
        for zid in zone_ids if zid in DATASET_HOTSPOT_PROFILES
    }

    slots = []
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    for day_idx, day_name in enumerate(days):
        dow_mult = DOW_MULTIPLIER[day_idx]
        for zone_id in zone_ids:
            daily_rate = ZONE_DAILY_RATE.get(zone_id, 30.0)
            zone_name  = ZONE_NAMES.get(zone_id, zone_id)

            # Find high-risk windows: group consecutive high-probability hours
            high_hours = []
            for h in range(24):
                rate = daily_rate * HOURLY_DISTRIBUTION.get(h, 0.001) * 24 * dow_mult
                if rate >= 3.0:
                    high_hours.append((h, rate))

            if not high_hours:
                continue

            # Merge consecutive hours into slots
            current_start = high_hours[0][0]
            current_end   = high_hours[0][0]
            current_total = high_hours[0][1]

            for h, rate in high_hours[1:]:
                if h == current_end + 1:
                    current_end = h
                    current_total += rate
                else:
                    slots.append(PatrolSlot(
                        zone_id=zone_id, zone_name=zone_name,
                        start_hour=current_start, end_hour=current_end,
                        day=day_name, expected_violations=round(current_total, 1),
                        priority_rank=0,
                    ))
                    current_start = h
                    current_end   = h
                    current_total = rate

            slots.append(PatrolSlot(
                zone_id=zone_id, zone_name=zone_name,
                start_hour=current_start, end_hour=current_end,
                day=day_name, expected_violations=round(current_total, 1),
                priority_rank=0,
            ))

    # Rank by expected violations
    slots.sort(key=lambda s: s.expected_violations, reverse=True)
    for i, s in enumerate(slots):
        s.priority_rank = i + 1

    return slots


def print_forecast(zone_id: str, hours_ahead: int = 8):
    """Print a forecast table for one zone."""
    forecasts = forecast_zone(zone_id, hours_ahead)
    print(f"\n  Forecast — {zone_id} — next {hours_ahead} hours")
    print(f"  {'Hour':<6} {'Day':<10} {'Prob':>6} {'Exp.Viol':>9} {'CI':>6} {'Risk':<8} {'Deploy'}")
    print(f"  {'─'*6} {'─'*10} {'─'*6} {'─'*9} {'─'*6} {'─'*8} {'─'*6}")
    for f in forecasts:
        deploy = "✓ YES" if f.deploy_recommended else "  —"
        print(
            f"  {f.hour:02d}:00  {f.day_name:<10} {f.violation_probability:>5.1%} "
            f"{f.expected_violations:>8.1f} {f.expected_ci:>5.1f}% {f.risk_level:<8} {deploy}"
        )


def print_top_patrol_slots(n: int = 20):
    """Print top N patrol deployment slots for the week."""
    slots = generate_weekly_patrol_schedule()
    print(f"\n  Top {n} Patrol Deployment Slots (by expected violations)")
    print(f"  {'#':<4} {'Zone':<30} {'Day':<10} {'Window':<12} {'Exp.Viol':>9}")
    print(f"  {'─'*4} {'─'*30} {'─'*10} {'─'*12} {'─'*9}")
    for s in slots[:n]:
        window = f"{s.start_hour:02d}:00–{s.end_hour+1:02d}:00"
        print(
            f"  {s.priority_rank:<4} {s.zone_name:<30} {s.day:<10} "
            f"{window:<12} {s.expected_violations:>8.1f}"
        )
