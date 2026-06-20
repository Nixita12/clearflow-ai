"""
digital_twin.py  —  Module M: City Digital Twin (What-If Simulator)
-----------------------------------------------------------------------
Lets a traffic officer test scenarios BEFORE implementing them:

    "What if I remove parking at this junction?"
    "What if a rally starts 1 hour earlier?"
    "What if this road is closed?"

The simulator perturbs the zone's baseline throughput/speed model
and recomputes the congestion index, so commanders see the predicted
outcome before committing resources.

This is a lightweight, transparent simulation (not a full SUMO/VISSIM
microsimulation) — appropriate for a hackathon demo, but the interface
is designed so a real microsimulator could be swapped in later.
"""

from dataclasses import dataclass
from typing import Optional
import copy


@dataclass
class SimulationResult:
    scenario_name: str
    zone_name: str
    baseline_ci: float
    simulated_ci: float
    ci_delta: float                  # negative = improvement
    baseline_throughput_vpm: float
    simulated_throughput_vpm: float
    baseline_speed_kmh: float
    simulated_speed_kmh: float
    verdict: str                     # IMPROVES / WORSENS / NEUTRAL
    narrative: str


def _ci_from_speed(baseline_speed: float, current_speed: float) -> float:
    if baseline_speed <= 0:
        return 0.0
    return max(0.0, (baseline_speed - current_speed) / baseline_speed * 100.0)


def simulate_remove_parking(
    zone_name: str,
    baseline_throughput: float,
    baseline_speed: float,
    current_throughput: float,
    current_speed: float,
    capacity_recovery_pct: float = 70.0,
) -> SimulationResult:
    """
    What if illegal parking is permanently removed from this zone
    (e.g. via a no-parking enforcement zone + bollards)?
    """
    baseline_ci = _ci_from_speed(baseline_speed, current_speed)

    recovered_frac = capacity_recovery_pct / 100.0
    sim_throughput = current_throughput + (baseline_throughput - current_throughput) * recovered_frac
    sim_speed = current_speed + (baseline_speed - current_speed) * recovered_frac
    sim_ci = _ci_from_speed(baseline_speed, sim_speed)

    delta = sim_ci - baseline_ci
    verdict = "IMPROVES" if delta < -2 else ("NEUTRAL" if abs(delta) <= 2 else "WORSENS")

    narrative = (
        f"Removing illegal parking entirely (e.g. bollards + enforcement zone) at "
        f"{zone_name} recovers ~{capacity_recovery_pct:.0f}% of lost capacity: "
        f"CI improves from {baseline_ci:.1f}% to {sim_ci:.1f}% "
        f"({delta:+.1f} pts), throughput rises from {current_throughput:.1f} "
        f"to {sim_throughput:.1f} veh/min."
    )

    return SimulationResult(
        scenario_name="Remove illegal parking",
        zone_name=zone_name, baseline_ci=round(baseline_ci, 1),
        simulated_ci=round(sim_ci, 1), ci_delta=round(delta, 1),
        baseline_throughput_vpm=baseline_throughput,
        simulated_throughput_vpm=round(sim_throughput, 2),
        baseline_speed_kmh=baseline_speed,
        simulated_speed_kmh=round(sim_speed, 1),
        verdict=verdict, narrative=narrative,
    )


def simulate_road_closure(
    zone_name: str,
    baseline_throughput: float,
    baseline_speed: float,
    current_throughput: float,
    current_speed: float,
    diverted_capacity_pct: float = 40.0,
) -> SimulationResult:
    """
    What if one approach road to this junction is closed
    (construction, event, etc.)? Diverted traffic load is absorbed
    elsewhere but remaining capacity here drops.
    """
    baseline_ci = _ci_from_speed(baseline_speed, current_speed)

    remaining_frac = 1.0 - diverted_capacity_pct / 100.0
    sim_throughput = current_throughput * remaining_frac
    # Speed drops further due to bottleneck at the closure point
    sim_speed = current_speed * remaining_frac * 0.9
    sim_ci = _ci_from_speed(baseline_speed, sim_speed)

    delta = sim_ci - baseline_ci
    verdict = "IMPROVES" if delta < -2 else ("NEUTRAL" if abs(delta) <= 2 else "WORSENS")

    narrative = (
        f"Closing one approach to {zone_name} (diverting {diverted_capacity_pct:.0f}% "
        f"of capacity) raises CI from {baseline_ci:.1f}% to {sim_ci:.1f}% "
        f"({delta:+.1f} pts). Recommend pre-positioning diversion signage "
        f"and an officer at the nearest alternate junction."
    )

    return SimulationResult(
        scenario_name="Road closure (1 approach)",
        zone_name=zone_name, baseline_ci=round(baseline_ci, 1),
        simulated_ci=round(sim_ci, 1), ci_delta=round(delta, 1),
        baseline_throughput_vpm=baseline_throughput,
        simulated_throughput_vpm=round(sim_throughput, 2),
        baseline_speed_kmh=baseline_speed,
        simulated_speed_kmh=round(sim_speed, 1),
        verdict=verdict, narrative=narrative,
    )


def simulate_event_timing_shift(
    zone_name: str,
    baseline_throughput: float,
    baseline_speed: float,
    current_throughput: float,
    current_speed: float,
    shift_hours_earlier: float = 1.0,
    event_intensity_pct: float = 50.0,
) -> SimulationResult:
    """
    What if a planned event (rally, concert, match) starts N hours
    earlier than scheduled? Models the overlap with existing peak-hour
    congestion: starting earlier can either avoid or compound the
    evening peak depending on direction of shift.
    """
    baseline_ci = _ci_from_speed(baseline_speed, current_speed)

    # If event shifts INTO peak commute hours, congestion compounds.
    # Heuristic: shifting earlier by 1-2h from a late slot often overlaps
    # peak more; we model this as additional load proportional to intensity.
    overlap_factor = max(0.0, 1.0 - abs(shift_hours_earlier - 1.5) / 3.0)
    extra_load_frac = (event_intensity_pct / 100.0) * overlap_factor * 0.35

    sim_speed = current_speed * (1.0 - extra_load_frac)
    sim_throughput = max(0.0, current_throughput * (1.0 - extra_load_frac * 0.6))
    sim_ci = _ci_from_speed(baseline_speed, sim_speed)

    delta = sim_ci - baseline_ci
    verdict = "WORSENS" if delta > 2 else ("IMPROVES" if delta < -2 else "NEUTRAL")

    direction = "earlier" if shift_hours_earlier > 0 else "later"
    narrative = (
        f"Shifting the event {abs(shift_hours_earlier):.1f}h {direction} near "
        f"{zone_name} changes overlap with peak commute traffic: "
        f"CI moves from {baseline_ci:.1f}% to {sim_ci:.1f}% ({delta:+.1f} pts). "
        f"{'Recommend additional traffic marshals for the overlap window.' if delta > 2 else 'Timing shift is traffic-neutral or favourable.'}"
    )

    return SimulationResult(
        scenario_name=f"Event shift {shift_hours_earlier:.1f}h {direction}",
        zone_name=zone_name, baseline_ci=round(baseline_ci, 1),
        simulated_ci=round(sim_ci, 1), ci_delta=round(delta, 1),
        baseline_throughput_vpm=baseline_throughput,
        simulated_throughput_vpm=round(sim_throughput, 2),
        baseline_speed_kmh=baseline_speed,
        simulated_speed_kmh=round(sim_speed, 1),
        verdict=verdict, narrative=narrative,
    )


def print_simulation(result: SimulationResult):
    arrow = "↓" if result.ci_delta < 0 else ("↑" if result.ci_delta > 0 else "→")
    print(f"\n  ── DIGITAL TWIN: {result.scenario_name} @ {result.zone_name} ──")
    print(f"  CI: {result.baseline_ci:.1f}%  {arrow}  {result.simulated_ci:.1f}%  "
          f"({result.ci_delta:+.1f} pts)  [{result.verdict}]")
    print(f"  Throughput: {result.baseline_throughput_vpm:.1f} → "
          f"{result.simulated_throughput_vpm:.1f} vpm")
    print(f"  Speed     : {result.baseline_speed_kmh:.1f} → "
          f"{result.simulated_speed_kmh:.1f} km/h")
    print(f"  {result.narrative}")
