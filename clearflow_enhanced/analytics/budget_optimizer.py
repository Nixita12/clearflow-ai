"""
budget_optimizer.py  —  Module R: Citywide Enforcement Budget Optimizer
----------------------------------------------------------------------------
Every existing module ranks zones independently (UMRI score, enforcement
ROI, forecast risk). None of them solve the actual constrained decision a
commander faces every morning:

    "I have exactly 14 officers and 4 tow trucks today. Which SPECIFIC
     combination of zone assignments maximizes total citywide congestion
     relief — not which zone looks worst in isolation?"

This is a genuine gap. Module L (UMRI) tells you Safina Plaza is the
single highest-risk zone. But if Safina Plaza already has 6 officers
assigned and is past the point of diminishing returns, the BEST use of
officer #7 might be opening a second front at KR Market instead. Ranking
zones independently can't see that; this module can, because it solves
the allocation as a constrained optimization across ALL zones jointly.

METHOD — Multiple-Choice Knapsack:
  - Each zone can receive 0, 1, 2, or 3 officers (diminishing marginal
    returns are modelled explicitly: officer 2 recovers less than
    officer 1, officer 3 less than officer 2 — this is what makes the
    joint optimization non-trivial; a naive "rank and fill" approach
    gets this wrong because it ignores diminishing returns).
  - Tow trucks are a separate scarce resource, allocated to whichever
    zones have the highest current impact_score from the live pipeline.
  - Objective: maximise total expected CI-point recovery citywide,
    subject to total officers <= budget and total tow trucks <= budget.
  - Solved by dynamic programming (true knapsack optimum, not greedy
    approximation) since the officer budget is small enough (<100) for
    exact DP to run in milliseconds.

This composes cleanly with existing modules: marginal officer recovery
curves are seeded from hotspot_intelligence.py's enforcement_roi and
(when available) rl_allocator.py's learned avg_reward — so the optimizer
gets smarter as the RL allocator accumulates more dispatch outcomes.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_OFFICERS_PER_ZONE = 3

# Diminishing-returns curve: officer N recovers this fraction of officer 1's
# marginal CI-point contribution. Grounded in basic queueing intuition —
# the first officer clears the worst blockage, the second handles overflow,
# the third has rapidly shrinking marginal work to do.
_MARGINAL_RETURN_MULTIPLIER = {1: 1.00, 2: 0.55, 3: 0.25}


@dataclass
class ZoneAllocationPlan:
    zone_id: str
    zone_name: str
    officers_assigned: int
    tow_trucks_assigned: int
    expected_ci_recovery_pts: float    # total, across all assigned officers
    marginal_value_per_officer: List[float]   # value of officer 1, 2, 3 if added
    rationale: str


@dataclass
class CitywideBudgetPlan:
    total_officers_available: int
    total_tow_trucks_available: int
    officers_allocated: int
    tow_trucks_allocated: int
    total_expected_ci_recovery_pts: float
    zone_plans: List[ZoneAllocationPlan]
    unallocated_officers: int
    optimality_note: str
    plan_narrative: str


def _base_officer_value(enforcement_roi: float) -> float:
    """
    Converts a zone's enforcement_roi (already 0-100 scale from
    hotspot_intelligence.py) into an expected CI-point recovery for
    ONE officer at that zone. This is the value used for officer #1;
    officers #2 and #3 are scaled down by _MARGINAL_RETURN_MULTIPLIER.
    """
    return round(enforcement_roi * 0.18, 2)


def _rl_adjusted_value(base_value: float, rl_allocator, zone_id: str, zone_name: str) -> float:
    """
    If the RL allocator has learned outcomes for this zone, blend its
    empirical avg_reward in — so zones with a proven track record of
    high CI recovery get boosted, and zones with poor historical
    clearance get discounted, beyond what the static ROI table alone says.
    """
    if rl_allocator is None:
        return base_value
    try:
        rec = rl_allocator.recommend(zone_id, zone_name)
        if rec.n_historical_trials >= 3:
            # Blend: 60% static ROI-derived value, 40% learned empirical value
            learned_value = rec.expected_ci_recovery_pct * 0.18
            return round(base_value * 0.6 + learned_value * 0.4, 2)
    except Exception:
        pass
    return base_value


def _build_marginal_curve(base_value: float) -> List[float]:
    return [
        round(base_value * _MARGINAL_RETURN_MULTIPLIER[n], 2)
        for n in (1, 2, 3)
    ]


def optimize_citywide_allocation(
    zone_profiles: Dict[str, object],     # DATASET_HOTSPOT_PROFILES
    total_officers: int,
    total_tow_trucks: int = 0,
    live_impact_scores: Optional[Dict[str, float]] = None,  # zone_id -> impact_score, for tow priority
    rl_allocator=None,                     # optional RLAllocator instance
) -> CitywideBudgetPlan:
    """
    Solves the joint officer-allocation problem via exact 0/1 knapsack DP
    over (zone, officer_count) "items", then allocates tow trucks greedily
    by live impact_score (a separate, simpler resource since tow trucks
    are typically assigned to the single worst current violation, not
    spread across a zone the way officers are).
    """
    zone_ids = list(zone_profiles.keys())

    # ── Build all (zone, n_officers) candidate items with marginal values ──
    # item = (zone_id, n_officers, marginal_value_of_THIS_increment, cost=1)
    items = []
    marginal_curves: Dict[str, List[float]] = {}

    for zid in zone_ids:
        profile = zone_profiles[zid]
        base_val = _base_officer_value(profile.enforcement_roi)
        base_val = _rl_adjusted_value(base_val, rl_allocator, zid, profile.zone_name)
        curve = _build_marginal_curve(base_val)
        marginal_curves[zid] = curve
        for n in (1, 2, 3):
            items.append((zid, n, curve[n - 1]))   # each item costs exactly 1 officer

    # ── Knapsack DP over total_officers budget ──────────────────────────────
    # Because each zone has at most 3 increments and they must be taken in
    # order (you can't assign officer #2 without #1), we DP over
    # (zone_index, officers_used) and pick how many officers (0-3) to give
    # each zone, in sequence — this is the "multiple-choice knapsack" variant.
    budget = total_officers
    n_zones = len(zone_ids)

    # dp[i][b] = best total value using first i zones with budget b
    dp = [[0.0] * (budget + 1) for _ in range(n_zones + 1)]
    choice = [[0] * (budget + 1) for _ in range(n_zones + 1)]   # officers given to zone i-1

    for i in range(1, n_zones + 1):
        zid = zone_ids[i - 1]
        curve = marginal_curves[zid]
        cum_value = [0.0, curve[0], curve[0] + curve[1], curve[0] + curve[1] + curve[2]]

        for b in range(budget + 1):
            best_val = dp[i - 1][b]
            best_n = 0
            for n in (1, 2, 3):
                if n <= b:
                    val = dp[i - 1][b - n] + cum_value[n]
                    if val > best_val:
                        best_val = val
                        best_n = n
            dp[i][b] = best_val
            choice[i][b] = best_n

    # ── Backtrack to recover the allocation ─────────────────────────────────
    allocation: Dict[str, int] = {zid: 0 for zid in zone_ids}
    b = budget
    for i in range(n_zones, 0, -1):
        n = choice[i][b]
        zid = zone_ids[i - 1]
        allocation[zid] = n
        b -= n

    total_value = dp[n_zones][budget]
    officers_used = sum(allocation.values())

    # ── Tow truck allocation — greedy by live impact_score (separate resource) ──
    tow_allocation: Dict[str, int] = {zid: 0 for zid in zone_ids}
    if total_tow_trucks > 0 and live_impact_scores:
        ranked = sorted(
            live_impact_scores.items(), key=lambda kv: kv[1], reverse=True
        )
        remaining_tows = total_tow_trucks
        for zid, score in ranked:
            if remaining_tows <= 0:
                break
            if zid in tow_allocation:
                tow_allocation[zid] += 1
                remaining_tows -= 1

    # ── Build per-zone plan objects ──────────────────────────────────────────
    zone_plans = []
    for zid in zone_ids:
        profile = zone_profiles[zid]
        n_off = allocation[zid]
        curve = marginal_curves[zid]
        recovered = sum(curve[:n_off])

        if n_off == 0:
            rationale = (
                f"No officers assigned — marginal value ({curve[0]:.1f} pts for "
                f"officer #1) was lower than the opportunity cost of officers "
                f"assigned elsewhere this round."
            )
        else:
            rationale = (
                f"{n_off} officer(s) assigned, recovering {recovered:.1f} CI-points "
                f"total. Marginal curve: "
                + ", ".join(f"#{i+1}={v:.1f}pt" for i, v in enumerate(curve[:n_off]))
                + ". Diminishing returns modelled explicitly."
            )

        zone_plans.append(ZoneAllocationPlan(
            zone_id=zid, zone_name=profile.zone_name,
            officers_assigned=n_off,
            tow_trucks_assigned=tow_allocation.get(zid, 0),
            expected_ci_recovery_pts=round(recovered, 1),
            marginal_value_per_officer=curve,
            rationale=rationale,
        ))

    zone_plans.sort(key=lambda p: p.expected_ci_recovery_pts, reverse=True)

    optimality_note = (
        "Solved via exact dynamic-programming knapsack (not greedy ranking) — "
        "guaranteed optimal allocation given the modelled diminishing-returns "
        "curve. A naive 'fill highest-ROI zone first' approach would have "
        "under-allocated to secondary hotspots once the top zone hit "
        "diminishing returns."
    )

    plan_narrative = (
        f"With {total_officers} officers and {total_tow_trucks} tow trucks today: "
        f"optimal allocation recovers {total_value:.1f} CI-points citywide "
        f"(vs. naive single-zone stacking). "
        f"Top allocation: {zone_plans[0].zone_name} gets "
        f"{zone_plans[0].officers_assigned} officer(s); "
        f"{sum(1 for p in zone_plans if p.officers_assigned > 0)} zones receive coverage "
        f"out of {len(zone_plans)} total."
    )

    return CitywideBudgetPlan(
        total_officers_available=total_officers,
        total_tow_trucks_available=total_tow_trucks,
        officers_allocated=officers_used,
        tow_trucks_allocated=sum(tow_allocation.values()),
        total_expected_ci_recovery_pts=round(total_value, 1),
        zone_plans=zone_plans,
        unallocated_officers=total_officers - officers_used,
        optimality_note=optimality_note,
        plan_narrative=plan_narrative,
    )


def print_budget_plan(plan: CitywideBudgetPlan):
    sep = "═" * 82
    print(f"\n{sep}")
    print(f"  ClearFlow AI — Module R: Citywide Enforcement Budget Optimizer")
    print(f"  (Exact knapsack DP — jointly optimal, not per-zone ranking)")
    print(sep)
    print(f"  Officers available : {plan.total_officers_available}  "
          f"(allocated: {plan.officers_allocated}, idle: {plan.unallocated_officers})")
    print(f"  Tow trucks available: {plan.total_tow_trucks_available}  "
          f"(allocated: {plan.tow_trucks_allocated})")
    print(f"  Total expected CI recovery: {plan.total_expected_ci_recovery_pts:.1f} points citywide")
    print(f"{'─'*82}")
    print(f"  {'Zone':<28} {'Officers':>8} {'Tows':>5} {'CI Recovery':>12} {'Marginal Curve':<28}")
    print(f"  {'─'*28} {'─'*8} {'─'*5} {'─'*12} {'─'*28}")
    for p in plan.zone_plans:
        curve_str = "/".join(f"{v:.1f}" for v in p.marginal_value_per_officer)
        print(
            f"  {p.zone_name[:27]:<28} {p.officers_assigned:>8} {p.tow_trucks_assigned:>5} "
            f"{p.expected_ci_recovery_pts:>11.1f}p   [{curve_str}]"
        )
    print(f"\n  {plan.plan_narrative}")
    print(f"  {plan.optimality_note}")
    print(sep)
