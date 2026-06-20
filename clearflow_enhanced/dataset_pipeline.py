"""
dataset_pipeline.py  —  ClearFlow AI (v3 — Full Enhanced)
----------------------------------------------------------
Runs the complete pipeline on the historical CSV.

New in v3:
  ✦ Module G: Hotspot Intelligence — enforcement mode recommendations
  ✦ Module H: Forecast Engine — next-6h violation probability + patrol schedule
  ✦ Module I: Feedback Engine — post-dispatch learning loop (simulated)
  ✦ Richer priority table (confidence, risk, ₹ loss columns)
  ✦ Economic summary block with annualised extrapolation

Usage:
    python dataset_pipeline.py \
        --dataset "jan to may police violation_anonymized791b166.csv" \
        --config  config/clearflow_config.yaml \
        --top     20 \
        --sample  1000
"""

import argparse
import dataclasses
import datetime
import json
import logging
import random
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from utils.dataset_loader import DatasetLoader
from pipeline.perception_engine import IllegalParkingEvent
from pipeline.flow_analyser import FlowAnalyser, FlowSnapshot
from analytics.congestion_engine import CongestionEngine, CongestionImpactRecord
from analytics.hotspot_intelligence import get_enforcement_plan, print_enforcement_plan
from analytics.forecast_engine import (
    forecast_zone, print_forecast, print_top_patrol_slots
)
from analytics.feedback_engine import FeedbackEngine
from analytics.recovery_ai import evaluate_actions, print_recovery_plan
from analytics.emergency_corridor import check_emergency_corridor, print_corridor_status
from analytics.risk_index import compute_citywide_risk_index, print_risk_table
from analytics.digital_twin import (
    simulate_remove_parking, simulate_road_closure,
    simulate_event_timing_shift, print_simulation,
)
from analytics.rl_allocator import RLAllocator
from analytics.towing_optimizer import compute_tow_priority, print_tow_queue
from analytics.traffic_gpt import TrafficGPT, run_interactive_demo
from analytics.surge_detector import detect_historical_surges, print_surge_report
from analytics.budget_optimizer import optimize_citywide_allocation, print_budget_plan
from analytics.hotspot_intelligence import DATASET_HOTSPOT_PROFILES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dataset_pipeline")

ZONE_THROUGHPUT_PROFILES = {
    "BTP051": {"baseline_q": 14.0, "baseline_v": 30.0},
    "BTP082": {"baseline_q": 12.5, "baseline_v": 27.0},
    "BTP040": {"baseline_q": 13.0, "baseline_v": 29.0},
    "BTP044": {"baseline_q": 12.0, "baseline_v": 28.0},
    "BTP211": {"baseline_q": 11.0, "baseline_v": 26.0},
    "BTP058": {"baseline_q": 10.5, "baseline_v": 25.0},
    "BTP027": {"baseline_q": 10.0, "baseline_v": 24.0},
    "BTP020": {"baseline_q": 9.5,  "baseline_v": 23.0},
}

VEHICLE_BLOCKAGE = {
    "CAR": (0.15, 0.20), "SCOOTER": (0.05, 0.12),
    "MOTOR_CYCLE": (0.05, 0.12), "MOPED": (0.04, 0.10),
    "PASSENGER_AUTO": (0.12, 0.18), "MAXI_CAB": (0.25, 0.40),
    "LGV": (0.30, 0.45), "HGV": (0.35, 0.50),
    "PRIVATE_BUS": (0.30, 0.48), "BUS": (0.30, 0.48),
    "VAN": (0.15, 0.25), "LORRY": (0.32, 0.48),
    "TANKER": (0.35, 0.50), "DEFAULT": (0.10, 0.20),
}


def simulate_flow_snapshot(zone_id, vehicle_type, duration_sec, hour, tod_weight, global_cfg):
    profile = ZONE_THROUGHPUT_PROFILES.get(
        zone_id,
        {"baseline_q": global_cfg["baseline"]["throughput_vehicles_per_min"],
         "baseline_v": global_cfg["baseline"]["speed_kmh"]},
    )
    bq, bv = profile["baseline_q"], profile["baseline_v"]
    lo, hi = VEHICLE_BLOCKAGE.get(vehicle_type.replace(" ","_").replace("-","_"),
                                   VEHICLE_BLOCKAGE["DEFAULT"])
    blockage = random.uniform(lo, hi) * min(1.3, 1.0+(duration_sec/1800)*0.3) * (0.5+tod_weight*0.5)
    return FlowSnapshot(
        timestamp=time.time(), zone_id=zone_id,
        throughput_vpm=round(max(0, bq*(1-blockage)), 2),
        avg_speed_kmh=round(max(0, bv*(1-blockage*0.8)), 1),
        congestion_index=round(max(0, (bv - bv*(1-blockage*0.8)) / bv * 100), 1),
        vehicle_count_in_zone=random.randint(8, 20),
        baseline_throughput=bq, baseline_speed=bv, violation_active=True,
    )


def record_to_event(row, zone_cfg_map):
    jname = row.get("junction_name", "")
    zone_cfg = next(
        (zc for zc in zone_cfg_map.values()
         if zc["name"] in jname or zc["id"] in jname),
        None
    )
    if zone_cfg is None:
        return None
    duration = random.uniform(60, 600)
    return IllegalParkingEvent(
        event_id=str(row.get("id", "UNKNOWN")),
        track_id=hash(str(row.get("vehicle_number", ""))) % 100000,
        vehicle_type=str(row.get("vehicle_type_canonical", "CAR")),
        zone_id=zone_cfg["id"], zone_name=zone_cfg["name"],
        police_station=str(row.get("police_station", zone_cfg["police_station"])),
        lat=float(row.get("latitude", zone_cfg["lat"])),
        lon=float(row.get("longitude", zone_cfg["lon"])),
        centroid_px=(0,0), bbox=(0,0,0,0),
        first_seen_ts=time.time()-duration, flagged_ts=time.time(),
        duration_sec=duration,
        offence_code=int(row.get("offence_code_primary", 113)),
        frame_idx=0,
    )


def run(dataset_path: str, config_path: str, top_n: int = 20, sample: int = 500):
    logger.info("=" * 70)
    logger.info("  ClearFlow AI — Full Pipeline (v3)")
    logger.info("=" * 70)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    zone_cfg_map = {z["id"]: z for z in cfg.get("hotspot_zones", [])}

    dl = DatasetLoader(dataset_path)
    dl.summary()
    tod_weights = dl.get_time_of_day_weights()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 0 — MULTI-VEHICLE SURGE DETECTION (Module Q)
    # Runs on the FULL dataset (not the sample) since surge events are rare
    # and would be missed by random sampling.
    # ════════════════════════════════════════════════════════════════════════
    sep0 = "═" * 70
    surge_events = detect_historical_surges(dl.df)
    print_surge_report(surge_events, top_n=15)

    jdf = dl.df[dl.df["is_junction"]].copy()
    logger.info(f"Junction records: {len(jdf):,}")
    if sample and len(jdf) > sample:
        jdf = jdf.sample(n=sample, random_state=42)
        logger.info(f"Sampled {sample} records.")

    engine   = CongestionEngine(config_path)
    feedback = FeedbackEngine()
    results  = []
    skipped  = 0

    for _, row in jdf.iterrows():
        event = record_to_event(row.to_dict(), zone_cfg_map)
        if event is None:
            skipped += 1
            continue

        hour  = int(row.get("hour", 5))
        tod_w = tod_weights.get(hour, 0.5)
        yolo_conf = random.uniform(0.70, 0.96)

        snap = simulate_flow_snapshot(
            event.zone_id, event.vehicle_type,
            event.duration_sec, hour, tod_w, cfg,
        )

        class _PF:
            latest = snap

        record = engine.process(event, _PF(), yolo_conf=yolo_conf)
        results.append(record)

        # ── Simulate dispatch feedback (Module I) ─────────────────────────
        # In production this comes from the officer app.
        # Here we simulate realistic outcomes based on impact score.
        if random.random() < 0.3:   # 30% of violations get feedback simulated
            confirmed  = random.random() < 0.80
            cleared    = confirmed and (random.random() < 0.72)
            resp_time  = random.uniform(120, 900) if confirmed else None
            ci_after   = snap.congestion_index * random.uniform(0.3, 0.7) if cleared else snap.congestion_index
            feedback.record_feedback(
                event_id=record.event_id, zone_id=event.zone_id,
                zone_name=event.zone_name,
                dispatch_confirmed=confirmed,
                response_time_sec=resp_time,
                violation_cleared=cleared,
                ci_before=snap.congestion_index,
                ci_after=ci_after,
                impact_score_predicted=record.impact_score,
            )

    engine.close()

    results.sort(key=lambda r: r.impact_score, reverse=True)
    econ = engine.economic_summary()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 1 — PRIORITY TABLE
    # ════════════════════════════════════════════════════════════════════════
    sep = "═" * 80
    print(f"\n{sep}")
    print(f"  ClearFlow AI — Full Enforcement Report  (v3)")
    print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(sep)
    print(f"  Records  : {len(results):,}  |  Skipped: {skipped:,}")
    print(f"  CRITICAL : {sum(1 for r in results if r.priority=='CRITICAL'):,}  "
          f"HIGH : {sum(1 for r in results if r.priority=='HIGH'):,}  "
          f"MEDIUM : {sum(1 for r in results if r.priority=='MEDIUM'):,}  "
          f"LOW : {sum(1 for r in results if r.priority=='LOW'):,}")
    print(f"{'─'*80}")

    if econ:
        print(f"  ECONOMIC IMPACT")
        print(f"  Commuter-hours lost  : {econ['total_commuter_hours_lost']:>10,.1f} hrs")
        print(f"  Productivity loss    : ₹{econ['total_productivity_loss_inr']:>12,.0f}")
        print(f"  Fuel wasted          : ₹{econ['total_fuel_waste_inr']:>12,.0f}")
        print(f"  Excess CO₂           : {econ['total_co2_kg']:>10,.1f} kg")
        scale = 298450 / max(len(results), 1)
        annual = econ['total_productivity_loss_inr'] * scale / 100_000
        print(f"  Annual loss (full dataset extrapolation): ₹{annual:,.1f} lakh")
        print(f"{'─'*80}")

    print(f"\n  TOP {top_n} ENFORCEMENT TARGETS\n")
    print(f"  {'#':<3} {'Event ID':<16} {'Junction':<24} {'Vehicle':<14} "
          f"{'Score':>5} {'Conf%':>5} {'Risk%':>5} {'Priority':<10} {'₹Loss':>10}")
    print(f"  {'─'*3} {'─'*16} {'─'*24} {'─'*14} {'─'*5} {'─'*5} {'─'*5} {'─'*10} {'─'*10}")

    for i, r in enumerate(results[:top_n], 1):
        print(
            f"  {i:<3} {r.event_id:<16} {r.junction_name[:23]:<24} "
            f"{r.vehicle_type:<14} {r.impact_score:>5.1f} "
            f"{r.violation_confidence_pct:>4.0f}% "
            f"{r.breakdown_risk_pct:>4.0f}% "
            f"{r.priority:<10} ₹{r.productivity_loss_inr:>8,.0f}"
        )

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 2 — HOTSPOT ENFORCEMENT INTELLIGENCE (Module G)
    # ════════════════════════════════════════════════════════════════════════
    print_enforcement_plan()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 3 — 6-HOUR FORECAST for top zone (Module H)
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  ClearFlow AI — Congestion Forecast (next 6 hours)")
    print(sep)
    print_forecast("BTP051", hours_ahead=6)
    print_forecast("BTP082", hours_ahead=6)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 4 — WEEKLY PATROL SCHEDULE (Module H)
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  ClearFlow AI — Optimal Weekly Patrol Schedule")
    print(sep)
    print_top_patrol_slots(n=15)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 5 — POST-DISPATCH LEARNING (Module I)
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  ClearFlow AI — Post-Dispatch Learning Summary")
    print(sep)
    feedback.print_insights()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 6 — URBAN MOBILITY RISK INDEX (Module L)
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  ClearFlow AI — Urban Mobility Risk Index (UMRI)")
    print(sep)
    zone_list = [
        {"id": z["id"], "name": z["name"], "lat": z["lat"], "lon": z["lon"]}
        for z in cfg.get("hotspot_zones", [])
    ]
    # Use latest CI per zone from results history if available
    live_ci_by_zone = {}
    for r in results:
        for z in zone_list:
            if z["name"] == r.junction_name:
                live_ci_by_zone[z["id"]] = r.congestion_index
    risk_scores = compute_citywide_risk_index(zone_list, live_ci_by_zone)
    print_risk_table(risk_scores)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 7 — CONGESTION RECOVERY AI (Module J)
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  ClearFlow AI — Congestion Recovery AI (decision intelligence)")
    print(sep)
    if results:
        top = results[0]
        print_recovery_plan(
            zone_name=top.junction_name,
            current_ci=top.congestion_index,
            blocked_pct=top.throughput_drop_pct,
            vehicles_involved=3,
            has_signal_control=True,
        )

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 8 — EMERGENCY CORRIDOR PROTECTION (Module K)
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  ClearFlow AI — Emergency Corridor Protection")
    print(sep)
    print_corridor_status()
    if results:
        top = results[0]
        alerts = check_emergency_corridor(
            zone_id="", zone_name=top.junction_name,
            lat=top.latitude, lon=top.longitude,
            congestion_index=top.congestion_index,
            speed_penalty_kmh=top.speed_penalty_kmh,
        )
        if alerts:
            print(f"\n  Active alerts near {top.junction_name}:")
            for a in alerts[:3]:
                print(f"  {a.narrative}")
        else:
            print(f"\n  No emergency corridor exposure detected near {top.junction_name}.")

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 9 — CITY DIGITAL TWIN (Module M)
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  ClearFlow AI — City Digital Twin (What-If Simulator)")
    print(sep)
    if results:
        top = results[0]
        sim1 = simulate_remove_parking(
            zone_name=top.junction_name,
            baseline_throughput=top.throughput_baseline_vpm,
            baseline_speed=top.speed_baseline_kmh,
            current_throughput=top.throughput_current_vpm,
            current_speed=top.speed_current_kmh,
        )
        print_simulation(sim1)
        sim2 = simulate_road_closure(
            zone_name=top.junction_name,
            baseline_throughput=top.throughput_baseline_vpm,
            baseline_speed=top.speed_baseline_kmh,
            current_throughput=top.throughput_current_vpm,
            current_speed=top.speed_current_kmh,
        )
        print_simulation(sim2)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 10 — TOWING OPTIMIZER (Module P)
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  ClearFlow AI — AI-Powered Towing Priority Queue")
    print(sep)
    tow_candidates = []
    for r in results[:8]:
        tow_candidates.append(compute_tow_priority(
            event_id=r.event_id, vehicle_type=r.vehicle_type,
            zone_name=r.junction_name, impact_score=r.impact_score,
            duration_seconds=r.duration_seconds,
            lat=r.latitude, lon=r.longitude,
            congestion_index=r.congestion_index,
            speed_penalty_kmh=r.speed_penalty_kmh,
        ))
    if tow_candidates:
        print_tow_queue(tow_candidates, n_trucks=2)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 11 — RL RESOURCE ALLOCATOR (Module N) + TRAFFICGPT (Module O)
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print("  ClearFlow AI — Self-Improving Resource Allocator (RL)")
    print(sep)
    allocator = RLAllocator()
    allocator.bootstrap_from_feedback_history(feedback._history)
    zone_names = {z["id"]: z["name"] for z in cfg.get("hotspot_zones", [])}
    for zid, zname in list(zone_names.items())[:3]:
        rec = allocator.recommend(zid, zname)
        print(f"\n  {rec.narrative}")
    allocator.print_learning_state(zone_names)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 12 — CITYWIDE BUDGET OPTIMIZER (Module R)
    # Solves the JOINT allocation problem across all zones simultaneously,
    # unlike every other module which ranks zones independently.
    # ════════════════════════════════════════════════════════════════════════
    live_impact_by_zone = {}
    for r in results:
        for zid, zname in zone_names.items():
            if zname == r.junction_name:
                live_impact_by_zone[zid] = max(
                    live_impact_by_zone.get(zid, 0.0), r.impact_score
                )

    budget_plan = optimize_citywide_allocation(
        zone_profiles       = DATASET_HOTSPOT_PROFILES,
        total_officers      = 14,     # example daily shift strength
        total_tow_trucks    = 4,
        live_impact_scores  = live_impact_by_zone,
        rl_allocator        = allocator,
    )
    print_budget_plan(budget_plan)

    run_interactive_demo(results)

    # ── Save JSON report ──────────────────────────────────────────────────
    report_path = Path(cfg["output"]["reports_dir"]) / "priority_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump([dataclasses.asdict(r) for r in results[:top_n]], f, indent=2)
    logger.info(f"Report → {report_path}")
    logger.info(f"Stream → {cfg['output']['json_stream']}")

    surge_path = Path(cfg["output"]["reports_dir"]) / "surge_events.json"
    with open(surge_path, "w") as f:
        json.dump([dataclasses.asdict(e) for e in surge_events[:50]], f, indent=2)
    logger.info(f"Surge report → {surge_path}")

    budget_path = Path(cfg["output"]["reports_dir"]) / "budget_allocation.json"
    with open(budget_path, "w") as f:
        json.dump(dataclasses.asdict(budget_plan), f, indent=2)
    logger.info(f"Budget plan → {budget_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ClearFlow AI v3")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config",  default="config/clearflow_config.yaml")
    parser.add_argument("--top",     type=int, default=20)
    parser.add_argument("--sample",  type=int, default=500)
    args = parser.parse_args()
    run(args.dataset, args.config, args.top, args.sample)
