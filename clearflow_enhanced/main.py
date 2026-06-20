"""
main.py  —  ClearFlow AI Live Pipeline  (Enhanced v2)
------------------------------------------------------
Ties Module A (PerceptionEngine) + Module B (FlowAnalyser)
+ Module C (CongestionEngine) + Economic/Confidence/Trend into one loop.

New in v2:
  • Confidence scores shown in HUD and console alerts
  • Economic impact (₹ productivity, commuter-hours) in console
  • Trend-based preemptive alerts (CI rising → early dispatch)
  • End-of-session economic summary table

Run modes:
  --source 0          : webcam
  --source video.mp4  : local video file
  --source rtsp://... : IP camera stream
  --demo              : synthetic demo mode (no camera needed)

Usage:
    python main.py --config config/clearflow_config.yaml --source 0 --zone BTP051
    python main.py --config config/clearflow_config.yaml --demo
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.perception_engine import PerceptionEngine
from pipeline.flow_analyser import FlowAnalyser
from analytics.congestion_engine import CongestionEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ── ANSI colours for richer console output ────────────────────────────────────
R  = "\033[91m"   # red
Y  = "\033[93m"   # yellow
G  = "\033[92m"   # green
C  = "\033[96m"   # cyan
B  = "\033[94m"   # blue
M  = "\033[95m"   # magenta
W  = "\033[97m"   # white
DIM= "\033[2m"
RST= "\033[0m"    # reset


def _conf_color(pct: float) -> str:
    if pct >= 85: return G
    if pct >= 60: return Y
    return R


def _priority_color(p: str) -> str:
    return {
        "CRITICAL": R, "HIGH": Y, "MEDIUM": C, "LOW": G
    }.get(p, W)


def get_zone_cfg(cfg: dict, zone_id: str) -> dict:
    for z in cfg.get("hotspot_zones", []):
        if z["id"] == zone_id:
            return z
    raise ValueError(f"Zone {zone_id} not found in config.")


def _print_violation_alert(record):
    """Rich console alert with confidence + economic + trend info."""
    pc  = _priority_color(record.priority)
    cc  = _conf_color(record.violation_confidence_pct)
    rc  = _conf_color(100 - record.breakdown_risk_pct)  # invert for risk
    trend_c = R if record.trend_alert_level == "PREEMPTIVE" else (
              Y if record.trend_alert_level == "WATCH" else G)

    sep = f"{W}{'═'*65}{RST}"
    print(f"\n{sep}")
    print(f"  {pc}▶  VIOLATION DETECTED  [{record.priority}]{RST}")
    print(f"{sep}")
    print(f"  {W}Event  :{RST} {record.event_id}")
    print(f"  {W}Vehicle:{RST} {record.vehicle_type}  at  {record.junction_name}")
    print(f"  {W}Parked :{RST} {record.duration_seconds:.0f}s")
    print()

    # ── Impact ────────────────────────────────────────────────────────────
    print(f"  {C}── TRAFFIC IMPACT ──────────────────────────{RST}")
    print(f"  Throughput ↓ {record.throughput_drop_pct:.1f}%  │  "
          f"Speed ↓ {record.speed_penalty_kmh:.1f} km/h  │  "
          f"~{record.vehicles_affected_per_hour} veh/hr delayed")
    print(f"  Impact score: {pc}{record.impact_score}/100{RST}  │  "
          f"CI = {record.congestion_index:.1f}%")

    # ── Confidence ────────────────────────────────────────────────────────
    print()
    print(f"  {C}── ML CONFIDENCE ───────────────────────────{RST}")
    print(f"  Detection  : {_conf_color(record.detection_confidence_pct)}"
          f"{record.detection_confidence_pct:.0f}%{RST}  "
          f"[{record.confidence_band}]")
    print(f"  Violation  : {cc}{record.violation_confidence_pct:.0f}%{RST}")
    print(f"  Breakdown risk: {rc}{record.breakdown_risk_pct:.0f}% confidence{RST}")

    # ── Trend ─────────────────────────────────────────────────────────────
    print()
    print(f"  {C}── TREND ANALYSIS ──────────────────────────{RST}")
    trend_dir = "↑" if record.ci_trend_per_min > 0.5 else (
                "↓" if record.ci_trend_per_min < -0.5 else "→")
    print(f"  CI trend: {trend_c}{trend_dir} {record.ci_trend_per_min:+.1f}%/min "
          f"[{record.trend_alert_level}]{RST}")
    if record.time_to_gridlock_min > 0:
        print(f"  {R}⚠  Gridlock predicted in ~{record.time_to_gridlock_min:.0f} min{RST}")

    # ── Economic ──────────────────────────────────────────────────────────
    print()
    print(f"  {C}── ECONOMIC IMPACT ─────────────────────────{RST}")
    print(f"  This event  : {M}₹{record.productivity_loss_inr:,.0f}{RST} "
          f"lost  │  {record.commuter_hours_lost:.1f} commuter-hours")
    print(f"  Zone/week   : {M}₹{record.productivity_loss_lakh_weekly:.2f} lakh{RST} "
          f"→ ~{record.commuter_hours_per_week_zone:.0f} hrs/week wasted")
    print(f"  Fuel wasted : ₹{record.fuel_waste_inr:,.0f}  │  "
          f"CO₂ excess: {record.co2_excess_kg:.1f} kg")

    print()
    print(f"  {W}Dispatch:{RST} {record.police_station} police station")
    print(sep)


def run_live(config_path: str, source, zone_id: str, headless: bool = False):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    zone_cfg = get_zone_cfg(cfg, zone_id)
    logger.info(f"Zone: {zone_cfg['name']} ({zone_id})")

    perception = PerceptionEngine(config_path)
    perception.load_model()

    flow = FlowAnalyser(zone_cfg, cfg)
    congestion = CongestionEngine(config_path)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open source: {source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    flow.set_fps(fps)
    logger.info(f"Source FPS: {fps:.1f}")

    frame_idx = 0
    active_events = {}
    t_start = time.time()

    logger.info("Starting frame loop. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("End of stream.")
            break

        detections = perception.infer(frame, frame_idx)
        new_parking_events = perception.check_illegal_parking(detections, frame_idx)

        violation_active = bool(new_parking_events) or bool(active_events)
        snap = flow.update(detections, frame_idx, violation_active)

        for event in new_parking_events:
            # Pass YOLO confidence from the matching detection
            yolo_conf = next(
                (d.confidence for d in detections if d.track_id == event.track_id),
                0.88
            )
            record = congestion.process(event, flow, yolo_conf=yolo_conf)
            active_events[event.event_id] = record
            _print_violation_alert(record)

        if not headless:
            annotated = perception.draw_annotations(frame, detections, new_parking_events)
            _draw_hud(annotated, snap, active_events, flow, congestion)
            cv2.imshow("ClearFlow AI", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("User quit.")
                break

        frame_idx += 1

        if frame_idx % int(fps * 5) == 0:
            logger.info(flow.summary_line())

    cap.release()
    if not headless:
        cv2.destroyAllWindows()
    congestion.close()

    elapsed = time.time() - t_start
    logger.info(
        f"Processed {frame_idx} frames in {elapsed:.1f}s "
        f"({frame_idx/elapsed:.1f} FPS avg)"
    )

    _print_session_summary(congestion)


def _draw_hud(frame, snap, active_events, flow, congestion):
    """Enhanced HUD with confidence + economic mini-stats."""
    h, w = frame.shape[:2]

    def put(text, y, color=(255, 255, 255)):
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,0,0), 3)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1)

    put(f"Zone: {snap.zone_id}", 25)
    put(f"q = {snap.throughput_vpm:.1f} vpm  (base {snap.baseline_throughput:.1f})", 48)
    put(f"v = {snap.avg_speed_kmh:.1f} km/h  (base {snap.baseline_speed:.1f})", 70)
    ci_color = (0,200,80) if snap.congestion_index < 30 else \
               (0,165,255) if snap.congestion_index < 60 else (0,0,220)
    put(f"CI = {snap.congestion_index:.1f}%  dq={flow.delta_throughput_pct():.1f}%  dv={flow.delta_speed_kmh():.1f}km/h", 92, ci_color)
    put(f"Active violations: {len(active_events)}", 114,
        (0,0,220) if active_events else (80,200,80))

    # Economic running total
    econ = congestion.economic_summary()
    if econ:
        put(f"Session loss: Rs {econ['total_productivity_loss_inr']:,.0f}  |  "
            f"{econ['total_commuter_hours_lost']:.1f} hrs", 136, (200, 180, 255))
        put(f"Fuel waste: Rs {econ['total_fuel_waste_inr']:,.0f}  |  "
            f"CO2: {econ['total_co2_kg']:.1f}kg", 158, (200, 180, 255))


def _print_session_summary(congestion):
    """End-of-session economic summary table."""
    econ = congestion.economic_summary()
    if not econ:
        return

    sep = f"{'─'*55}"
    print(f"\n{W}{'═'*55}{RST}")
    print(f"  {C}ClearFlow AI — Session Economic Summary{RST}")
    print(f"{W}{'═'*55}{RST}")
    print(f"  Violations processed    : {econ['violations_processed']}")
    print(sep)
    print(f"  {M}Commuter-hours lost     : {econ['total_commuter_hours_lost']:.1f} hrs{RST}")
    print(f"  {M}Productivity loss (₹)   : ₹{econ['total_productivity_loss_inr']:,.0f}{RST}")
    print(f"  Fuel wasted (₹)         : ₹{econ['total_fuel_waste_inr']:,.0f}")
    print(f"  Excess CO₂              : {econ['total_co2_kg']:.1f} kg")
    print(f"{W}{'═'*55}{RST}")

    if congestion.history:
        print(f"\n  {C}Top 5 by Impact Score:{RST}")
        for i, r in enumerate(congestion.top_priorities(5), 1):
            pc = _priority_color(r.priority)
            print(
                f"  {i}. {pc}[{r.priority}]{RST} {r.event_id:20s} | "
                f"Score={r.impact_score:5.1f} | "
                f"Conf={r.violation_confidence_pct:4.0f}% | "
                f"Risk={r.breakdown_risk_pct:4.0f}% | "
                f"₹{r.productivity_loss_inr:,.0f}"
            )
    print()


def run_demo(config_path: str, fast: bool = True):
    """
    Demo mode: synthetic frames, no camera required.

    BUG FIX (2026-06-20): the original implementation ran 150 frames at
    ~0.04s/frame (~6 real seconds total), but the production config's
    `stationary_threshold_seconds: 60` could never be reached in that
    window — so no violation ever fired and the demo looked broken.

    Fix: when fast=True (default), the demo overrides ONLY this
    PerceptionEngine instance's in-memory threshold to 3 seconds for
    demo purposes. The YAML config on disk is never modified, so the
    real dataset_pipeline.py run (which needs the real 60s threshold
    for accurate scoring) is completely unaffected.
    """
    import numpy as np
    logger.info("Running in DEMO mode (synthetic frames).")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    zone_cfg = cfg["hotspot_zones"][0]
    perception = PerceptionEngine(config_path)

    if fast:
        original_threshold = perception.stationary_thresh_s
        perception.stationary_thresh_s = 3.0   # demo-only override, in-memory
        perception.min_frames_stat = 2          # also lower frame-count gate
        logger.info(
            f"[DEMO] Stationary threshold overridden {original_threshold:.0f}s → "
            f"3s for demo purposes only (config file on disk is untouched)."
        )

    flow = FlowAnalyser(zone_cfg, cfg)
    flow.set_fps(10.0)
    # DEMO-ONLY: finer pixel-to-metre calibration so slow background traffic
    # (down to ~18 km/h post-blockage) still clears the 8px/frame movement
    # threshold and isn't misclassified as parked. Production config's
    # default (8.0 px/metre, tuned for a real camera mount) is untouched —
    # this override only exists on this FlowAnalyser instance, in memory.
    flow.set_pixel_to_metre(20.0)
    congestion = CongestionEngine(config_path)

    n_frames = 80   # ~8 real seconds at 0.1s/frame, enough to clear 3s threshold
    violation_fired = False

    # ── DEMO-ONLY FIX: synthetic moving background traffic ──────────────────
    # The stub detections in perception_engine.py are 3 STATIONARY vehicles
    # (by design, to trigger the parking-violation timer). With no moving
    # traffic in the frame, FlowAnalyser has nothing to measure a speed/
    # throughput drop against, so every economic figure showed ₹0 — not
    # wrong, just an uninformative demo. We add 2 background vehicles here
    # that drift across the configured counting line, slowing down once the
    # violation is active, so the demo actually shows the Δq/Δv/₹ effect the
    # whole point of the system is to surface. This only affects --demo;
    # dataset_pipeline.py and live camera mode are untouched.
    from pipeline.perception_engine import Detection as _Det
    cl = zone_cfg["counting_line"]
    bg_y = cl[1]
    bg_x_start = [cl[0] - 20, cl[0] - 80]
    # Calibrated against the demo's 20px/metre override (see above) so the
    # resulting *_kmh readings shown to the user are realistic, not just
    # "above threshold": 15.6px/frame ≈ 28km/h (matches config baseline),
    # 10.0px/frame ≈ 18km/h (35% slowdown once the violation is active).
    bg_speed_px_per_frame_clear = 15.6
    bg_speed_px_per_frame_blocked = 10.0

    for frame_idx in range(n_frames):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (25, 35, 55)

        detections = perception.infer(frame, frame_idx)

        # Add moving background vehicles, speed depends on whether a
        # violation is currently active in this zone (mirrors real-world
        # effect: blocked lane slows surrounding traffic)
        speed = bg_speed_px_per_frame_blocked if violation_fired else bg_speed_px_per_frame_clear
        for i, x0 in enumerate(bg_x_start):
            bx = x0 + speed * frame_idx
            if bx > cl[2] + 50:
                bx = cl[0] - 20 + (bx - (cl[2] + 50))  # wrap around
            detections.append(_Det(
                track_id=100 + i, class_id=2, class_name="CAR",
                confidence=0.85, bbox=(int(bx), bg_y - 15, int(bx) + 70, bg_y + 15),
                centroid=(int(bx) + 35, bg_y), frame_idx=frame_idx,
                timestamp=time.time(),
            ))

        new_events = perception.check_illegal_parking(detections, frame_idx)
        snap = flow.update(detections, frame_idx, bool(new_events) or violation_fired)

        for evt in new_events:
            violation_fired = True
            yolo_conf = next(
                (d.confidence for d in detections if d.track_id == evt.track_id), 0.88
            )
            record = congestion.process(evt, flow, yolo_conf=yolo_conf)
            _print_violation_alert(record)

        if frame_idx % 20 == 0:
            logger.info(f"Frame {frame_idx:03d} | {flow.summary_line()}")

        time.sleep(0.1)

    congestion.close()
    print(f"\n{C}[DEMO COMPLETE]{RST}")
    if not violation_fired:
        print(
            f"{C}[WARNING] No violation fired in this demo run — this would "
            f"indicate a real regression. Check stationary_threshold_seconds "
            f"and the stub detection positions in perception_engine.py.{RST}"
        )
    _print_session_summary(congestion)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ClearFlow AI v2")
    parser.add_argument("--config",   default="config/clearflow_config.yaml")
    parser.add_argument("--source",   default="0")
    parser.add_argument("--zone",     default="BTP051")
    parser.add_argument("--demo",     action="store_true")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    if args.demo:
        run_demo(args.config)
    else:
        source = int(args.source) if args.source.isdigit() else args.source
        run_live(args.config, source, args.zone, args.headless)
