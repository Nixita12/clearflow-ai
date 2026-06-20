"""
congestion_engine.py  —  Module C  (Enhanced v2)
-------------------------------------------------
Quantifies the traffic cost of each illegal parking event.

ENHANCEMENTS over v1:
  ✦ Confidence scores (detection, violation, breakdown risk)
  ✦ Economic impact (₹ productivity loss, commuter-hours, fuel waste, CO₂)
  ✦ Trend analysis (CI slope, time-to-gridlock, preemptive alerting)
  ✦ Rich government-pitch narrative with economic justification
  ✦ Confidence bands rendered in dispatch UI

Impact Score Formula (unchanged — calibrated from dataset):
  Score = (ΔThroughput_pct × 0.40)
        + (ΔSpeed_norm      × 0.30)
        + (ZoneWeight        × 0.20)
        + (TimeOfDayWeight   × 0.10)
"""

import json
import time
import logging
import datetime
from dataclasses import dataclass, asdict, field
from pathlib import Path

import yaml

from pipeline.perception_engine import IllegalParkingEvent
from pipeline.flow_analyser import FlowAnalyser, FlowSnapshot
from analytics.economic_engine import compute_economic_impact, EconomicImpact
from analytics.confidence_engine import compute_confidence, ConfidenceReport
from analytics.trend_engine import TrendEngine, TrendSnapshot

logger = logging.getLogger(__name__)


@dataclass
class CongestionImpactRecord:
    """
    Full enhanced output record.
    Extends v1 with economic, confidence, and trend fields.
    """
    # ── Dataset-equivalent fields ──────────────────────────────────────────
    event_id: str
    vehicle_type: str
    violation_type: list
    offence_code: list
    police_station: str
    junction_name: str
    latitude: float
    longitude: float
    created_datetime: str
    data_sent_to_scita: bool

    # ── Live impact fields ─────────────────────────────────────────────────
    duration_seconds: float
    throughput_baseline_vpm: float
    throughput_current_vpm: float
    throughput_drop_pct: float
    speed_baseline_kmh: float
    speed_current_kmh: float
    speed_penalty_kmh: float
    congestion_index: float
    vehicles_affected_per_hour: int
    zone_weight: float
    time_of_day_weight: float
    impact_score: float
    priority: str
    narrative: str

    # ── NEW: Confidence fields ─────────────────────────────────────────────
    detection_confidence_pct: float     # YOLO detection certainty
    violation_confidence_pct: float     # P(genuine illegal parking)
    breakdown_risk_pct: float           # P(escalates to gridlock)
    confidence_band: str                # overall band for dispatch UI

    # ── NEW: Economic fields ───────────────────────────────────────────────
    commuter_hours_lost: float          # this event
    commuter_hours_per_week_zone: float # zone-level weekly waste
    productivity_loss_inr: float        # ₹ this event
    productivity_loss_lakh_weekly: float# ₹ lakhs/week zone
    fuel_waste_inr: float
    co2_excess_kg: float
    economic_narrative: str

    # ── NEW: Trend fields ─────────────────────────────────────────────────
    ci_trend_per_min: float             # +ve = worsening
    time_to_gridlock_min: float         # -1 = no imminent risk
    trend_alert_level: str              # PREEMPTIVE / WATCH / NORMAL
    trend_narrative: str


class CongestionEngine:
    """
    Enhanced congestion engine with economic quantification,
    ML confidence scores, and trend-based preemptive alerting.
    """

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        sc = self.cfg["impact_score"]
        self.w_throughput = sc["w_throughput"]
        self.w_speed      = sc["w_speed"]
        self.w_zone       = sc["w_zone"]
        self.w_time       = sc["w_time"]

        self.thresh_critical = sc["critical_threshold"]
        self.thresh_high     = sc["high_threshold"]
        self.thresh_medium   = sc["medium_threshold"]

        self.tod_weights: dict[int, float] = {
            int(k): float(v)
            for k, v in self.cfg["time_of_day_weights"].items()
        }

        self.zone_weights: dict[str, float] = {
            z["id"]: float(z["zone_weight"])
            for z in self.cfg.get("hotspot_zones", [])
        }

        # ── New engines ───────────────────────────────────────────────────
        self._trend_engine = TrendEngine()
        self._prev_ci: dict[str, float] = {}   # for CI trend delta

        # Output stream
        out_path = Path(self.cfg["output"]["json_stream"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = open(out_path, "a", buffering=1)

        self.history: list[CongestionImpactRecord] = []

    # ── Main entry point ──────────────────────────────────────────────────────

    def process(
        self,
        event: IllegalParkingEvent,
        flow: FlowAnalyser,
        yolo_conf: float = 0.88,      # override from Detection.confidence
        frames_stationary: int = 30,
    ) -> CongestionImpactRecord:

        snap: FlowSnapshot = flow.latest
        if snap is None:
            snap = self._empty_snapshot(event.zone_id)

        current_hour = datetime.datetime.now().hour

        # ── Compute base score components ─────────────────────────────────
        if snap.baseline_throughput > 0:
            delta_q_pct = max(
                0.0,
                (snap.baseline_throughput - snap.throughput_vpm)
                / snap.baseline_throughput * 100.0,
            )
        else:
            delta_q_pct = 0.0

        if snap.baseline_speed > 0:
            delta_v_pct = max(
                0.0,
                (snap.baseline_speed - snap.avg_speed_kmh)
                / snap.baseline_speed * 100.0,
            )
        else:
            delta_v_pct = 0.0

        zone_w = self.zone_weights.get(event.zone_id, 0.30) * 100.0
        tod_w  = self.tod_weights.get(current_hour, 0.5) * 100.0

        score = (
            delta_q_pct * self.w_throughput
            + delta_v_pct * self.w_speed
            + zone_w      * self.w_zone
            + tod_w       * self.w_time
        )
        score = round(min(100.0, max(0.0, score)), 1)
        priority = self._priority(score)

        baseline_hourly    = snap.baseline_throughput * 60
        vehicles_affected  = int(baseline_hourly * delta_q_pct / 100.0)
        zone_weight_val    = self.zone_weights.get(event.zone_id, 0.3)
        tod_weight_val     = round(self.tod_weights.get(current_hour, 0.5), 2)

        # ── MODULE D: Economic impact ──────────────────────────────────────
        econ: EconomicImpact = compute_economic_impact(
            vehicles_affected_per_hour = vehicles_affected,
            throughput_drop_pct        = delta_q_pct,
            duration_seconds           = event.duration_sec,
            zone_name                  = event.zone_name,
        )

        # ── MODULE E: Confidence scores ────────────────────────────────────
        ci_prev = self._prev_ci.get(event.zone_id, snap.congestion_index)
        ci_trend = snap.congestion_index - ci_prev    # Δci since last event

        conf: ConfidenceReport = compute_confidence(
            yolo_conf              = yolo_conf,
            frames_stationary      = frames_stationary,
            stationary_threshold_s = self.cfg["parking"]["stationary_threshold_seconds"],
            duration_sec           = event.duration_sec,
            congestion_index       = snap.congestion_index,
            zone_weight            = zone_weight_val,
            time_of_day_weight     = tod_weight_val,
            congestion_trend       = ci_trend,
            vehicle_type           = event.vehicle_type,
        )
        self._prev_ci[event.zone_id] = snap.congestion_index

        # Derive overall confidence band (worst of detection + violation)
        def _band_rank(b: str) -> int:
            return {"HIGH": 2, "MODERATE": 1, "LOW": 0}.get(b, 0)

        overall_band = min(
            [conf.detection_band, conf.violation_band],
            key=_band_rank
        )

        # ── MODULE F: Trend analysis ───────────────────────────────────────
        trend: TrendSnapshot = self._trend_engine.update(
            zone_id = event.zone_id,
            ci      = snap.congestion_index,
        )
        self._trend_engine.record_violation(event.zone_id)

        # ── Build enhanced narrative ───────────────────────────────────────
        narrative = self._build_narrative(
            event, snap, delta_q_pct,
            snap.baseline_speed - snap.avg_speed_kmh,
            vehicles_affected, score, priority,
            conf, econ, trend,
        )

        # ── Build record ──────────────────────────────────────────────────
        record = CongestionImpactRecord(
            event_id                    = event.event_id,
            vehicle_type                = event.vehicle_type,
            violation_type              = [event.offence_name],
            offence_code                = [event.offence_code],
            police_station              = event.police_station,
            junction_name               = event.zone_name,
            latitude                    = event.lat,
            longitude                   = event.lon,
            created_datetime            = datetime.datetime.utcnow().isoformat() + "Z",
            data_sent_to_scita          = True,
            duration_seconds            = round(event.duration_sec, 1),
            throughput_baseline_vpm     = snap.baseline_throughput,
            throughput_current_vpm      = snap.throughput_vpm,
            throughput_drop_pct         = round(delta_q_pct, 1),
            speed_baseline_kmh          = snap.baseline_speed,
            speed_current_kmh           = snap.avg_speed_kmh,
            speed_penalty_kmh           = round(snap.baseline_speed - snap.avg_speed_kmh, 1),
            congestion_index            = snap.congestion_index,
            vehicles_affected_per_hour  = vehicles_affected,
            zone_weight                 = zone_weight_val,
            time_of_day_weight          = tod_weight_val,
            impact_score                = score,
            priority                    = priority,
            narrative                   = narrative,
            # Confidence
            detection_confidence_pct    = conf.detection_confidence_pct,
            violation_confidence_pct    = conf.violation_confidence_pct,
            breakdown_risk_pct          = conf.breakdown_risk_pct,
            confidence_band             = overall_band,
            # Economic
            commuter_hours_lost         = econ.commuter_hours_lost_per_event,
            commuter_hours_per_week_zone= econ.commuter_hours_per_week_zone,
            productivity_loss_inr       = econ.productivity_loss_inr,
            productivity_loss_lakh_weekly= econ.productivity_loss_lakh_weekly,
            fuel_waste_inr              = econ.fuel_waste_inr,
            co2_excess_kg               = econ.co2_excess_kg,
            economic_narrative          = econ.economic_narrative,
            # Trend
            ci_trend_per_min            = trend.ci_trend_per_min,
            time_to_gridlock_min        = trend.time_to_gridlock_min,
            trend_alert_level           = trend.alert_level,
            trend_narrative             = trend.trend_narrative,
        )

        self._stream.write(json.dumps(asdict(record)) + "\n")
        self.history.append(record)

        logger.info(
            f"[IMPACT] {event.event_id} | Score={score} | {priority} | "
            f"Conf={conf.violation_confidence_pct}% | Risk={conf.breakdown_risk_pct}% | "
            f"Δq={delta_q_pct:.1f}% | Δv={snap.baseline_speed - snap.avg_speed_kmh:.1f}km/h | "
            f"₹{round(econ.productivity_loss_inr):,} lost | "
            f"Trend={trend.alert_level}"
        )

        return record

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _priority(self, score: float) -> str:
        if score >= self.thresh_critical: return "CRITICAL"
        if score >= self.thresh_high:     return "HIGH"
        if score >= self.thresh_medium:   return "MEDIUM"
        return "LOW"

    def _build_narrative(
        self,
        event: IllegalParkingEvent,
        snap: FlowSnapshot,
        delta_q_pct: float,
        delta_v: float,
        vehicles_affected: int,
        score: float,
        priority: str,
        conf: ConfidenceReport,
        econ: EconomicImpact,
        trend: TrendSnapshot,
    ) -> str:
        ttg_str = (
            f" Gridlock predicted in ~{trend.time_to_gridlock_min:.0f} min."
            if trend.time_to_gridlock_min > 0
            else ""
        )
        econ_short = (
            f"₹{round(econ.productivity_loss_inr):,} productivity cost this event "
            f"({round(econ.commuter_hours_lost_per_event, 1)} commuter-hours); "
            f"zone-level waste: ₹{round(econ.productivity_loss_lakh_weekly, 2)} lakh/week."
        )
        return (
            f"[{priority}] {event.event_id} — {event.vehicle_type} at {event.zone_name} "
            f"parked {event.duration_sec:.0f}s. "
            f"Throughput ↓{delta_q_pct:.1f}%, speed ↓{delta_v:.1f} km/h, "
            f"~{vehicles_affected} vehicles/hr delayed. "
            f"Impact score: {score}/100. "
            f"Violation confidence: {conf.violation_confidence_pct}% | "
            f"Breakdown risk: {conf.breakdown_risk_pct}%. "
            f"{econ_short}"
            f" CI trend: {trend.ci_trend_per_min:+.1f}%/min [{trend.alert_level}].{ttg_str} "
            f"Dispatch to: {event.police_station} police station."
        )

    @staticmethod
    def _empty_snapshot(zone_id: str) -> FlowSnapshot:
        return FlowSnapshot(
            timestamp=time.time(), zone_id=zone_id,
            throughput_vpm=0.0, avg_speed_kmh=0.0, congestion_index=0.0,
            vehicle_count_in_zone=0, baseline_throughput=12.0,
            baseline_speed=28.0, violation_active=True,
        )

    def top_priorities(self, n: int = 5) -> list[CongestionImpactRecord]:
        return sorted(self.history, key=lambda r: r.impact_score, reverse=True)[:n]

    def economic_summary(self) -> dict:
        """Aggregate economic totals across all violations in history."""
        if not self.history:
            return {}
        total_hours = sum(r.commuter_hours_lost for r in self.history)
        total_inr   = sum(r.productivity_loss_inr for r in self.history)
        total_fuel  = sum(r.fuel_waste_inr for r in self.history)
        total_co2   = sum(r.co2_excess_kg for r in self.history)
        return {
            "total_commuter_hours_lost" : round(total_hours, 1),
            "total_productivity_loss_inr": round(total_inr, 2),
            "total_fuel_waste_inr"       : round(total_fuel, 2),
            "total_co2_kg"               : round(total_co2, 2),
            "violations_processed"       : len(self.history),
        }

    def close(self):
        self._stream.close()
