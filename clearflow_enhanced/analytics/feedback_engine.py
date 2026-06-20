"""
feedback_engine.py  —  Module I: Post-Dispatch Learning Loop
-------------------------------------------------------------
Addresses the problem statement gap:
  "No post-event learning system."

Tracks:
  - Was dispatch acted on? (dispatch_confirmed)
  - How long did it take officer to arrive? (response_time_sec)
  - Was violation cleared? (cleared)
  - Actual CI recovery after clearance (ci_after)

Uses this feedback to:
  1. Recalibrate zone weights (high-response-time zones get boosted priority)
  2. Update enforcement ROI scores
  3. Detect zones where dispatches are consistently ignored (flag for escalation)
  4. Generate a post-event learning summary for commanders

All feedback stored in a local JSONL log so learning persists across sessions.
"""

import json
import logging
import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class DispatchFeedback:
    event_id: str
    zone_id: str
    zone_name: str
    dispatched_at: str              # ISO8601
    dispatch_confirmed: bool        # officer acknowledged
    response_time_sec: Optional[float]   # None if not responded
    violation_cleared: bool
    ci_before: float
    ci_after: Optional[float]       # CI measured 10min after clearance
    ci_recovery_pct: float          # how much CI recovered (0–100)
    impact_score_predicted: float
    actual_severity: str            # CONFIRMED / FALSE_POSITIVE / EXAGGERATED


@dataclass
class LearningInsight:
    zone_id: str
    zone_name: str
    total_dispatches: int
    confirmation_rate_pct: float    # % dispatches acknowledged
    avg_response_time_sec: float
    clearance_rate_pct: float       # % violations actually cleared
    avg_ci_recovery_pct: float
    false_positive_rate_pct: float
    recommendation: str             # action for commander


class FeedbackEngine:
    """
    Records dispatch outcomes and derives learning insights.
    Call record_feedback() after each dispatch outcome is known.
    Call get_insights() to get zone-level learning summary.
    """

    def __init__(self, feedback_path: str = "output/feedback/dispatch_feedback.jsonl"):
        self.path = Path(feedback_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._history: list[DispatchFeedback] = []
        self._load_history()

    def _load_history(self):
        """Load existing feedback from disk on startup."""
        if not self.path.exists():
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        d = json.loads(line)
                        self._history.append(DispatchFeedback(**d))
                    except Exception:
                        pass
        logger.info(f"Loaded {len(self._history)} feedback records.")

    def record_feedback(
        self,
        event_id: str,
        zone_id: str,
        zone_name: str,
        dispatch_confirmed: bool,
        response_time_sec: Optional[float],
        violation_cleared: bool,
        ci_before: float,
        ci_after: Optional[float],
        impact_score_predicted: float,
    ) -> DispatchFeedback:
        """Record the outcome of a dispatch for a violation event."""

        ci_recovery = 0.0
        if ci_after is not None and ci_before > 0:
            ci_recovery = max(0.0, (ci_before - ci_after) / ci_before * 100.0)

        # Classify actual severity
        if not dispatch_confirmed:
            severity = "FALSE_POSITIVE" if not violation_cleared else "CONFIRMED"
        elif violation_cleared and ci_recovery > 20:
            severity = "CONFIRMED"
        elif impact_score_predicted > 60 and ci_recovery < 10:
            severity = "EXAGGERATED"
        else:
            severity = "CONFIRMED"

        fb = DispatchFeedback(
            event_id               = event_id,
            zone_id                = zone_id,
            zone_name              = zone_name,
            dispatched_at          = datetime.datetime.utcnow().isoformat() + "Z",
            dispatch_confirmed     = dispatch_confirmed,
            response_time_sec      = response_time_sec,
            violation_cleared      = violation_cleared,
            ci_before              = ci_before,
            ci_after               = ci_after,
            ci_recovery_pct        = round(ci_recovery, 1),
            impact_score_predicted = impact_score_predicted,
            actual_severity        = severity,
        )

        # Persist
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(fb)) + "\n")

        self._history.append(fb)
        logger.info(
            f"[FEEDBACK] {event_id} | cleared={violation_cleared} | "
            f"CI recovery={ci_recovery:.1f}% | severity={severity}"
        )
        return fb

    def get_insights(self) -> list[LearningInsight]:
        """Derive per-zone learning insights from all recorded feedback."""
        if not self._history:
            return []

        by_zone: dict[str, list[DispatchFeedback]] = defaultdict(list)
        for fb in self._history:
            by_zone[fb.zone_id].append(fb)

        insights = []
        for zone_id, records in by_zone.items():
            zone_name = records[0].zone_name
            n = len(records)

            confirmed       = sum(1 for r in records if r.dispatch_confirmed)
            cleared         = sum(1 for r in records if r.violation_cleared)
            false_positives = sum(1 for r in records if r.actual_severity == "FALSE_POSITIVE")
            response_times  = [r.response_time_sec for r in records if r.response_time_sec]
            ci_recoveries   = [r.ci_recovery_pct for r in records if r.ci_after is not None]

            conf_rate  = confirmed / n * 100
            clear_rate = cleared / n * 100
            fp_rate    = false_positives / n * 100
            avg_resp   = sum(response_times) / len(response_times) if response_times else -1
            avg_ci_rec = sum(ci_recoveries) / len(ci_recoveries) if ci_recoveries else 0

            # Generate recommendation
            if fp_rate > 30:
                rec = f"High false-positive rate ({fp_rate:.0f}%) — review zone polygon or threshold."
            elif avg_resp > 600 and avg_resp > 0:
                rec = f"Slow response ({avg_resp/60:.0f} min avg) — consider closer patrol base."
            elif clear_rate < 50:
                rec = f"Low clearance rate ({clear_rate:.0f}%) — escalate to tow authority."
            elif conf_rate > 85 and avg_ci_rec > 30:
                rec = f"High-performing zone — strong ROI, maintain fixed post."
            else:
                rec = "Performance within expected range — continue monitoring."

            insights.append(LearningInsight(
                zone_id               = zone_id,
                zone_name             = zone_name,
                total_dispatches      = n,
                confirmation_rate_pct = round(conf_rate, 1),
                avg_response_time_sec = round(avg_resp, 1),
                clearance_rate_pct    = round(clear_rate, 1),
                avg_ci_recovery_pct   = round(avg_ci_rec, 1),
                false_positive_rate_pct= round(fp_rate, 1),
                recommendation        = rec,
            ))

        return sorted(insights, key=lambda i: i.clearance_rate_pct, reverse=True)

    def print_insights(self):
        insights = self.get_insights()
        if not insights:
            print("  No feedback data yet.")
            return

        sep = "═" * 80
        print(f"\n{sep}")
        print("  ClearFlow AI — Post-Dispatch Learning Report")
        print(sep)
        print(f"  {'Zone':<28} {'Disp':>5} {'Conf%':>6} {'Clear%':>7} "
              f"{'FP%':>5} {'CI Rec%':>8} {'Resp(s)':>8}")
        print(f"  {'─'*28} {'─'*5} {'─'*6} {'─'*7} {'─'*5} {'─'*8} {'─'*8}")
        for ins in insights:
            resp = f"{ins.avg_response_time_sec:.0f}" if ins.avg_response_time_sec > 0 else "N/A"
            print(
                f"  {ins.zone_name:<28} {ins.total_dispatches:>5} "
                f"{ins.confirmation_rate_pct:>5.0f}% {ins.clearance_rate_pct:>6.0f}% "
                f"{ins.false_positive_rate_pct:>4.0f}% {ins.avg_ci_recovery_pct:>7.0f}% "
                f"{resp:>8}"
            )
        print(sep)
        print("\n  Recommendations:")
        for ins in insights:
            print(f"  [{ins.zone_id}] {ins.recommendation}")
        print()
