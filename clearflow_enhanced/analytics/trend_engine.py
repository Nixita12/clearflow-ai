"""
trend_engine.py  —  Module F: Zone Trend Analysis & Predictive Alerting
------------------------------------------------------------------------
Tracks congestion index history per zone, detects:
  - Rising CI trend (early warning before gridlock)
  - Recurring violation patterns (same zone/hour/day)
  - Predicted time-to-gridlock (TTG) in minutes

Used by the dispatch UI to show "⚠️ CI rising — dispatch preemptively"
"""

from collections import deque, defaultdict
from dataclasses import dataclass
import time
import statistics


@dataclass
class TrendSnapshot:
    zone_id: str
    ci_now: float
    ci_1min_ago: float
    ci_trend_per_min: float          # +ve = worsening, -ve = improving
    time_to_gridlock_min: float      # estimated minutes to CI=100; inf if not rising
    recurrence_score: float          # 0–1: how often this zone/hour repeats
    alert_level: str                 # PREEMPTIVE / WATCH / NORMAL
    trend_narrative: str


class TrendEngine:
    """
    Per-zone CI history tracker with trend and gridlock prediction.
    Instantiate once; call update() every time a new FlowSnapshot arrives.
    """

    GRIDLOCK_CI = 90.0               # we call it "gridlock" at CI=90
    HISTORY_WINDOW = 120             # seconds of CI history to keep

    def __init__(self):
        # zone_id → deque of (timestamp, ci) tuples
        self._ci_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=500)
        )
        # zone_id+hour → violation count (recurrence tracking)
        self._recurrence: dict[str, int] = defaultdict(int)

    def record_violation(self, zone_id: str):
        """Call when a violation fires, to update recurrence counter."""
        import datetime
        hour = datetime.datetime.now().hour
        key = f"{zone_id}:{hour}"
        self._recurrence[key] += 1

    def update(self, zone_id: str, ci: float) -> TrendSnapshot:
        """Update CI history and return the current trend snapshot."""
        now = time.time()
        hist = self._ci_history[zone_id]
        hist.append((now, ci))

        # ── Trend: slope of CI over last 60s ─────────────────────────────────
        cutoff = now - 60.0
        recent = [(t, c) for t, c in hist if t >= cutoff]
        ci_1min_ago = recent[0][1] if recent else ci

        if len(recent) >= 2:
            times = [t - recent[0][0] for t, _ in recent]
            cis   = [c for _, c in recent]
            # Simple linear slope (Δci / Δt in seconds → per minute)
            if times[-1] > 0:
                slope_per_sec = (cis[-1] - cis[0]) / times[-1]
                slope_per_min = slope_per_sec * 60.0
            else:
                slope_per_min = 0.0
        else:
            slope_per_min = 0.0

        # ── Time to gridlock ──────────────────────────────────────────────────
        if slope_per_min > 0.5:   # rising meaningfully
            gap = self.GRIDLOCK_CI - ci
            ttg = gap / slope_per_min if gap > 0 else 0.0
        else:
            ttg = float("inf")

        # ── Recurrence score ──────────────────────────────────────────────────
        import datetime
        hour = datetime.datetime.now().hour
        key = f"{zone_id}:{hour}"
        max_seen = max(self._recurrence.values()) if self._recurrence else 1
        recurrence = self._recurrence[key] / max(max_seen, 1)

        # ── Alert level ───────────────────────────────────────────────────────
        if slope_per_min > 3.0 or (ttg < 10 and ttg > 0):
            alert = "PREEMPTIVE"
        elif slope_per_min > 1.0 or ci > 60:
            alert = "WATCH"
        else:
            alert = "NORMAL"

        # ── Narrative ─────────────────────────────────────────────────────────
        if ttg < float("inf"):
            ttg_str = f"Gridlock predicted in ~{ttg:.0f} min."
        else:
            ttg_str = "No imminent gridlock risk."

        trend_dir = "rising" if slope_per_min > 0.5 else (
            "falling" if slope_per_min < -0.5 else "stable"
        )
        narrative = (
            f"CI is {trend_dir} at {slope_per_min:+.1f}%/min. "
            f"{ttg_str} "
            f"Recurrence at this zone/hour: {round(recurrence*100)}% of peak."
        )

        return TrendSnapshot(
            zone_id             = zone_id,
            ci_now              = round(ci, 1),
            ci_1min_ago         = round(ci_1min_ago, 1),
            ci_trend_per_min    = round(slope_per_min, 2),
            time_to_gridlock_min= round(ttg, 1) if ttg < float("inf") else -1,
            recurrence_score    = round(recurrence, 3),
            alert_level         = alert,
            trend_narrative     = narrative,
        )
