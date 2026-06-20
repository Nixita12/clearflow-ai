"""
flow_analyser.py  —  Module B
------------------------------
Computes real-time traffic flow metrics from tracked vehicle positions.

Metrics produced per frame window:
  q  — throughput (vehicles crossing counting line per minute)
  v  — average speed of moving vehicles (km/h)
  CI — Congestion Index: (V_baseline - V_current) / V_baseline * 100

These feed directly into Module C (congestion_engine.py) for impact scoring.
"""

import time
import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FlowSnapshot:
    """One window's worth of traffic measurements."""
    timestamp: float
    zone_id: str
    throughput_vpm: float        # vehicles per minute crossing counting line
    avg_speed_kmh: float         # moving vehicles only
    congestion_index: float      # 0–100 (0 = free flow, 100 = gridlock)
    vehicle_count_in_zone: int   # vehicles currently inside ROI
    baseline_throughput: float
    baseline_speed: float
    violation_active: bool       # True if an illegal parking event is live


@dataclass
class _TrackHistory:
    """Internal per-track state for speed estimation."""
    positions: deque = field(default_factory=lambda: deque(maxlen=30))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=30))
    crossed_line: bool = False


class FlowAnalyser:
    """
    Per-zone traffic flow analyser.
    Instantiate one per hotspot zone.
    """

    # Pixel-to-metre conversion: calibrated per camera.
    # Override via set_pixel_to_metre() once camera is calibrated.
    DEFAULT_PX_PER_METRE = 8.0   # rough default for 4-lane arterial at 5m mount height

    def __init__(self, zone_cfg: dict, global_cfg: dict):
        """
        zone_cfg : one entry from hotspot_zones list in YAML
        global_cfg : the full parsed YAML dict
        """
        self.zone_id   = zone_cfg["id"]
        self.zone_name = zone_cfg["name"]

        # Virtual counting line [x1, y1, x2, y2]
        cl = zone_cfg["counting_line"]
        self.line_p1 = (cl[0], cl[1])
        self.line_p2 = (cl[2], cl[3])

        bc = global_cfg["baseline"]
        self.baseline_throughput = bc["throughput_vehicles_per_min"]
        self.baseline_speed      = bc["speed_kmh"]
        self.rolling_window      = bc["rolling_window_frames"]

        self.px_per_metre = self.DEFAULT_PX_PER_METRE
        self.fps = 30.0  # updated by set_fps()

        # Rolling history for baseline recalibration (clean periods only)
        self._clean_throughput_buf: deque = deque(maxlen=self.rolling_window)
        self._clean_speed_buf: deque      = deque(maxlen=self.rolling_window)

        # Per-track state
        self._tracks: dict[int, _TrackHistory] = {}

        # Crossing counter (resets every minute)
        self._crossing_count = 0
        self._window_start_ts = time.time()

        # Latest snapshot
        self.latest: FlowSnapshot | None = None

    # ── Calibration ──────────────────────────────────────────────────────────

    def set_pixel_to_metre(self, px_per_metre: float):
        self.px_per_metre = px_per_metre

    def set_fps(self, fps: float):
        self.fps = fps

    # ── Per-frame update ─────────────────────────────────────────────────────

    def update(
        self,
        detections: list,          # list of Detection from PerceptionEngine
        frame_idx: int,
        violation_active: bool = False,
    ) -> FlowSnapshot:
        """
        Process one frame's worth of detections.
        Returns a FlowSnapshot with current metrics.

        Call this every frame in the main loop.
        """
        now = time.time()
        active_ids = {d.track_id for d in detections}

        # ── Update track histories ─────────────────────────────────────────
        for det in detections:
            tid = det.track_id
            if tid not in self._tracks:
                self._tracks[tid] = _TrackHistory()

            hist = self._tracks[tid]
            cx, cy = det.centroid
            hist.positions.append((cx, cy))
            hist.timestamps.append(now)

            # ── Line crossing detection ────────────────────────────────────
            if len(hist.positions) >= 2:
                prev = hist.positions[-2]
                curr = hist.positions[-1]
                if self._crosses_line(prev, curr) and not hist.crossed_line:
                    hist.crossed_line = True
                    self._crossing_count += 1
                elif not self._crosses_line(prev, curr):
                    hist.crossed_line = False   # allow re-crossing

        # ── Purge stale tracks ────────────────────────────────────────────
        stale = [tid for tid in self._tracks if tid not in active_ids]
        for tid in stale:
            del self._tracks[tid]

        # ── Compute throughput (q) ────────────────────────────────────────
        elapsed = now - self._window_start_ts
        if elapsed >= 60.0:
            throughput_vpm = self._crossing_count / (elapsed / 60.0)
            self._crossing_count = 0
            self._window_start_ts = now
        else:
            # Extrapolate partial window
            throughput_vpm = (
                self._crossing_count / max(elapsed, 1.0)
            ) * 60.0

        # ── Compute average speed (v) of moving vehicles ──────────────────
        speeds = []
        for det in detections:
            speed = self._estimate_speed_kmh(det.track_id)
            if speed is not None and speed > 2.0:   # filter parked (<2 km/h)
                speeds.append(speed)

        avg_speed = float(np.mean(speeds)) if speeds else self.baseline_speed

        # ── Update rolling baseline (only during clean periods) ───────────
        if not violation_active:
            self._clean_throughput_buf.append(throughput_vpm)
            self._clean_speed_buf.append(avg_speed)
            if len(self._clean_throughput_buf) >= 10:
                self.baseline_throughput = float(
                    np.mean(self._clean_throughput_buf)
                )
                self.baseline_speed = float(
                    np.mean(self._clean_speed_buf)
                )

        # ── Congestion Index ──────────────────────────────────────────────
        if self.baseline_speed > 0:
            ci = max(
                0.0,
                (self.baseline_speed - avg_speed) / self.baseline_speed * 100.0
            )
        else:
            ci = 0.0

        snap = FlowSnapshot(
            timestamp              = now,
            zone_id                = self.zone_id,
            throughput_vpm         = round(throughput_vpm, 2),
            avg_speed_kmh          = round(avg_speed, 1),
            congestion_index       = round(ci, 1),
            vehicle_count_in_zone  = len(detections),
            baseline_throughput    = round(self.baseline_throughput, 2),
            baseline_speed         = round(self.baseline_speed, 1),
            violation_active       = violation_active,
        )
        self.latest = snap
        return snap

    # ── Speed estimation ─────────────────────────────────────────────────────

    def _estimate_speed_kmh(self, track_id: int) -> float | None:
        """
        Estimates speed from pixel displacement over time for a track.
        Uses last 2 positions in history.
        Returns speed in km/h, or None if insufficient history.
        """
        hist = self._tracks.get(track_id)
        if hist is None or len(hist.positions) < 2:
            return None

        p1 = hist.positions[-2]
        p2 = hist.positions[-1]
        t1 = hist.timestamps[-2]
        t2 = hist.timestamps[-1]

        dt = t2 - t1
        if dt <= 0:
            return None

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist_px = (dx**2 + dy**2) ** 0.5
        dist_m  = dist_px / self.px_per_metre
        speed_ms = dist_m / dt
        speed_kmh = speed_ms * 3.6
        return round(speed_kmh, 1)

    # ── Counting line geometry ────────────────────────────────────────────────

    def _crosses_line(
        self, p_prev: tuple, p_curr: tuple
    ) -> bool:
        """
        Detects if the segment p_prev→p_curr crosses the counting line.
        Uses 2D line intersection test.
        """
        def _cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        def _on_segment(p, q, r):
            return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                    min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

        a, b = self.line_p1, self.line_p2
        c, d = p_prev, p_curr

        d1 = _cross(c, d, a)
        d2 = _cross(c, d, b)
        d3 = _cross(a, b, c)
        d4 = _cross(a, b, d)

        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True

        if d1 == 0 and _on_segment(c, a, d): return True
        if d2 == 0 and _on_segment(c, b, d): return True
        if d3 == 0 and _on_segment(a, c, b): return True
        if d4 == 0 and _on_segment(a, d, b): return True

        return False

    # ── Helpers ───────────────────────────────────────────────────────────────

    def delta_throughput_pct(self) -> float:
        """% drop in throughput vs baseline. Positive = degraded."""
        if self.latest is None or self.baseline_throughput == 0:
            return 0.0
        drop = self.baseline_throughput - self.latest.throughput_vpm
        return round(drop / self.baseline_throughput * 100.0, 1)

    def delta_speed_kmh(self) -> float:
        """Absolute speed drop vs baseline. Positive = degraded."""
        if self.latest is None:
            return 0.0
        return round(self.baseline_speed - self.latest.avg_speed_kmh, 1)

    def summary_line(self) -> str:
        if self.latest is None:
            return f"[{self.zone_id}] No data yet."
        s = self.latest
        return (
            f"[{self.zone_id}] q={s.throughput_vpm:.1f}vpm "
            f"v={s.avg_speed_kmh:.1f}km/h "
            f"CI={s.congestion_index:.1f}% "
            f"Δq={self.delta_throughput_pct():.1f}% "
            f"Δv={self.delta_speed_kmh():.1f}km/h"
        )
