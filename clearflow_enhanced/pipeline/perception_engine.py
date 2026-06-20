"""
perception_engine.py  —  Module A
----------------------------------
Runs YOLO inference on each frame and flags vehicles whose centroid
has remained inside a "No Parking" polygon for > T_threshold seconds.

Inputs  : video frame (numpy array), zone config from YAML
Outputs : list of Detection objects (one per vehicle bounding box)
          list of IllegalParkingEvent objects (subset that crossed T_threshold)

Dependencies:
    pip install ultralytics opencv-python-headless pyyaml numpy
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """Single YOLO detection for one vehicle in one frame."""
    track_id: int
    class_id: int
    class_name: str          # canonical name from VEHICLE_TYPE_MAP
    confidence: float
    bbox: tuple              # (x1, y1, x2, y2) in pixels
    centroid: tuple          # (cx, cy) in pixels
    frame_idx: int
    timestamp: float         # epoch seconds

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return float((x2 - x1) * (y2 - y1))


@dataclass
class IllegalParkingEvent:
    """
    Raised when a tracked vehicle has been stationary inside
    a no-parking polygon for >= T_threshold seconds.
    """
    event_id: str
    track_id: int
    vehicle_type: str
    zone_id: str                    # e.g. "BTP051"
    zone_name: str
    police_station: str
    lat: float
    lon: float
    centroid_px: tuple
    bbox: tuple
    first_seen_ts: float
    flagged_ts: float
    duration_sec: float
    offence_code: int               # 113 = NO_PARKING, 112 = WRONG_PARKING
    frame_idx: int

    @property
    def offence_name(self) -> str:
        _map = {112: "WRONG_PARKING", 113: "NO_PARKING",
                107: "PARKING_MAIN_ROAD", 105: "PARKING_ON_FOOTPATH"}
        return _map.get(self.offence_code, "PARKING_VIOLATION")


# ── COCO class ID → canonical vehicle name ────────────────────────────────────
COCO_TO_VEHICLE = {
    2: "CAR",
    3: "MOTOR_CYCLE",   # covers SCOOTER, MOPED too
    5: "BUS",           # covers MAXI_CAB, PRIVATE_BUS
    7: "TRUCK",         # covers LGV, HGV, LORRY
}

# Offence code logic: junction zone → 113 (NO_PARKING), main road → 112
def _infer_offence_code(zone_id: str, is_junction: bool) -> int:
    return 113 if is_junction else 112


# ── Perception Engine ─────────────────────────────────────────────────────────

class PerceptionEngine:
    """
    Wraps YOLOv8 + ByteTrack for per-frame vehicle detection and
    stationary-vehicle flagging inside no-parking polygons.
    """

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        mc = self.cfg["model"]
        self.conf_thresh = mc["confidence_threshold"]
        self.iou_thresh  = mc["iou_threshold"]
        self.input_size  = mc["input_size"]
        self.device      = mc["device"]
        self.target_ms   = mc["inference_target_ms"]

        # Classes we care about (COCO IDs for vehicles)
        self.vehicle_class_ids = set(COCO_TO_VEHICLE.keys())  # {2,3,5,7}

        pc = self.cfg["parking"]
        self.stationary_thresh_s = pc["stationary_threshold_seconds"]
        self.movement_px         = pc["centroid_movement_px"]
        self.min_frames_stat     = pc["min_frames_stationary"]

        # Zone polygons loaded from config
        self.zones = self._build_zone_polygons()

        # Per-track state: {track_id: {centroid, first_stationary_ts, frames_stationary}}
        self._track_state: dict = {}
        # Already-fired events (avoid re-firing for same track in same zone)
        self._fired_events: set = set()

        self.model = None  # loaded lazily

    # ── Setup ────────────────────────────────────────────────────────────────

    def _build_zone_polygons(self) -> list:
        """Convert YAML roi_polygon lists to numpy arrays for cv2.pointPolygonTest."""
        zones = []
        for z in self.cfg.get("hotspot_zones", []):
            poly = np.array(z["roi_polygon"], dtype=np.int32)
            zones.append({**z, "polygon_np": poly})
        return zones

    def load_model(self):
        """Lazy-load YOLO. Call once before the frame loop."""
        try:
            from ultralytics import YOLO
            weights = self.cfg["model"]["weights"]
            logger.info(f"Loading YOLO model: {weights} on device={self.device}")
            self.model = YOLO(weights)
            logger.info("Model loaded.")
        except ImportError:
            logger.warning(
                "ultralytics not installed. "
                "Run: pip install ultralytics\n"
                "Running in STUB mode — returning synthetic detections."
            )
            self.model = None

    # ── Core Methods ─────────────────────────────────────────────────────────

    def infer(self, frame: np.ndarray, frame_idx: int) -> list[Detection]:
        """
        Run YOLO+ByteTrack on a single frame.
        Returns list of Detection objects for tracked vehicles only.
        """
        ts = time.time()

        if self.model is None:
            return self._stub_detections(frame, frame_idx, ts)

        t0 = time.perf_counter()
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.input_size,
            device=self.device,
            tracker="bytetrack.yaml",
            classes=list(self.vehicle_class_ids),
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if elapsed_ms > self.target_ms:
            logger.warning(f"Inference {elapsed_ms:.1f}ms > target {self.target_ms}ms")

        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                if boxes.id is None:
                    continue
                track_id = int(boxes.id[i])
                cls_id   = int(boxes.cls[i])
                conf     = float(boxes.conf[i])
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                detections.append(Detection(
                    track_id  = track_id,
                    class_id  = cls_id,
                    class_name= COCO_TO_VEHICLE.get(cls_id, "UNKNOWN"),
                    confidence= conf,
                    bbox      = (x1, y1, x2, y2),
                    centroid  = (cx, cy),
                    frame_idx = frame_idx,
                    timestamp = ts,
                ))

        return detections

    def check_illegal_parking(
        self,
        detections: list[Detection],
        frame_idx: int,
    ) -> list[IllegalParkingEvent]:
        """
        For each detection, check if its centroid is inside a no-parking
        polygon and has been stationary for >= T_threshold seconds.

        Returns list of new IllegalParkingEvent objects this frame.
        """
        events = []
        ts_now = time.time()

        for det in detections:
            tid = det.track_id
            cx, cy = det.centroid

            # Which zone is this centroid in?
            zone = self._find_zone(cx, cy)
            if zone is None:
                # Outside all monitored zones — reset state
                self._track_state.pop(tid, None)
                continue

            state = self._track_state.get(tid)

            if state is None:
                # First time we see this track in a zone
                self._track_state[tid] = {
                    "centroid": (cx, cy),
                    "first_stationary_ts": ts_now,
                    "frames_stationary": 1,
                    "zone_id": zone["id"],
                }
                continue

            prev_cx, prev_cy = state["centroid"]
            movement = ((cx - prev_cx)**2 + (cy - prev_cy)**2) ** 0.5

            if movement < self.movement_px:
                # Vehicle hasn't moved
                state["frames_stationary"] += 1
                state["centroid"] = (cx, cy)

                if state["frames_stationary"] < self.min_frames_stat:
                    continue

                duration = ts_now - state["first_stationary_ts"]

                if duration >= self.stationary_thresh_s:
                    event_key = (tid, zone["id"])
                    if event_key not in self._fired_events:
                        self._fired_events.add(event_key)
                        event = IllegalParkingEvent(
                            event_id       = f"VID-{zone['id']}-{tid}-{frame_idx}",
                            track_id       = tid,
                            vehicle_type   = det.class_name,
                            zone_id        = zone["id"],
                            zone_name      = zone["name"],
                            police_station = zone["police_station"],
                            lat            = zone["lat"],
                            lon            = zone["lon"],
                            centroid_px    = (cx, cy),
                            bbox           = det.bbox,
                            first_seen_ts  = state["first_stationary_ts"],
                            flagged_ts     = ts_now,
                            duration_sec   = duration,
                            offence_code   = _infer_offence_code(
                                zone["id"], is_junction=True
                            ),
                            frame_idx      = frame_idx,
                        )
                        events.append(event)
                        logger.info(
                            f"[VIOLATION] {event.event_id} | "
                            f"{det.class_name} | {zone['name']} | "
                            f"{duration:.0f}s stationary"
                        )
            else:
                # Vehicle moved — reset stationary timer
                self._track_state[tid] = {
                    "centroid": (cx, cy),
                    "first_stationary_ts": ts_now,
                    "frames_stationary": 0,
                    "zone_id": zone["id"],
                }
                # If it was previously flagged, allow re-flagging if it parks again
                self._fired_events.discard((tid, zone["id"]))

        return events

    def _find_zone(self, cx: int, cy: int) -> Optional[dict]:
        """Returns the first zone whose polygon contains (cx, cy), or None."""
        for zone in self.zones:
            dist = cv2.pointPolygonTest(zone["polygon_np"], (cx, cy), False)
            if dist >= 0:
                return zone
        return None

    # ── Visualisation helper ─────────────────────────────────────────────────

    def draw_annotations(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        events: list[IllegalParkingEvent],
    ) -> np.ndarray:
        """
        Draws bounding boxes, track IDs, zone polygons, and
        red VIOLATION banner for any flagged vehicle.
        """
        annotated = frame.copy()
        event_track_ids = {e.track_id for e in events}

        # Draw zone polygons
        for zone in self.zones:
            cv2.polylines(
                annotated,
                [zone["polygon_np"]],
                isClosed=True,
                color=(0, 165, 255),   # orange
                thickness=2,
            )
            # Zone label
            pt = tuple(zone["polygon_np"][0])
            cv2.putText(
                annotated, zone["id"],
                pt, cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 165, 255), 1,
            )

        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            is_violation = det.track_id in event_track_ids
            color = (0, 0, 220) if is_violation else (0, 200, 0)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"#{det.track_id} {det.class_name}"
            if is_violation:
                label += " [VIOLATION]"
            cv2.putText(
                annotated, label,
                (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
            )

        return annotated

    # ── Stub for demo without camera ─────────────────────────────────────────

    def _stub_detections(
        self, frame: np.ndarray, frame_idx: int, ts: float
    ) -> list[Detection]:
        """
        Returns synthetic detections for testing without a real model.
        Simulates 3 vehicles, one of which is stationary inside BTP051 zone.
        """
        h, w = frame.shape[:2]
        stub = [
            Detection(1, 2, "CAR",        0.88, (100, 200, 200, 300),  (150, 250), frame_idx, ts),
            Detection(2, 3, "SCOOTER",     0.76, (300, 220, 370, 290),  (335, 255), frame_idx, ts),
            Detection(3, 5, "MAXI_CAB",    0.91, (180, 280, 350, 390),  (265, 335), frame_idx, ts),
        ]
        return stub
