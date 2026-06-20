# ClearFlow AI — Instructions to Run

## Requirements
- Python 3.10+
- `pip install -r requirements.txt` (installs pandas, pyyaml, numpy, opencv-python)

## Option A — Full Analytics Demo (recommended for judging, no camera needed)

Runs all 18 analytics modules against the real 298,450-record Bengaluru
parking violation dataset (Nov 2023 – Apr 2024). Produces a full console
report plus JSON outputs.

```bash
cd clearflow_enhanced
pip install -r requirements.txt
python dataset_pipeline.py --dataset data/dataset.csv --config config/clearflow_config.yaml --top 20 --sample 1000
```

Expect ~15-20 seconds runtime. Outputs:
- Console: enforcement priority report, hotspot intelligence, congestion
  forecast, patrol schedule, post-dispatch learning summary, Urban Mobility
  Risk Index, Recovery AI recommendations, Emergency Corridor alerts,
  Digital Twin what-if simulations, Towing priority queue, RL allocator
  learned values, Multi-Vehicle Surge Detection, Citywide Budget Optimizer,
  and a scripted TrafficGPT Q&A demo
- `output/reports/priority_report.json` — top-N scored violations
- `output/reports/surge_events.json` — detected coordinated multi-vehicle events
- `output/reports/budget_allocation.json` — optimal officer/tow allocation

To run on the full dataset instead of a sample, set `--sample 0`. Note this
will take longer (the full junction dataset is ~150,000 records).

## Option B — Live Detection Demo (synthetic frames, no camera needed)

Demonstrates the real-time detection pipeline (Module A: perception +
tracking, Module B: flow analysis) firing an actual violation end-to-end
in ~8 seconds, with full economic/confidence/trend output per event.

```bash
cd clearflow_enhanced
python main.py --config config/clearflow_config.yaml --demo
```

This uses synthetic frames (no camera/video file required) so it runs
identically on any machine. It will print 3 violation alerts with impact
scores, ML confidence, economic cost (₹, commuter-hours, CO₂), and a
final session summary.

**Note on demo mode:** the YOLO object-detection model (`ultralytics`
package) is not bundled in this submission to keep the package small — if
it isn't installed, `main.py` automatically falls back to a stub detector
with synthetic vehicle positions so the rest of the pipeline (tracking,
flow analysis, scoring, economic quantification) can still be evaluated
end-to-end. To run against a real video file or webcam with actual YOLO
inference, install `ultralytics` (`pip install ultralytics`) and run:

```bash
python main.py --config config/clearflow_config.yaml --source path/to/video.mp4 --zone BTP051
# or for a webcam:
python main.py --config config/clearflow_config.yaml --source 0 --zone BTP051
```

## Troubleshooting
- If `data/dataset.csv` is missing, place the provided CSV at that exact
  path (or pass `--dataset /full/path/to/your.csv`).
- All commands above were verified to exit with code 0 and produce
  non-empty output as of this submission.
