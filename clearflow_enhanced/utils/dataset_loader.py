"""
dataset_loader.py
-----------------
Loads and preprocesses jan_to_may_police_violation_anonymized.csv
Provides historical hotspot intelligence to feed the live pipeline.

Usage:
    from utils.dataset_loader import DatasetLoader
    dl = DatasetLoader("path/to/dataset.csv")
    hotspots = dl.get_hotspot_risk_table()
"""

import ast
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


OFFENCE_CODE_MAP = {
    112: "WRONG_PARKING",
    113: "NO_PARKING",
    107: "PARKING_MAIN_ROAD",
    116: "DEFECTIVE_NUMBER_PLATE",
    105: "PARKING_ON_FOOTPATH",
    111: "PARKING_NEAR_BUSSTOP_SCHOOL_HOSPITAL",
    109: "DOUBLE_PARKING",
    104: "PARKING_NEAR_ROAD_CROSSING",
    124: "OTHER",
}

# Map dataset vehicle_type strings → pipeline canonical names
VEHICLE_TYPE_MAP = {
    "CAR": "CAR",
    "SCOOTER": "SCOOTER",
    "MOTOR CYCLE": "MOTOR_CYCLE",
    "PASSENGER AUTO": "PASSENGER_AUTO",
    "MAXI-CAB": "MAXI_CAB",
    "LGV": "LGV",
    "GOODS AUTO": "PASSENGER_AUTO",
    "MOPED": "MOPED",
    "PRIVATE BUS": "PRIVATE_BUS",
    "BUS (BMTC/KSRTC)": "BUS",
    "VAN": "VAN",
    "TEMPO": "VAN",
    "HGV": "HGV",
    "LORRY/GOODS VEHICLE": "LORRY",
    "TANKER": "TANKER",
    "JEEP": "CAR",
    "MINI LORRY": "LGV",
    "SCHOOL VEHICLE": "BUS",
    "TOURIST BUS": "BUS",
    "FACTORY BUS": "BUS",
    "TRACTOR": "HGV",
    "OTHERS": "DEFAULT",
}


class DatasetLoader:
    """
    Wraps the historical violation CSV and exposes pre-computed
    intelligence tables for the live detection pipeline.
    """

    def __init__(self, csv_path: str):
        self.path = Path(csv_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {csv_path}")
        logger.info(f"Loading dataset: {self.path}")
        self.df = self._load_and_clean()
        logger.info(f"Loaded {len(self.df):,} records.")

    # ── Private ──────────────────────────────────────────────────────────────

    def _load_and_clean(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)

        # Datetime parsing (mixed format: some rows have microseconds, some don't)
        df["created_datetime"] = pd.to_datetime(
            df["created_datetime"], format="mixed", utc=True
        )

        # Derived time columns
        df["hour"] = df["created_datetime"].dt.hour
        df["month"] = df["created_datetime"].dt.month
        df["day_of_week"] = df["created_datetime"].dt.day_name()

        # Normalise vehicle type
        df["vehicle_type_canonical"] = (
            df["vehicle_type"].map(VEHICLE_TYPE_MAP).fillna("DEFAULT")
        )

        # Parse violation_type JSON list → flat string
        df["violation_primary"] = df["violation_type"].apply(
            self._parse_primary_violation
        )

        # Parse offence_code JSON list → first code
        df["offence_code_primary"] = df["offence_code"].apply(
            self._parse_primary_code
        )
        df["offence_name"] = df["offence_code_primary"].map(OFFENCE_CODE_MAP)

        # is_junction flag
        df["is_junction"] = df["junction_name"] != "No Junction"

        # Approved only (mirrors data_sent_to_scita = True filter used by BTP)
        df["is_approved"] = df["validation_status"] == "approved"

        return df

    @staticmethod
    def _parse_primary_violation(val: str) -> str:
        try:
            lst = ast.literal_eval(val)
            return lst[0] if lst else "UNKNOWN"
        except Exception:
            return "UNKNOWN"

    @staticmethod
    def _parse_primary_code(val) -> int:
        try:
            lst = ast.literal_eval(str(val))
            return int(lst[0]) if lst else -1
        except Exception:
            return -1

    # ── Public API ───────────────────────────────────────────────────────────

    def get_hotspot_risk_table(self) -> pd.DataFrame:
        """
        Returns a DataFrame of junction hotspots with:
        - historical violation count
        - dominant vehicle type
        - dominant violation type
        - peak hour
        - normalised risk score (0–1) for seeding the Impact Score engine
        """
        jdf = self.df[self.df["is_junction"]].copy()

        risk = (
            jdf.groupby("junction_name")
            .agg(
                violation_count=("id", "count"),
                lat=("latitude", "mean"),
                lon=("longitude", "mean"),
                police_station=("police_station", lambda x: x.mode()[0]),
                dominant_vehicle=(
                    "vehicle_type_canonical",
                    lambda x: x.value_counts().index[0],
                ),
                dominant_violation=(
                    "violation_primary",
                    lambda x: x.value_counts().index[0],
                ),
                peak_hour=(
                    "hour",
                    lambda x: x.value_counts().index[0],
                ),
                approved_pct=("is_approved", "mean"),
            )
            .reset_index()
        )

        max_count = risk["violation_count"].max()
        risk["historical_risk_score"] = (
            risk["violation_count"] / max_count
        ).round(3)

        return risk.sort_values("violation_count", ascending=False)

    def get_time_of_day_weights(self) -> dict:
        """
        Returns hour→weight dict (0–23) normalised so peak hour = 1.0.
        Derived from actual dataset hourly distribution.
        """
        hourly = self.df["hour"].value_counts().sort_index()
        max_count = hourly.max()
        return {int(h): round(c / max_count, 3) for h, c in hourly.items()}

    def get_vehicle_type_distribution(self) -> dict:
        """Returns vehicle_type → count from dataset."""
        return (
            self.df["vehicle_type_canonical"]
            .value_counts()
            .to_dict()
        )

    def get_police_station_load(self) -> pd.DataFrame:
        """Station-level workload table for dispatch prioritisation."""
        return (
            self.df.groupby("police_station")
            .agg(
                total_violations=("id", "count"),
                approved=("is_approved", "sum"),
                junction_violations=("is_junction", "sum"),
                center_code=("center_code", "first"),
            )
            .sort_values("total_violations", ascending=False)
            .reset_index()
        )

    def get_approved_records(self) -> pd.DataFrame:
        """Only BTP-approved records — mirrors data_sent_to_scita=True."""
        return self.df[self.df["is_approved"]].copy()

    def summary(self):
        """Print a quick dataset summary."""
        df = self.df
        print(f"{'─'*50}")
        print(f"  ClearFlow AI — Dataset Summary")
        print(f"{'─'*50}")
        print(f"  Total records   : {len(df):,}")
        print(f"  Date range      : {df['created_datetime'].min().date()} → "
              f"{df['created_datetime'].max().date()}")
        print(f"  Unique stations : {df['police_station'].nunique()}")
        print(f"  Unique junctions: "
              f"{df[df['is_junction']]['junction_name'].nunique()}")
        print(f"  Approved records: {df['is_approved'].sum():,} "
              f"({df['is_approved'].mean()*100:.1f}%)")
        print(f"  Junction records: {df['is_junction'].sum():,} "
              f"({df['is_junction'].mean()*100:.1f}%)")
        print(f"  Top vehicle     : {df['vehicle_type_canonical'].mode()[0]}")
        print(f"  Peak hour       : {int(df['hour'].mode()[0]):02d}:00")
        print(f"{'─'*50}")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    path = sys.argv[1] if len(sys.argv) > 1 else "data/dataset.csv"
    dl = DatasetLoader(path)
    dl.summary()
    print("\nTop 5 hotspots:")
    print(dl.get_hotspot_risk_table().head(5)[
        ["junction_name", "violation_count", "dominant_vehicle",
         "peak_hour", "historical_risk_score"]
    ].to_string(index=False))
