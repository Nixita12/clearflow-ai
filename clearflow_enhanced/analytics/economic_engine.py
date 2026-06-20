"""
economic_engine.py  —  Module D: Economic Impact Quantifier
------------------------------------------------------------
Translates traffic congestion into ₹ productivity losses
and commuter-hour waste that government stakeholders understand.

Key outputs:
  - commuter_hours_lost_per_week   : float
  - productivity_loss_inr_lakh     : float (₹ lakhs)
  - fuel_waste_inr                 : float
  - co2_excess_kg                  : float
  - economic_narrative             : str (human-readable for government reports)

Assumptions (RBI / MoRTH calibrated):
  - Average commuter hourly wage (Bengaluru): ₹180/hour (2024)
  - Fuel waste: 0.08 L/min at idle, petrol ₹103/L
  - CO2 emission factor: 2.31 kg CO2/L petrol
  - Peak-hour vehicles on an arterial: 840 veh/hr baseline
"""

from dataclasses import dataclass


# ── Constants (2024 Bengaluru calibration) ───────────────────────────────────
HOURLY_WAGE_INR        = 180.0   # avg commuter
PEAK_VEHICLES_PER_HR   = 840     # arterial road, peak hour
FUEL_WASTE_LPMIN       = 0.08    # L/min per idling vehicle
PETROL_PRICE_INR       = 103.0   # ₹/L (Bengaluru, 2024)
CO2_PER_LITRE_KG       = 2.31    # kg CO2 / L petrol
WORKING_DAYS_PER_YEAR  = 250
WEEKS_PER_YEAR         = 52
INR_LAKH               = 100_000.0


@dataclass
class EconomicImpact:
    commuter_hours_lost_per_event: float     # total commuter-hours for this violation
    commuter_hours_per_week_zone: float      # annualised weekly hours for the zone
    productivity_loss_inr: float             # ₹ this event
    productivity_loss_lakh_weekly: float     # ₹ lakhs/week zone-level
    fuel_waste_inr: float                    # ₹ fuel wasted by idling vehicles
    co2_excess_kg: float                     # kg excess CO2
    economic_narrative: str                  # government-pitch narrative


def compute_economic_impact(
    vehicles_affected_per_hour: int,
    throughput_drop_pct: float,
    duration_seconds: float,
    zone_name: str,
    violations_per_week_zone: int = 30,     # avg from dataset: ~30 events/week/hotspot
) -> EconomicImpact:
    """
    Compute economic impact of a single violation event.

    Parameters
    ----------
    vehicles_affected_per_hour  : estimated vehicles delayed
    throughput_drop_pct         : % throughput reduction
    duration_seconds            : how long the vehicle was parked
    zone_name                   : human-readable zone for narrative
    violations_per_week_zone    : historical weekly violation rate for zone
    """
    duration_min = duration_seconds / 60.0
    duration_hr  = duration_seconds / 3600.0

    # ── Commuter-hours lost (this event) ─────────────────────────────────────
    # Each delayed vehicle loses approximately (throughput_drop_pct/100) × duration
    # as extra wait time
    delay_fraction = throughput_drop_pct / 100.0
    hours_lost_event = vehicles_affected_per_hour * delay_fraction * duration_hr

    # ── Annualised weekly estimate for this zone ──────────────────────────────
    hours_per_week_zone = hours_lost_event * violations_per_week_zone

    # ── Productivity loss ─────────────────────────────────────────────────────
    prod_loss_event = hours_lost_event * HOURLY_WAGE_INR
    prod_loss_weekly_lakh = (
        hours_per_week_zone * HOURLY_WAGE_INR / INR_LAKH
    )

    # ── Fuel waste (idling vehicles × duration) ───────────────────────────────
    # Affected vehicles crawl/idle for ~duration_min extra
    fuel_litres = vehicles_affected_per_hour * delay_fraction * duration_min * FUEL_WASTE_LPMIN
    fuel_waste_inr = fuel_litres * PETROL_PRICE_INR

    # ── CO2 excess ────────────────────────────────────────────────────────────
    co2_kg = fuel_litres * CO2_PER_LITRE_KG

    # ── Narrative ─────────────────────────────────────────────────────────────
    weekly_hours = round(hours_per_week_zone)
    weekly_lakh  = round(prod_loss_weekly_lakh, 2)
    event_hours  = round(hours_lost_event, 1)
    event_prod   = round(prod_loss_event)
    fuel_str     = round(fuel_waste_inr)
    co2_str      = round(co2_kg, 1)

    narrative = (
        f"This single violation at {zone_name} cost approximately "
        f"{event_hours} commuter-hours and ₹{event_prod:,} in productivity. "
        f"At current violation rates, {zone_name} alone wastes ~{weekly_hours:,} commuter hours/week, "
        f"equivalent to ₹{weekly_lakh} lakh/week in lost productivity "
        f"(₹{round(weekly_lakh * WEEKS_PER_YEAR, 1)} lakh/year). "
        f"Fuel wasted by idling vehicles: ₹{fuel_str:,}. "
        f"Excess CO₂ emitted: {co2_str} kg. "
        f"Deploying ClearFlow AI citywide across all 8 hotspots could recover "
        f"an estimated ₹{round(weekly_lakh * 8 * WEEKS_PER_YEAR, 0):,.0f} lakh "
        f"in annual productivity."
    )

    return EconomicImpact(
        commuter_hours_lost_per_event = round(hours_lost_event, 2),
        commuter_hours_per_week_zone  = round(hours_per_week_zone, 1),
        productivity_loss_inr         = round(prod_loss_event, 2),
        productivity_loss_lakh_weekly = round(prod_loss_weekly_lakh, 4),
        fuel_waste_inr                = round(fuel_waste_inr, 2),
        co2_excess_kg                 = round(co2_kg, 2),
        economic_narrative            = narrative,
    )
