"""
household.py — A single household's health state machine.

What changed from the static version:
  • Real (x, y) coordinates are stored here
  • road_quality stored per household
  • tick(season) uses seasonal multipliers — monsoon worsens newborns faster,
    winter worsens TB patients faster, etc.
  • est_visit_duration_min() gives the agent a planning estimate BEFORE visiting
  • actual_visit_duration_min() computes the real time AFTER seeing what's inside
  • receive_visit(energy_pct) — ASHA fatigue reduces visit effectiveness
    below 30% energy: reset is weaker, visit takes longer
"""

import math
import random
from typing import Optional


# ── Visit durations in minutes by category ───────────────────────────────────

BASE_VISIT_MIN = {
    "newborn":        35,   # full neonatal assessment, breastfeeding check
    "tb_patient":     25,   # DOTS supervision + symptom review
    "high_risk_preg": 40,   # BP, fundal height, danger-sign screening
    "diabetic":       20,   # BP, blood-glucose check, foot inspection
    "routine":        15,   # wellness check, immunisation status
}

DANGER_SIGN_EXTRA_MIN   = 20   # extra assessment when danger sign found
REFERRAL_PREP_EXTRA_MIN = 25   # writing slip, arranging transport


# ── Seasonal deterioration rate multipliers ──────────────────────────────────
# Encodes clinical reality:
#   Monsoon → malaria + waterborne diseases surge; heat + humidity worsen maternal outcomes
#   Winter  → cold + respiratory infections hit newborns and TB patients hardest

SEASONAL_MULTIPLIERS = {
    "summer": {
        "newborn": 1.0, "tb_patient": 1.0,
        "high_risk_preg": 1.0, "diabetic": 1.0, "routine": 1.0,
    },
    "monsoon": {
        "newborn":        1.5,   # sepsis + waterborne fever surge
        "tb_patient":     1.2,   # humidity worsens respiratory TB
        "high_risk_preg": 1.4,   # heat, dehydration, eclampsia risk
        "diabetic":       1.1,
        "routine":        2.0,   # malaria + waterborne illnesses spike sharply
    },
    "winter": {
        "newborn":        1.6,   # hypothermia + pneumonia — highest winter risk
        "tb_patient":     1.4,   # cold air worsens pulmonary TB markedly
        "high_risk_preg": 1.1,
        "diabetic":       1.3,   # cold → poor circulation
        "routine":        1.5,   # respiratory surge
    },
}

BASE_DETERIORATION_RATES = {
    "newborn":        0.12,
    "tb_patient":     0.06,
    "high_risk_preg": 0.04,
    "diabetic":       0.02,
    "routine":        0.005,
}


class Household:
    """
    One household in the village.

    Attributes added vs static version:
        x, y            — real coordinates in km
        road_quality    — 0 (mud) to 1 (paved), used for travel time
        days_since_birth— for newborns, tracks 48-hour critical window
    """

    def __init__(
        self,
        id: int,
        category: str,
        geo_cluster: int,
        x: float,
        y: float,
        road_quality: float,
        rng: random.Random,
    ):
        self.id = id
        self.category = category
        self.geo_cluster = geo_cluster
        self.x = x
        self.y = y
        self.road_quality = road_quality
        self.rng = rng

        # Health state
        self.risk_score: float = self._initial_risk()
        self.days_since_visit: int = rng.randint(0, 2)
        self.danger_sign_active: bool = self.risk_score > 0.75
        self.known_to_asha: bool = True
        self.is_dead: bool = False

        # TB tracking
        self._tb_window_counter: int = 0
        self.tb_doses_missed: int = 0
        self.tb_doses_given: int = 0

        # Newborn tracking
        self.days_since_birth: Optional[int] = (
            rng.randint(0, 2) if category == "newborn" else None
        )

        # Stats
        self.total_visits: int = 0

    # ── Initialisation ───────────────────────────────────────────────────────

    def _initial_risk(self) -> float:
        base = {
            "newborn":        0.40,
            "tb_patient":     0.30,
            "high_risk_preg": 0.30,
            "diabetic":       0.20,
            "routine":        0.05,
        }[self.category]
        return min(0.80, base + self.rng.uniform(0.0, 0.15))

    # ── Spatial helpers ──────────────────────────────────────────────────────

    def dist_from(self, ox: float, oy: float) -> float:
        """Straight-line distance in km from point (ox, oy)."""
        return math.sqrt((self.x - ox) ** 2 + (self.y - oy) ** 2)

    # ── Visit duration estimation ────────────────────────────────────────────

    def est_visit_duration_min(self) -> int:
        """
        Estimated visit time BEFORE entering — the agent uses this for planning.
        Adds time if danger sign is already visible.
        """
        base = BASE_VISIT_MIN[self.category]
        if self.danger_sign_active:
            base += DANGER_SIGN_EXTRA_MIN
        return base

    def _actual_visit_duration_min(
        self, found_danger: bool, needs_referral: bool, energy_pct: float
    ) -> int:
        """
        Actual time AFTER entering, based on what's found.
        Low ASHA energy means each task takes longer.
        """
        base = BASE_VISIT_MIN[self.category]
        extra = 0
        if found_danger:
            extra += DANGER_SIGN_EXTRA_MIN
        if needs_referral:
            extra += REFERRAL_PREP_EXTRA_MIN

        # Energy penalty: each 10% below 60% adds 8% more time
        if energy_pct < 60:
            energy_factor = 1.0 + ((60 - energy_pct) / 100) * 0.8
        else:
            energy_factor = 1.0

        return max(10, int((base + extra) * energy_factor))

    # ── Daily tick (no visit today) ──────────────────────────────────────────

    def tick(self, season: str = "summer") -> bool:
        """
        Called every day this household is NOT visited.
        Returns True if a preventable death just occurred.

        Key dynamics:
        - Rate is multiplied by the seasonal factor
        - Longer without a visit → compounding penalty (days_penalty)
        - TB tracks missed dose windows (every 3 days without visit = 1 missed dose)
        """
        if self.is_dead:
            return False

        base_rate = BASE_DETERIORATION_RATES.get(self.category, 0.01)
        seasonal_mult = SEASONAL_MULTIPLIERS[season].get(self.category, 1.0)

        # Compounding: risk grows faster the longer the household has been missed.
        # 0 days missed → ×1.0; 5 days missed → ×1.25; 10 days → ×1.50
        days_compound = 1.0 + (self.days_since_visit * 0.05)

        self.risk_score = min(
            1.0,
            self.risk_score + base_rate * seasonal_mult * days_compound,
        )
        self.days_since_visit += 1

        if self.days_since_birth is not None:
            self.days_since_birth += 1

        if self.risk_score > 0.75:
            self.danger_sign_active = True

        # Death condition: critical risk held for 3+ days without visit
        if self.risk_score >= 0.97 and self.days_since_visit >= 3:
            self.is_dead = True
            return True  # preventable death

        # TB dose tracking: every 3-day window without visit = one missed dose
        if self.category == "tb_patient":
            self._tb_window_counter += 1
            if self._tb_window_counter > 0 and self._tb_window_counter % 3 == 0:
                self.tb_doses_missed += 1

        return False

    # ── Visit execution ──────────────────────────────────────────────────────

    def receive_visit(self, energy_pct: float = 100.0) -> dict:
        """
        Execute a visit.  Returns a dict describing what was found,
        what was done, and how long the visit actually took.

        energy_pct effects:
          ≥ 60 → full effectiveness (risk reset to 15% of original)
          < 60 → partial effectiveness (reset to 25–35%) + longer visit
          < 30 → visit quality drops noticeably
        """
        found = {
            "danger_sign":      self.danger_sign_active,
            "risk_before":      self.risk_score,
            "referral_needed":  self.risk_score > 0.75,
            "category":         self.category,
            "days_since_visit": self.days_since_visit,
            "days_since_birth": self.days_since_birth,
            "tb_dose_on_time": (
                self.category == "tb_patient" and self.days_since_visit <= 3
            ),
            "newborn_48hr": (
                self.category == "newborn"
                and self.days_since_birth is not None
                and self.days_since_birth <= 2
            ),
        }

        # Actual visit duration (may be longer than estimated)
        found["visit_duration_min"] = self._actual_visit_duration_min(
            found["danger_sign"], found["referral_needed"], energy_pct
        )

        # Risk reset — quality depends on ASHA energy
        if energy_pct >= 60:
            reset_factor = 0.15    # full visit: cuts risk to 15% of original
        elif energy_pct >= 30:
            reset_factor = 0.25    # tired ASHA: less thorough
        else:
            reset_factor = 0.40    # exhausted: minimal benefit

        self.risk_score = max(0.05, self.risk_score * reset_factor)
        self.days_since_visit = 0
        self.danger_sign_active = False
        self._tb_window_counter = 0
        self.total_visits += 1

        if self.category == "tb_patient":
            self.tb_doses_given += 1

        return found
