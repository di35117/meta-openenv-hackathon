"""
village.py — Manages all households, the spatial map, weather, and
             season-driven disease dynamics.

Key additions vs static version:

SPATIAL MAP
  Village is a 10 km × 10 km grid.
  5 clusters are placed at real geographic positions:
      0 = North  (5, 8)      1 = South  (5, 2)
      2 = East   (8, 5)      3 = West   (2, 5)
      4 = Centre (5, 5)
  Each household is scattered around its cluster centre with Gaussian noise.
  ASHA home is at (5, 5).  PHC (referral destination) is at (5, 9).

ROAD NETWORK
  A horizontal main road runs at y = 5.0.
  A vertical secondary road runs at x = 5.0.
  Households close to either road get higher road_quality.
  Quality degrades with distance (paved → gravel → dirt path).

TRAVEL TIME
  travel_time_minutes(from, to, road_quality, weather)
  Walking speed varies:
      sunny  + paved   → 5.0 km/h
      rainy  + paved   → 3.5 km/h
      sunny  + dirt    → 3.5 km/h
      rainy  + dirt    → 1.5 km/h  ← monsoon reality
      heavy_rain + dirt → 0.8 km/h ← roads become streams

WEATHER ENGINE
  Each day generates fresh weather drawn from a seasonal distribution.
  Weather affects:
    • road_quality_modifier (multiplied into every household's road_quality)
    • disease event rates
    • ASHA travel speed

SEASONAL DISEASE DYNAMICS
  Monsoon: alert_rate 3× higher, waterborne and malaria events
  Winter:  alert_rate 2× higher, respiratory events
  Disease spikes arrive as Alerts in the observation.
"""

import math
import random
from typing import Dict, List, Tuple, Optional

from .household import Household

# ── Village constants ────────────────────────────────────────────────────────

VILLAGE_KM = 10.0

# Geographic cluster centres (x, y) in km
CLUSTER_CENTRES: Dict[int, Tuple[float, float]] = {
    0: (5.0, 8.0),   # North
    1: (5.0, 2.0),   # South
    2: (8.0, 5.0),   # East
    3: (2.0, 5.0),   # West
    4: (5.0, 5.0),   # Centre
}

ASHA_HOME: Tuple[float, float] = (5.0, 5.0)   # Centre of village
PHC_LOCATION: Tuple[float, float] = (5.0, 9.2) # Primary Health Centre — north

MAIN_ROAD_Y = 5.0   # Horizontal paved road
MAIN_ROAD_X = 5.0   # Vertical secondary road


# ── Road quality calculation ─────────────────────────────────────────────────

def _compute_road_quality(x: float, y: float, rng: random.Random) -> float:
    """
    Assign road quality based on proximity to the road network.
    Households near the main road get paved quality.
    Households far from any road get dirt-path quality.
    Random variation adds realism.
    """
    dist_h = abs(y - MAIN_ROAD_Y)   # distance to horizontal road
    dist_v = abs(x - MAIN_ROAD_X)   # distance to vertical road
    min_dist = min(dist_h, dist_v)

    if min_dist < 0.4:
        # Essentially on the road — paved
        quality = rng.uniform(0.80, 0.98)
    elif min_dist < 1.0:
        # Short walk to road — gravel / semi-paved
        quality = rng.uniform(0.50, 0.75)
    elif min_dist < 2.0:
        # Off the beaten path — dirt
        quality = rng.uniform(0.25, 0.50)
    elif min_dist < 3.5:
        # Remote — rough track
        quality = rng.uniform(0.12, 0.30)
    else:
        # Deep rural — footpath only
        quality = rng.uniform(0.05, 0.18)

    return round(min(1.0, quality), 3)


# ── Travel speed model ───────────────────────────────────────────────────────

def _walking_speed_kmh(road_quality: float, weather: str) -> float:
    """
    Effective walking speed given road quality and weather.

    This is the core dynamic that makes distance meaningful:
    - A household 2 km away on paved road in sun  → 24 min travel
    - A household 2 km away on dirt in heavy rain  → 150 min travel
    """
    # Base speed from road quality
    if road_quality >= 0.70:
        base = 5.0
    elif road_quality >= 0.45:
        base = 4.0
    elif road_quality >= 0.25:
        base = 3.0
    else:
        base = 2.0   # footpath — slow even in good weather

    # Weather multiplier
    weather_mult = {
        "sunny":      1.00,
        "cloudy":     0.95,
        "rainy":      0.65,
        "heavy_rain": 0.40,
    }.get(weather, 1.0)

    # Dirt paths become near-impassable in heavy rain
    if road_quality < 0.30 and weather == "heavy_rain":
        weather_mult *= 0.50   # streams form, paths flood
    elif road_quality < 0.30 and weather == "rainy":
        weather_mult *= 0.70   # mud slows movement significantly

    return max(0.5, base * weather_mult)


def travel_time_minutes(
    from_x: float, from_y: float,
    to_x: float,   to_y: float,
    road_quality: float,
    weather_condition: str,
) -> float:
    """
    Compute travel time in minutes between two points.
    Uses straight-line (crow-flies) distance — a deliberate simplification
    that still captures the essential tradeoffs.
    """
    distance_km = math.sqrt((to_x - from_x) ** 2 + (to_y - from_y) ** 2)
    speed = _walking_speed_kmh(road_quality, weather_condition)
    return round((distance_km / speed) * 60.0, 1)


# ── Weather engine ───────────────────────────────────────────────────────────

# Seasonal probability distributions for weather conditions
SEASONAL_WEATHER_DIST: Dict[str, Dict[str, float]] = {
    "summer": {
        "sunny": 0.60, "cloudy": 0.28, "rainy": 0.10, "heavy_rain": 0.02,
    },
    "monsoon": {
        "sunny": 0.08, "cloudy": 0.22, "rainy": 0.42, "heavy_rain": 0.28,
    },
    "winter": {
        "sunny": 0.52, "cloudy": 0.34, "rainy": 0.11, "heavy_rain": 0.03,
    },
}

# Temperature ranges by season
TEMP_RANGES = {
    "summer":  (36, 46),
    "monsoon": (28, 36),
    "winter":  (10, 22),
}

# Rainfall by condition
RAINFALL_RANGES = {
    "sunny":      (0,  0),
    "cloudy":     (0,  2),
    "rainy":      (5, 28),
    "heavy_rain": (30, 90),
}

# How much weather degrades road quality
WEATHER_ROAD_MODIFIER = {
    "sunny":      1.00,
    "cloudy":     0.95,
    "rainy":      0.68,
    "heavy_rain": 0.38,
}


class WeatherState:
    def __init__(self, condition: str, temp_celsius: float, rainfall_mm: float):
        self.condition = condition
        self.temp_celsius = temp_celsius
        self.rainfall_mm = rainfall_mm
        self.road_quality_modifier = WEATHER_ROAD_MODIFIER[condition]

    def to_dict(self) -> dict:
        return {
            "condition":            self.condition,
            "temp_celsius":         self.temp_celsius,
            "rainfall_mm":          self.rainfall_mm,
            "road_quality_modifier": self.road_quality_modifier,
        }


# ── Category distribution & weights ─────────────────────────────────────────

CATEGORY_DIST = {
    "routine":        0.60,
    "diabetic":       0.15,
    "high_risk_preg": 0.10,
    "tb_patient":     0.10,
    "newborn":        0.05,
}

# Weight used in disease burden index calculation
CATEGORY_DBI_WEIGHT = {
    "newborn":        3.0,
    "high_risk_preg": 2.5,
    "tb_patient":     2.0,
    "diabetic":       1.5,
    "routine":        1.0,
}

# How often illness events occur per household per day, by season
SEASONAL_ALERT_RATE = {
    "summer":  0.012,   # baseline
    "monsoon": 0.042,   # 3.5× — malaria, waterborne surge
    "winter":  0.022,   # 1.8× — respiratory surge
}

ILLNESS_NAMES = {
    "summer":  "heat-related illness / dehydration",
    "monsoon": "suspected malaria / waterborne fever",
    "winter":  "acute respiratory infection",
}


# ── Village class ─────────────────────────────────────────────────────────────

class Village:
    def __init__(self, n_households: int, seed: int, season: str = "summer",
                 start_day_of_year: int = 180):
        self.n_households = n_households
        self.seed = seed
        self.season = season
        self.start_day_of_year = start_day_of_year
        self.rng = random.Random(seed)
        self._death_count = 0

        # ASHA home and PHC positions
        self.asha_home_x, self.asha_home_y = ASHA_HOME
        self.phc_x,       self.phc_y       = PHC_LOCATION

        # Generate village map with real coordinates
        self.households: Dict[int, Household] = {}
        self._generate_households()

        # Generate opening weather
        self.weather = self._sample_weather()

    # ── Village generation ───────────────────────────────────────────────────

    def _generate_households(self):
        categories = list(CATEGORY_DIST.keys())
        weights    = [CATEGORY_DIST[c] for c in categories]

        for i in range(self.n_households):
            # Assign cluster, then scatter around its centre
            cluster = self.rng.randint(0, 4)
            cx, cy  = CLUSTER_CENTRES[cluster]

            # Gaussian scatter around cluster centre, std = 1.5 km
            x = max(0.3, min(9.7, cx + self.rng.gauss(0, 1.5)))
            y = max(0.3, min(9.7, cy + self.rng.gauss(0, 1.5)))

            road_q   = _compute_road_quality(x, y, self.rng)
            category = self.rng.choices(categories, weights=weights)[0]

            self.households[i] = Household(
                id=i,
                category=category,
                geo_cluster=cluster,
                x=round(x, 3),
                y=round(y, 3),
                road_quality=round(road_q, 3),
                rng=self.rng,
            )

    # ── Weather engine ───────────────────────────────────────────────────────

    def _sample_weather(self) -> WeatherState:
        dist       = SEASONAL_WEATHER_DIST[self.season]
        conditions = list(dist.keys())
        weights    = list(dist.values())
        condition  = self.rng.choices(conditions, weights=weights)[0]

        lo, hi = TEMP_RANGES[self.season]
        temp    = round(self.rng.uniform(lo, hi), 1)

        rlo, rhi = RAINFALL_RANGES[condition]
        rainfall  = round(self.rng.uniform(rlo, rhi), 1)

        return WeatherState(condition=condition, temp_celsius=temp, rainfall_mm=rainfall)

    def update_season_and_weather(self, day: int):
        """
        Call at end of each step.  Updates season based on day-of-year
        (offset by start_day_of_year so task3 starts in monsoon)
        and draws fresh weather for tomorrow.
        """
        cycle = (day + self.start_day_of_year) % 365
        if   60 <= cycle <= 180:  self.season = "monsoon"
        elif cycle > 300 or cycle < 30: self.season = "winter"
        else:                     self.season = "summer"

        self.weather = self._sample_weather()

    # ── Effective road quality (weather applied) ─────────────────────────────

    def effective_road_quality(self, household_id: int) -> float:
        """
        Road quality AFTER today's weather is applied.
        This is what ASHA experiences when walking to the household.
        """
        base = self.households[household_id].road_quality
        return round(max(0.04, base * self.weather.road_quality_modifier), 3)

    # ── Route travel time ────────────────────────────────────────────────────

    def compute_route_time_minutes(
        self,
        visit_sequence: List[int],
        start_x: float = None,
        start_y: float = None,
    ) -> Tuple[float, List[float]]:
        """
        Compute total travel time for a planned route.
        Returns (total_travel_min, per_hop_times).
        Useful for the agent to evaluate candidate routes before committing.
        """
        cx = start_x if start_x is not None else self.asha_home_x
        cy = start_y if start_y is not None else self.asha_home_y

        total = 0.0
        hops  = []

        for hh_id in visit_sequence:
            if hh_id not in self.households:
                continue
            hh      = self.households[hh_id]
            road_q  = self.effective_road_quality(hh_id)
            t       = travel_time_minutes(cx, cy, hh.x, hh.y, road_q, self.weather.condition)
            hops.append(t)
            total  += t
            cx, cy  = hh.x, hh.y

        return round(total, 1), hops

    # ── Daily tick ───────────────────────────────────────────────────────────

    def tick_all_unvisited(self, visited_ids: set) -> int:
        """
        Advance all unvisited households by one day.
        Returns the number of preventable deaths this step.
        """
        deaths = 0
        for hh in self.households.values():
            if hh.id not in visited_ids:
                died = hh.tick(season=self.season)
                if died:
                    deaths += 1
                    self._death_count += 1
        return deaths

    # ── Dynamic events ───────────────────────────────────────────────────────

    def generate_daily_events(self, day: int) -> list:
        """
        Generate illness outbreaks and new births.
        Rate is season-dependent: monsoon is 3.5× more active than summer.
        """
        alerts      = []
        alert_rate  = SEASONAL_ALERT_RATE[self.season]
        illness_name = ILLNESS_NAMES[self.season]

        for hh in self.households.values():
            if hh.is_dead:
                continue

            # Sudden illness event
            if self.rng.random() < alert_rate and not hh.danger_sign_active:
                spike = 0.30 if self.season == "summer" else 0.45
                hh.risk_score = min(1.0, hh.risk_score + spike)
                if hh.risk_score > 0.75:
                    hh.danger_sign_active = True
                    if self.rng.random() < 0.55:
                        dist = round(hh.dist_from(self.asha_home_x, self.asha_home_y), 1)
                        alerts.append({
                            "household_id": hh.id,
                            "message": (
                                f"HH {hh.id}: {illness_name} — "
                                f"cluster {hh.geo_cluster}, {dist} km away"
                            ),
                            "urgency": "high",
                        })

            # New birth
            if hh.category == "routine" and self.rng.random() < 0.004:
                hh.category         = "newborn"
                hh.risk_score       = 0.40
                hh.days_since_visit = 0
                hh.days_since_birth = 0
                hh.danger_sign_active = False
                dist = round(hh.dist_from(self.asha_home_x, self.asha_home_y), 1)
                road_q = self.effective_road_quality(hh.id)
                est_travel = travel_time_minutes(
                    self.asha_home_x, self.asha_home_y,
                    hh.x, hh.y, road_q, self.weather.condition,
                )
                alerts.append({
                    "household_id": hh.id,
                    "message": (
                        f"HH {hh.id}: NEW BIRTH — 48-hr visit required. "
                        f"{dist} km away, est. {round(est_travel)} min travel "
                        f"(road quality {road_q:.2f}, {self.weather.condition})"
                    ),
                    "urgency": "critical",
                })

        return alerts

    # ── Metrics ──────────────────────────────────────────────────────────────

    @property
    def preventable_deaths(self) -> int:
        return self._death_count

    def compute_dbi(self) -> float:
        """
        Disease Burden Index — category-weighted average risk.
        Normalised to 0–1.  Higher = worse village health.
        """
        alive = [hh for hh in self.households.values() if not hh.is_dead]
        if not alive:
            return 1.0
        numerator   = sum(hh.risk_score * CATEGORY_DBI_WEIGHT[hh.category] for hh in alive)
        denominator = sum(CATEGORY_DBI_WEIGHT[hh.category] for hh in alive)
        return round(numerator / denominator, 4)

    def get_tb_compliance(self) -> float:
        tb = [hh for hh in self.households.values() if hh.category == "tb_patient"]
        if not tb:
            return 1.0
        given  = sum(hh.tb_doses_given  for hh in tb)
        missed = sum(hh.tb_doses_missed for hh in tb)
        total  = given + missed
        return round(given / total, 4) if total > 0 else 1.0

    def count_missed_critical(self) -> int:
        return sum(1 for hh in self.households.values() if hh.danger_sign_active)
