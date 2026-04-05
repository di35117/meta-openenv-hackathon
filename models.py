
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class HouseholdState(BaseModel):
    # Identity
    id: int
    category: str          # "newborn" | "tb_patient" | "high_risk_preg" | "diabetic" | "routine"

    # Health
    risk_score: float = Field(ge=0.0, le=1.0)
    days_since_visit: int
    danger_sign_active: bool = False

    # Spatial — NEW
    geo_cluster: int       # 0–4 (North/South/East/West/Centre)
    x: float               # position in km from west edge (0–10)
    y: float               # position in km from south edge (0–10)
    road_quality: float    # 0.0 = mud path, 1.0 = paved road
                           # Already adjusted for today's weather
    dist_from_asha_home_km: float  # straight-line distance from ASHA's home

    # Time planning — NEW
    est_visit_duration_min: int    # how long a visit here will probably take
                                   # (before entering; agent uses this for planning)


# ─────────────────────────────────────────────────────────────────────────────
# Alerts
# ─────────────────────────────────────────────────────────────────────────────

class Alert(BaseModel):
    household_id: int
    message: str
    urgency: str    # "critical" | "high" | "routine"


# ─────────────────────────────────────────────────────────────────────────────
# Action (what the agent sends each day)
# ─────────────────────────────────────────────────────────────────────────────

class Action(BaseModel):
    visit_sequence: List[int]   # ordered household IDs; first = first visited


# ─────────────────────────────────────────────────────────────────────────────
# Weather — NEW
# ─────────────────────────────────────────────────────────────────────────────

class WeatherInfo(BaseModel):
    condition: str           # "sunny" | "cloudy" | "rainy" | "heavy_rain"
    temp_celsius: float
    rainfall_mm: float
    road_quality_modifier: float  # already baked into HouseholdState.road_quality
                                  # shown here so agent can understand WHY roads are slow


# ─────────────────────────────────────────────────────────────────────────────
# Observation (what the agent receives every step)
# ─────────────────────────────────────────────────────────────────────────────

class Observation(BaseModel):
    # Episode position
    day: int
    season: str              # "summer" | "monsoon" | "winter"
    weather: WeatherInfo     # TODAY's weather

    # Time tracking — NEW
    current_time_min: int    # minutes elapsed since 7:00 AM
                             # 0 = just started, 360 = day is over (1:00 PM)
    asha_energy_pct: float   # 100 = fresh start, 0 = exhausted
                             # below 30 → visits take longer and reset risk less

    # ASHA spatial state — NEW
    asha_home_x: float       # always the same; agent uses this as routing start point
    asha_home_y: float

    # Village
    households: List[HouseholdState]
    new_alerts: List[Alert]

    # Required by OpenEnv framework
    reward: float = 0.0
    done: bool = False
    metadata: Dict[str, Any] = {}


# ─────────────────────────────────────────────────────────────────────────────
# EnvironmentState (full internal state; used by grader, not shown to agent)
# ─────────────────────────────────────────────────────────────────────────────

class EnvironmentState(BaseModel):
    task_id: str
    current_day: int
    episode_done: bool
    all_households: List[dict]
    cumulative_reward: float
    disease_burden_index: float
    preventable_deaths: int
    tb_compliance_rate: float
    visit_history: List[dict]
