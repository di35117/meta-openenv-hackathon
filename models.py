from pydantic import BaseModel, Field
from typing import List, Dict, Any

class HouseholdState(BaseModel):
    id: int
    category: str
    risk_score: float = Field(ge=0.0, le=1.0)
    days_since_visit: int
    danger_sign_active: bool = False
    geo_cluster: int

class Alert(BaseModel):
    household_id: int
    message: str
    urgency: str

class Action(BaseModel):
    visit_sequence: List[int] # Ordered list of household IDs to visit

class Observation(BaseModel):
    day: int
    households: List[HouseholdState]
    asha_hours_remaining: float
    new_alerts: List[Alert]
    # Required by OpenEnv framework:
    reward: float = 0.0
    done: bool = False
    metadata: Dict[str, Any] = {}

class EnvironmentState(BaseModel):
    task_id: str
    current_day: int
    episode_done: bool
    all_households: List[dict]
    cumulative_reward: float
    disease_burden_index: float
    preventable_deaths: int
    visit_history: List[dict]