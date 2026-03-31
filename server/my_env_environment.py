from uuid import uuid4
from typing import List
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import Action, Observation, HouseholdState, Alert
except ImportError:
    from models import Action, Observation, HouseholdState, Alert

from .village import Village
from .tasks import TASKS

class MyEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.task = TASKS["task1"]
        self.village = Village(n_households=self.task.n_households, seed=self.task.seed)
        self.day = 0
        self.history = []
        self._current_alerts = []

    def reset(self, *args, **kwargs) -> Observation:
        # Check if validator passed a task_id
        task_id = kwargs.get("task_id", "task1")
        
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.task = TASKS.get(task_id, TASKS["task1"])
        self.village = Village(n_households=self.task.n_households, seed=self.task.seed)
        self.day = 0
        self.history = []
        self._current_alerts = []
        
        return self._make_observation(reward=0.0, done=False, info={})

    def step(self, action: Action) -> Observation:
        self._state.step_count += 1
        danger_ids_at_start = [hh.id for hh in self.village.households.values() if hh.danger_sign_active]
        travel_time = self._compute_travel(action.visit_sequence)
        visit_results = []
        hours_used = 0

        for hh_id in action.visit_sequence:
            visit_time = 0.5   
            if hours_used + visit_time > 6.0:
                break          
            
            if hh_id in self.village.households:
                result = self.village.households[hh_id].receive_visit()
                result["category"] = self.village.households[hh_id].category
                visit_results.append(result)
            hours_used += visit_time

        visited_ids = set(action.visit_sequence[:len(visit_results)])
        
        for hh in self.village.households.values():
            if hh.id not in visited_ids:
                hh.tick()

        self._current_alerts = self.village.generate_daily_events(self.day)
        
        # Calculate Reward
        r = 0.0
        for res in visit_results:
            if res["danger_sign"]: r += 0.4
            if res["referral_needed"]: r += 0.3
            if res["category"] == "newborn": r += 0.25   
            if res["category"] == "tb_patient": r += 0.15    
        r -= 0.1 * max(0, travel_time - 6.0) 
        r -= 1.0 * self.village.preventable_deaths
        final_reward = max(-1.0, min(1.0, r))
        
        self.history.append({
            "day": self.day,
            "visited": list(visited_ids),
            "reward": final_reward,
            "danger_ids_at_start": danger_ids_at_start 
        })

        self.day += 1
        done = self.day >= self.task.max_days

        return self._make_observation(
            reward=final_reward, 
            done=done, 
            info={"travel_time": travel_time, "preventable_deaths": self.village.preventable_deaths}
        )

    def _compute_travel(self, sequence: List[int]) -> float:
        return len(sequence) * 0.25 

    def _make_observation(self, reward: float, done: bool, info: dict) -> Observation:
        hhs = [
            HouseholdState(
                id=hh.id, category=hh.category, risk_score=hh.risk_score,
                days_since_visit=hh.days_since_visit, danger_sign_active=hh.danger_sign_active,
                geo_cluster=hh.geo_cluster
            )
            for hh in self.village.households.values()
        ]
        alerts = [Alert(**a) for a in self._current_alerts]
        
        return Observation(
            day=self.day, households=hhs, asha_hours_remaining=6.0,
            new_alerts=alerts, reward=reward, done=done, metadata=info
        )

    @property
    def state(self) -> State:
        return self._state