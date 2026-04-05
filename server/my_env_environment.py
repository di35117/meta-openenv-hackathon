"""
my_env_environment.py — The RL environment.  Implements the OpenEnv interface.

What changed from the static version:

TIME TRACKING
  A 6-hour clock runs each day (7:00 AM → 1:00 PM = 0 → 360 minutes).
  Every travel hop costs real minutes (distance / weather-adjusted speed).
  Every visit costs real minutes (base + danger-sign extra + referral prep).
  The loop stops when the clock reaches 360, not when a visit count is hit.
  This means: far households cost the ASHA more time than close ones,
  rainy days shrink the effective visit window, and batching nearby households
  frees up time for more visits.

ASHA POSITION TRACKING
  The ASHA starts at home (5, 5) each morning.
  Her position updates after each travel hop.
  Travel time to the next household is computed FROM her current position,
  not from home — so route order genuinely matters.

ASHA ENERGY / FATIGUE
  Energy starts at 100 each morning.
  Walking costs 0.15 energy per minute.
  Visits cost 0.10 energy per minute.
  Below 60% energy → visits take longer and reset risk less effectively.
  Below 30% energy → significant quality degradation.
  Energy resets to 100 at the start of each new day.

SEASONAL / WEATHER DYNAMICS
  season and weather update at the end of every step.
  Deterioration rates on tick() include the seasonal multiplier.
  Alert rates in generate_daily_events() scale with season.
  Road quality presented to the agent already has weather modifier applied.
"""

import math
from uuid import uuid4
from typing import List

# openenv-core is optional — works with or without it installed
try:
    from openenv.core.env_server.interfaces import Environment as _Base
    from openenv.core.env_server.types import State
except ImportError:
    _Base = object
    class State:  # type: ignore
        def __init__(self, episode_id: str, step_count: int = 0):
            self.episode_id  = episode_id
            self.step_count  = step_count

try:
    from ..models import Action, Observation, HouseholdState, Alert, WeatherInfo
except ImportError:
    from models import Action, Observation, HouseholdState, Alert, WeatherInfo

from .village import Village, travel_time_minutes
from .tasks import TASKS

# ── Time constants ────────────────────────────────────────────────────────────
DAY_START_MIN  = 0     # 7:00 AM
DAY_END_MIN    = 360   # 1:00 PM  (6 working hours)
MAX_VISITS     = 15    # absolute cap even if time allows

# ── ASHA energy model ─────────────────────────────────────────────────────────
ENERGY_START             = 100.0
ENERGY_COST_TRAVEL_PER_MIN = 0.15   # walking is physically taxing
ENERGY_COST_VISIT_PER_MIN  = 0.10   # mental + clinical effort per minute


class MyEnvironment(_Base):
    """
    Full dynamic ASHA scheduling environment.

    Concurrency note: SUPPORTS_CONCURRENT_SESSIONS = True means the OpenEnv
    framework creates one MyEnvironment instance per WebSocket session,
    so parallel agents each have their own isolated village.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state    = State(episode_id=str(uuid4()), step_count=0)
        self.task      = TASKS["task1"]
        self.village   = Village(
            n_households=self.task.n_households,
            seed=self.task.seed,
            season=self.task.season,
            start_day_of_year=self.task.start_day_of_year,
        )
        self.day       = 0
        self.history: list = []
        self._current_alerts: list = []
        self._reset_daily_asha()

    # ── OpenEnv API ───────────────────────────────────────────────────────────

    def reset(self, *args, **kwargs) -> Observation:
        """
        Start a fresh episode.  Accepts task_id as a keyword argument.
        Creates a brand-new Village so no state leaks between episodes.
        """
        task_id = kwargs.get("task_id", "task1")
        self._state  = State(episode_id=str(uuid4()), step_count=0)
        self.task    = TASKS.get(task_id, TASKS["task1"])
        self.village = Village(
            n_households=self.task.n_households,
            seed=self.task.seed,
            season=self.task.season,
            start_day_of_year=self.task.start_day_of_year,
        )
        self.day     = 0
        self.history = []
        self._current_alerts = []
        self._reset_daily_asha()
        return self._make_observation(reward=0.0, done=False, info={})

    def step(self, action: Action) -> Observation:
        """
        Execute one day of ASHA work.

        The route is simulated minute-by-minute:
          for each household in visit_sequence:
            1. compute travel time from current position to household
            2. check if time + visit will fit within the day
            3. move ASHA to household (time + energy consumed)
            4. execute visit (time + energy consumed; result depends on energy)
            5. update ASHA position
          tick() all unvisited households
          generate_daily_events()
          compute reward
          update season + weather for next day
          reset ASHA for tomorrow
        """
        self._state.step_count += 1

        # Snapshot which households had danger signs BEFORE any visit
        # (visiting clears the flag — grader needs the pre-visit truth)
        danger_ids_before = {
            hh.id for hh in self.village.households.values()
            if hh.danger_sign_active
        }

        # ── Execute route ─────────────────────────────────────────────────
        visit_results: list = []
        cx, cy = self.asha_x, self.asha_y    # ASHA current position
        elapsed_min  = self.current_time_min
        energy       = self.asha_energy

        for hh_id in action.visit_sequence[:MAX_VISITS]:
            if hh_id not in self.village.households:
                continue
            hh = self.village.households[hh_id]
            if hh.is_dead:
                continue

            # ① Real travel time from CURRENT POSITION to this household
            road_q     = self.village.effective_road_quality(hh_id)
            travel_min = travel_time_minutes(
                cx, cy, hh.x, hh.y, road_q, self.village.weather.condition
            )

            # ② Check if travel + estimated visit fits in the day
            est_visit  = hh.est_visit_duration_min()
            if elapsed_min + travel_min + est_visit > DAY_END_MIN:
                # No time — stop routing for today
                break

            # ③ ASHA travels to household (uses time + energy)
            elapsed_min += travel_min
            energy       = max(0.0, energy - travel_min * ENERGY_COST_TRAVEL_PER_MIN)

            # ④ Execute visit (actual duration may differ from estimate)
            result = hh.receive_visit(energy_pct=energy)
            actual_visit_min = result["visit_duration_min"]
            elapsed_min += actual_visit_min
            energy       = max(0.0, energy - actual_visit_min * ENERGY_COST_VISIT_PER_MIN)

            # ⑤ Update ASHA position to this household
            cx, cy = hh.x, hh.y

            # Annotate result with routing data (useful for debugging + history)
            result["household_id"]  = hh_id
            result["travel_min"]    = round(travel_min, 1)
            result["time_of_visit"] = round(elapsed_min, 1)
            result["energy_at_visit"] = round(energy, 1)
            visit_results.append(result)

        # Save ASHA's end-of-day state
        self.asha_x          = cx
        self.asha_y          = cy
        self.asha_energy     = round(energy, 1)
        self.current_time_min = round(elapsed_min, 1)

        visited_ids = {r["household_id"] for r in visit_results}
        total_travel_min = round(sum(r["travel_min"] for r in visit_results), 1)

        # ── Tick unvisited households ─────────────────────────────────────
        new_deaths = self.village.tick_all_unvisited(visited_ids)

        # ── Generate dynamic events ───────────────────────────────────────
        self._current_alerts = self.village.generate_daily_events(self.day)

        # ── Update season + weather for next day ──────────────────────────
        self.village.update_season_and_weather(self.day)

        # ── Compute reward ────────────────────────────────────────────────
        reward = self._compute_reward(
            visit_results, new_deaths, total_travel_min, elapsed_min
        )

        # ── Log this day ──────────────────────────────────────────────────
        self.history.append({
            "day":                 self.day,
            "visited":             list(visited_ids),
            "danger_ids_before":   list(danger_ids_before),
            "reward":              reward,
            "elapsed_min":         self.current_time_min,
            "total_travel_min":    total_travel_min,
            "asha_energy_end":     self.asha_energy,
            "new_deaths":          new_deaths,
            "season":              self.village.season,
            "weather":             self.village.weather.condition,
            "visits_completed":    len(visit_results),
        })

        self.day += 1
        done = self.day >= self.task.max_days

        # Reset ASHA for next morning
        self._reset_daily_asha()

        return self._make_observation(
            reward=reward,
            done=done,
            info={
                "visits_completed":  len(visit_results),
                "elapsed_min":       self.current_time_min,
                "total_travel_min":  total_travel_min,
                "new_deaths":        new_deaths,
                "asha_energy_end":   self.asha_energy,
                "season":            self.village.season,
                "weather":           self.village.weather.condition,
            },
        )

    @property
    def state(self) -> State:
        return self._state

    # ── Reward function ───────────────────────────────────────────────────────

    def _compute_reward(
        self,
        visit_results: list,
        new_deaths: int,
        total_travel_min: float,
        elapsed_min: float,
    ) -> float:
        r = 0.0

        for res in visit_results:
            cat = res["category"]

            # Core clinical rewards
            if res["danger_sign"]:        r += 0.40   # caught an emergency
            if res["referral_needed"]:    r += 0.30   # initiated referral
            if res.get("newborn_48hr"):   r += 0.25   # 48-hr newborn visit
            if res.get("tb_dose_on_time"):r += 0.15   # DOTS dose on schedule

            # Partial credit for important-but-not-emergency categories
            if cat in ("high_risk_preg", "diabetic"):
                r += 0.05

            # Penalty for wasting a slot on a very stable routine household
            if cat == "routine" and res["risk_before"] < 0.10 and not res["danger_sign"]:
                r -= 0.05

        # Time efficiency bonus:
        # Finishing with time to spare means the agent could have fit more visits.
        # This nudges the agent to batch nearby households efficiently.
        time_remaining = max(0.0, DAY_END_MIN - elapsed_min)
        time_efficiency_bonus = (time_remaining / DAY_END_MIN) * 0.04
        r += time_efficiency_bonus

        # Travel efficiency bonus:
        # If total travel was less than 30% of the working day, the agent
        # clustered households well.
        if len(visit_results) > 0 and total_travel_min < DAY_END_MIN * 0.30:
            r += 0.03

        # Death penalty — the harshest signal in the system
        r -= 1.0 * new_deaths

        return max(-1.0, min(1.0, round(r, 4)))

    # ── Full grading state ────────────────────────────────────────────────────

    def get_full_state(self) -> dict:
        """
        Returns the complete simulation state as a plain dict.
        Used by the /state and /grade HTTP endpoints.
        The grader functions receive a SimpleNamespace of this dict.
        """
        return {
            "task_id":               self.task.id,
            "current_day":           self.day,
            "episode_done":          self.day >= self.task.max_days,
            "all_households": [
                {
                    "id":                hh.id,
                    "category":          hh.category,
                    "geo_cluster":       hh.geo_cluster,
                    "risk_score":        round(hh.risk_score, 4),
                    "danger_sign_active":hh.danger_sign_active,
                    "is_dead":           hh.is_dead,
                }
                for hh in self.village.households.values()
            ],
            "cumulative_reward":     round(sum(d["reward"] for d in self.history), 4),
            "disease_burden_index":  self.village.compute_dbi(),
            "preventable_deaths":    self.village.preventable_deaths,
            "tb_compliance_rate":    self.village.get_tb_compliance(),
            "visit_history":         self.history,
            "season":                self.village.season,
            "weather":               self.village.weather.condition,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _reset_daily_asha(self):
        """Reset ASHA's position and energy to start-of-day values."""
        self.asha_x          = self.village.asha_home_x
        self.asha_y          = self.village.asha_home_y
        self.asha_energy     = ENERGY_START
        self.current_time_min = DAY_START_MIN

    def _make_observation(self, reward: float, done: bool, info: dict) -> Observation:
        """
        Convert internal simulation state into a typed Observation.

        For each household, the agent sees:
          - Health state (risk, category, danger, days since visit)
          - Real coordinates (x, y) for route planning
          - Road quality AFTER today's weather modifier is applied
          - Straight-line distance from ASHA's home
          - Estimated visit duration (before entering)
        """
        hh_states = []
        for hh in self.village.households.values():
            road_q = self.village.effective_road_quality(hh.id)
            dist   = round(
                math.sqrt(
                    (hh.x - self.village.asha_home_x) ** 2
                    + (hh.y - self.village.asha_home_y) ** 2
                ),
                2,
            )
            hh_states.append(
                HouseholdState(
                    id=hh.id,
                    category=hh.category,
                    risk_score=round(hh.risk_score, 3),
                    days_since_visit=hh.days_since_visit,
                    danger_sign_active=hh.danger_sign_active,
                    geo_cluster=hh.geo_cluster,
                    x=hh.x,
                    y=hh.y,
                    road_quality=road_q,
                    dist_from_asha_home_km=dist,
                    est_visit_duration_min=hh.est_visit_duration_min(),
                )
            )

        alerts = [Alert(**a) for a in self._current_alerts]
        w = self.village.weather

        return Observation(
            day=self.day,
            season=self.village.season,
            weather=WeatherInfo(
                condition=w.condition,
                temp_celsius=w.temp_celsius,
                rainfall_mm=w.rainfall_mm,
                road_quality_modifier=w.road_quality_modifier,
            ),
            current_time_min=self.current_time_min,
            asha_energy_pct=self.asha_energy,
            asha_home_x=self.village.asha_home_x,
            asha_home_y=self.village.asha_home_y,
            households=hh_states,
            new_alerts=alerts,
            reward=reward,
            done=done,
            metadata=info,
        )
