"""
my_env_environment.py — The RL environment.  Implements the OpenEnv interface.

REWARD FUNCTION IMPROVEMENTS (this revision):
  1. Missed danger sign penalty: -0.15 per known danger sign left unvisited.
     Previously the agent only paid -1.0 when a death occurred (days later).
     This gives immediate signal to not skip critical cases.

  2. Near-death proactive penalty: -0.20 per household with risk >= 0.90
     that was NOT visited.  This fires the day BEFORE death, giving the agent
     a chance to course-correct without waiting for the terminal -1.0.

  3. Scaled newborn reward: day-0 birth = +0.40, day-1 = +0.30, day-2 = +0.20.
     Previously flat +0.25 regardless of urgency.  The 48-hour window is real:
     neonatal sepsis mortality doubles every 12 hours of delay.

  4. TB overdue penalty: if a TB patient is visited but already past their
     3-day window, apply a small penalty (-0.03 per day late, capped at -0.10).
     Previously only rewarded on-time doses, never penalized late ones.

  5. Scaled geographic equity bonus: +0.01 per cluster covered beyond 1,
     up to +0.05 (previously flat 0 or +0.03 at exactly 3 clusters).

  6. Near-critical TB penalty: if a TB patient has been missed for 5+ days
     (drug-resistance window), apply -0.08 per such patient not visited.

UNCHANGED FROM PREVIOUS VERSION:
  - Time tracking, ASHA position, energy model
  - tanh normalisation (no hard clip)
  - Throughput bonus, routing efficiency bonus
  - Death penalty (-1.0 per death, floored at -1.0 with tanh)
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
ENERGY_START               = 100.0
ENERGY_COST_TRAVEL_PER_MIN = 0.15
ENERGY_COST_VISIT_PER_MIN  = 0.10


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
        self._seed_task1_dangers()
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
        task_id      = kwargs.get("task_id", "task1")
        self._state  = State(episode_id=str(uuid4()), step_count=0)
        self.task    = TASKS.get(task_id, TASKS["task1"])
        self.village = Village(
            n_households=self.task.n_households,
            seed=self.task.seed,
            season=self.task.season,
            start_day_of_year=self.task.start_day_of_year,
        )
        self._seed_task1_dangers()
        self.day     = 0
        self.history = []
        self._current_alerts = []
        self._reset_daily_asha()
        return self._make_observation(reward=0.0, done=False, info={})

    def _seed_task1_dangers(self):
        """
        For task1: guarantee exactly 8 danger-sign households.
        Uses a secondary RNG seeded from the task seed so it is always
        deterministic but doesn't disturb the village-generation sequence.
        """
        if self.task.id != "task1":
            return
        import random
        seed_rng = random.Random(self.task.seed + 999)
        candidates = [
            hh for hh in self.village.households.values()
            if not hh.is_dead
        ]
        for hh in seed_rng.sample(candidates, min(8, len(candidates))):
            hh.risk_score = seed_rng.uniform(0.76, 0.88)
            hh.danger_sign_active = True

    def step(self, action: Action) -> Observation:
        """
        Execute one day of ASHA work.

        Route simulation:
          for each household in visit_sequence:
            1. compute travel time from current position to household
            2. check if travel + visit fits in the day
            3. travel (time + energy consumed)
            4. visit (time + energy consumed; effectiveness depends on energy)
            5. update position
          tick() all unvisited households
          generate daily events
          compute reward
          update season + weather
          reset ASHA for tomorrow
        """
        self._state.step_count += 1

        # Snapshot danger signs BEFORE any visit (grader needs pre-visit truth)
        danger_ids_before: set = {
            hh.id for hh in self.village.households.values()
            if hh.danger_sign_active
        }

        # Snapshot near-death households BEFORE visiting
        # (risk >= 0.90 but not yet dead — proactive signal)
        near_death_before: set = {
            hh.id for hh in self.village.households.values()
            if hh.risk_score >= 0.90 and not hh.is_dead
        }

        # ── Execute route ─────────────────────────────────────────────────
        visit_results: list = []
        cx, cy = self.asha_x, self.asha_y
        elapsed_min  = self.current_time_min
        energy       = self.asha_energy

        for hh_id in action.visit_sequence[:MAX_VISITS]:
            if hh_id not in self.village.households:
                continue
            hh = self.village.households[hh_id]
            if hh.is_dead:
                continue

            road_q     = self.village.effective_road_quality(hh_id)
            travel_min = travel_time_minutes(
                cx, cy, hh.x, hh.y, road_q, self.village.weather.condition
            )

            est_visit  = hh.est_visit_duration_min()
            if elapsed_min + travel_min + est_visit > DAY_END_MIN:
                break

            elapsed_min += travel_min
            energy       = max(0.0, energy - travel_min * ENERGY_COST_TRAVEL_PER_MIN)

            result = hh.receive_visit(energy_pct=energy)
            actual_visit_min = result["visit_duration_min"]
            elapsed_min += actual_visit_min
            energy       = max(0.0, energy - actual_visit_min * ENERGY_COST_VISIT_PER_MIN)

            cx, cy = hh.x, hh.y

            result["household_id"]    = hh_id
            result["travel_min"]      = round(travel_min, 1)
            result["time_of_visit"]   = round(elapsed_min, 1)
            result["energy_at_visit"] = round(energy, 1)
            visit_results.append(result)

        # Save ASHA end-of-day state
        self.asha_x           = cx
        self.asha_y           = cy
        self.asha_energy      = round(energy, 1)
        self.current_time_min = round(elapsed_min, 1)

        visited_ids      = {r["household_id"] for r in visit_results}
        total_travel_min = round(sum(r["travel_min"] for r in visit_results), 1)

        # ── Tick unvisited households ─────────────────────────────────────
        new_deaths = self.village.tick_all_unvisited(visited_ids)

        # ── Generate dynamic events ───────────────────────────────────────
        self._current_alerts = self.village.generate_daily_events(self.day)

        # ── Update season + weather for next day ──────────────────────────
        self.village.update_season_and_weather(self.day)

        # ── Compute reward ────────────────────────────────────────────────
        reward = self._compute_reward(
            visit_results=visit_results,
            new_deaths=new_deaths,
            total_travel_min=total_travel_min,
            elapsed_min=elapsed_min,
            danger_ids_before=danger_ids_before,
            near_death_before=near_death_before,
            visited_ids=visited_ids,
        )

        # ── Log this day ──────────────────────────────────────────────────
        self.history.append({
            "day":               self.day,
            "visited":           list(visited_ids),
            "danger_ids_before": list(danger_ids_before),
            "reward":            reward,
            "elapsed_min":       self.current_time_min,
            "total_travel_min":  total_travel_min,
            "asha_energy_end":   self.asha_energy,
            "new_deaths":        new_deaths,
            "season":            self.village.season,
            "weather":           self.village.weather.condition,
            "visits_completed":  len(visit_results),
        })

        self.day += 1
        done = self.day >= self.task.max_days

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
        danger_ids_before: set,
        near_death_before: set,
        visited_ids: set,
    ) -> float:
        """
        Shaped per-step reward grounded in real ASHA clinical priorities.

        Positive signals (per visit):
          +0.40        danger sign caught
          +0.20        referral needed (non-danger high-risk case)
          +0.40/0.30/0.20  newborn visited at day 0/1/2 of life
          +0.15        TB dose supervised on time
          +0.05        high-risk pregnancy or diabetic visited
          +0.00–0.05   throughput bonus (visits per hour)
          +0.00–0.04   routing efficiency bonus (low travel fraction)
          +0.01–0.05   geographic equity (clusters covered beyond first)

        Negative signals:
          -0.05 × scaled  wasted visit (stable routine household)
          -0.15           per known danger sign left unvisited (immediate)
          -0.20           per near-death household (risk ≥ 0.90) left unvisited
          -0.03/day late  TB patient visited but already overdue (max -0.10)
          -0.08           per TB patient with 5+ days missed (drug-resistance)
          -1.00 × deaths  preventable death (dominates all clinical gains)

        All pre-penalty values are summed then passed through tanh(r/4.0)
        so the agent can distinguish a good day from a great day without
        extreme reward magnitudes.
        """
        r = 0.0

        for res in visit_results:
            cat = res["category"]

            # ── Danger sign caught ────────────────────────────────────────
            if res["danger_sign"]:
                r += 0.40

            # ── Referral (non-danger high-risk) ───────────────────────────
            # Fires only when danger_sign is NOT active to avoid double-count.
            if res["referral_needed"] and not res["danger_sign"]:
                r += 0.20

            # ── Newborn: scaled by urgency ────────────────────────────────
            # Day-0 birth has same severity as a caught danger sign.
            # Neonatal sepsis mortality approximately doubles every 12h of delay.
            if res.get("newborn_48hr"):
                dsb = res.get("days_since_birth") or 0
                if dsb == 0:
                    r += 0.40   # same-day birth — maximum urgency
                elif dsb == 1:
                    r += 0.30
                else:
                    r += 0.20   # day 2 — still within window but less acute

            # ── TB dose supervised on time ────────────────────────────────
            if res.get("tb_dose_on_time"):
                r += 0.15
            elif cat == "tb_patient":
                # TB patient visited but already past the 3-day window.
                # Late is still better than never, but apply a small penalty.
                days_late = max(0, res.get("days_since_visit", 0) - 3)
                r -= min(0.10, days_late * 0.03)

            # ── High-priority category (stable but needs monitoring) ───────
            if cat in ("high_risk_preg", "diabetic"):
                r += 0.05

            # ── Wasted visit: stable routine household ────────────────────
            # Penalty scales with how low the risk is — the lower the risk,
            # the more wasteful visiting is relative to skipping a critical HH.
            if cat == "routine" and not res["danger_sign"] and res["risk_before"] < 0.10:
                waste_penalty = 0.05 * (1.0 - res["risk_before"] / 0.10)
                r -= waste_penalty

        # ── Missed danger sign penalty ────────────────────────────────────
        # Fires immediately when a known danger sign is not visited today.
        # This is the most important new signal: the agent cannot rationally
        # skip a danger sign and wait for the deferred -1.0 death penalty.
        missed_dangers = danger_ids_before - visited_ids
        r -= 0.15 * len(missed_dangers)

        # ── Near-death proactive penalty ──────────────────────────────────
        # Households at risk >= 0.90 are one bad tick away from dying.
        # Penalize leaving them unvisited to teach the agent to act before
        # the terminal -1.0 death penalty fires.
        near_death_missed = near_death_before - visited_ids
        r -= 0.20 * len(near_death_missed)

        # ── TB drug-resistance penalty ────────────────────────────────────
        # TB patients unvisited for 5+ days begin developing drug resistance.
        # This fires even if they don't die, because resistance is irreversible.
        for hh in self.village.households.values():
            if (hh.category == "tb_patient"
                    and hh.id not in visited_ids
                    and hh.days_since_visit >= 5
                    and not hh.is_dead):
                r -= 0.08

        # ── Throughput bonus ──────────────────────────────────────────────
        # Rewards doing more effective visits per hour of working time.
        # Max practical throughput ≈ 10 visits / 360 min = 1.67 visits/hr.
        # Capped at +0.05 to stay smaller than clinical rewards.
        if elapsed_min > 0:
            visits_per_hour  = (len(visit_results) / elapsed_min) * 60.0
            throughput_bonus = min(0.05, (visits_per_hour / 1.67) * 0.05)
            r += throughput_bonus

        # ── Routing efficiency bonus (continuous) ─────────────────────────
        # Scales from 0 (all time spent walking) to +0.04 (minimal travel).
        # Rewards cluster batching without binary thresholds.
        if len(visit_results) > 0 and elapsed_min > 0:
            travel_fraction = total_travel_min / elapsed_min
            routing_bonus   = max(0.0, (0.50 - travel_fraction) / 0.50) * 0.04
            r += routing_bonus

        # ── Geographic equity bonus (scaled) ─────────────────────────────
        # +0.01 per cluster covered beyond the first, up to +0.05.
        # Prevents the agent from permanently camping in the nearest cluster.
        visited_clusters: set = set()
        for res in visit_results:
            hh_id = res.get("household_id")
            if hh_id is not None and hh_id in self.village.households:
                visited_clusters.add(self.village.households[hh_id].geo_cluster)
        n_clusters = len(visited_clusters)
        if n_clusters >= 2:
            equity_bonus = min(0.05, (n_clusters - 1) * 0.01)
            r += equity_bonus

        # ── Death penalty + tanh normalisation ───────────────────────────
        # -1.0 per death overwhelms any single-day clinical gain.
        # tanh(r/4.0) preserves the ordering of good vs great days while
        # bounding the output to (-1, 1).
        r -= 1.0 * new_deaths
        if new_deaths == 0:
            normalized = math.tanh(r / 4.0)
        else:
            normalized = max(-1.0, math.tanh(r / 4.0))

        return round(normalized, 4)

    # ── Full grading state ────────────────────────────────────────────────────

    def get_full_state(self) -> dict:
        return {
            "task_id":              self.task.id,
            "current_day":          self.day,
            "episode_done":         self.day >= self.task.max_days,
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
            "cumulative_reward":    round(sum(d["reward"] for d in self.history), 4),
            "disease_burden_index": self.village.compute_dbi(),
            "preventable_deaths":   self.village.preventable_deaths,
            "tb_compliance_rate":   self.village.get_tb_compliance(),
            "visit_history":        self.history,
            "season":               self.village.season,
            "weather":              self.village.weather.condition,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _reset_daily_asha(self):
        self.asha_x           = self.village.asha_home_x
        self.asha_y           = self.village.asha_home_y
        self.asha_energy      = ENERGY_START
        self.current_time_min = DAY_START_MIN

    def _make_observation(self, reward: float, done: bool, info: dict) -> Observation:
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
