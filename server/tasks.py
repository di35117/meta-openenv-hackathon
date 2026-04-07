"""
tasks.py — Task configurations and deterministic graders.

What changed:
  • TaskConfig now carries season (affects starting disease rates, weather)
  • Graders use danger_ids_before from history (not post-visit state)
    to correctly credit the agent for catching danger signs
  • Task 3 grader adds a routing-efficiency component:
    agents that wasted time on long travel routes are penalised
  • All grader scores clamped to (0.001, 0.999) — validator requires
    strictly open interval (0, 1); exact 0.0 or 1.0 would fail the check
  • All external state values (tb_compliance_rate, disease_burden_index)
    are sanitized before use — None / NaN / out-of-range all handled
"""

from dataclasses import dataclass
from typing import Any
import math


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clamp(score: float) -> float:
    """Force score into open interval (0, 1) as required by validator."""
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return 0.001
    return round(max(0.001, min(0.999, float(score))), 4)


def _safe_float(value, default: float = 0.0) -> float:
    """Convert value to float safely; return default if None/NaN/invalid."""
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


@dataclass
class TaskConfig:
    id:                 str
    name:               str
    difficulty:         str
    description:        str
    n_households:       int
    max_days:           int
    seed:               int
    season:             str = "summer"
    start_day_of_year:  int = 180   # used for season cycling
    # day 0   → winter  (cycle <  30 or > 300)
    # day 90  → summer  (30 ≤ cycle ≤ 59  or 181 ≤ cycle ≤ 299)
    # day 120 → monsoon (60 ≤ cycle ≤ 180)


TASKS: dict[str, TaskConfig] = {
    "task1": TaskConfig(
        id="task1",
        name="Single-day triage",
        difficulty="easy",
        description=(
            "30 households, 1 summer day. 8 danger-sign cases planted. "
            "Full visibility, flat terrain.  "
            "Goal: visit all critical households within 6 hours."
        ),
        n_households=30,
        max_days=1,
        seed=42,
        season="summer",
        start_day_of_year=200,   # mid-summer
    ),
    "task2": TaskConfig(
        id="task2",
        name="Week-long scheduling",
        difficulty="medium",
        description=(
            "100 households, 7 days, mixed summer/early-monsoon weather. "
            "TB patients need 3-day DOTS cycles. High-risk pregnancies require "
            "weekly monitoring.  2 surprise alerts arrive mid-week.  "
            "Travel time matters — dirt-path households cost more time.  "
            "Goal: maximise weighted coverage and TB compliance."
        ),
        n_households=100,
        max_days=7,
        seed=101,
        season="summer",
        start_day_of_year=200,   # mid-summer
    ),
    "task3": TaskConfig(
        id="task3",
        name="Monthly village management",
        difficulty="hard",
        description=(
            "200 households, 30 days, monsoon season. "
            "Heavy rain degrades dirt-path roads to near-impassable. "
            "A TB cluster forms silently in the east cluster. "
            "3 high-risk pregnancies approaching term. "
            "Disease events 3.5× more frequent than summer. "
            "Goal: minimise disease burden index while maintaining "
            "equitable coverage across all geographic clusters."
        ),
        n_households=200,
        max_days=30,
        seed=2026,
        season="monsoon",
        start_day_of_year=120,   # deep monsoon (day 60–180)
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Graders
# ─────────────────────────────────────────────────────────────────────────────

def grade_task1(state: Any) -> float:
    """
    Score = fraction of pre-visit danger-sign households that were visited.

    Uses danger_ids_before from history (snapshotted before visits on that day)
    so that visiting a household and clearing its danger flag still counts.
    Score range: (0.001, 0.999) after clamp.  Deterministic.
    """
    visit_history = getattr(state, "visit_history", None) or []

    if not visit_history:
        return 0.001

    # Collect all danger-sign IDs that existed at the START of day 0
    danger_before: set[int] = set()
    for day_log in visit_history:
        danger_before.update(day_log.get("danger_ids_before", []))

    if not danger_before:
        # No danger signs in this episode — give full credit
        return 0.999

    visited_ids: set[int] = set()
    for day_log in visit_history:
        visited_ids.update(day_log.get("visited", []))

    caught = len(danger_before & visited_ids)
    return _clamp(caught / len(danger_before))


def grade_task2(state: Any) -> float:
    """
    Score = 0.5 × weighted_coverage  +  0.5 × tb_compliance_rate

    Weighted coverage: visits are weighted by household category importance.
    TB compliance: fraction of 3-day DOTS windows honoured.
    Score range: (0.001, 0.999) after clamp.  Deterministic.
    """
    weights = {
        "newborn":        3.0,
        "tb_patient":     2.0,
        "high_risk_preg": 2.5,
        "diabetic":       1.5,
        "routine":        1.0,
    }

    all_hh = getattr(state, "all_households", None) or []
    visit_history = getattr(state, "visit_history", None) or []

    # Build set of all household IDs visited across the episode
    visited_ids: set[int] = set()
    for day_log in visit_history:
        visited_ids.update(day_log.get("visited", []))

    total_weight = sum(weights.get(h.get("category", "routine"), 1.0) for h in all_hh)
    visited_weight = sum(
        weights.get(h.get("category", "routine"), 1.0)
        for h in all_hh
        if h.get("id") in visited_ids
    )
    coverage_score = visited_weight / total_weight if total_weight > 0 else 0.0

    # Sanitize tb_compliance_rate — could be None/NaN if no TB patients exist
    tb_score = _safe_float(getattr(state, "tb_compliance_rate", 0.0), default=0.0)
    # Clamp tb_score to [0, 1] before combining
    tb_score = max(0.0, min(1.0, tb_score))

    return _clamp(0.5 * coverage_score + 0.5 * tb_score)


def grade_task3(state: Any) -> float:
    """
    Score = 0.60 × (1 − disease_burden_index)
          + 0.25 × equity_score
          + 0.15 × routing_efficiency_score

    disease_burden_index: weighted average risk across all alive households.
    equity_score:         how evenly visits are distributed across clusters.
    routing_efficiency:   fraction of days where travel < 35% of working time.
                          Penalises agents that waste time on inefficient routes.
    Score range: (0.001, 0.999) after clamp.  Deterministic.
    """
    all_hh = getattr(state, "all_households", None) or []
    visit_history = getattr(state, "visit_history", None) or []

    # Sanitize disease_burden_index — clamp to [0, 1] before inverting
    raw_dbi = _safe_float(getattr(state, "disease_burden_index", 0.5), default=0.5)
    raw_dbi = max(0.0, min(1.0, raw_dbi))
    dbi_score = 1.0 - raw_dbi

    # Equity: visits per geo cluster (Gini-based)
    cluster_visits: dict[int, int] = {i: 0 for i in range(5)}
    hh_cluster_map = {h.get("id"): h.get("geo_cluster", 4) for h in all_hh}
    for day_log in visit_history:
        for hh_id in day_log.get("visited", []):
            c = hh_cluster_map.get(hh_id, 4)
            cluster_visits[c] = cluster_visits.get(c, 0) + 1

    total_v = sum(cluster_visits.values())
    if total_v > 0:
        n = len(cluster_visits)
        props = [v / total_v for v in cluster_visits.values()]
        gini = sum(abs(props[i] - props[j]) for i in range(n) for j in range(n))
        gini /= (2 * n * max(sum(props) / n, 1e-9) * n)
        equity_score = max(0.0, min(1.0, 1.0 - gini))
    else:
        equity_score = 0.0

    # Routing efficiency: fraction of days where travel time was < 35% of DAY_END_MIN
    DAY_END = 360.0
    efficient_days = sum(
        1 for d in visit_history
        if _safe_float(d.get("total_travel_min", DAY_END), DAY_END) < DAY_END * 0.35
    )
    routing_score = (
        efficient_days / len(visit_history)
        if visit_history else 0.0
    )

    return _clamp(
        0.60 * dbi_score
        + 0.25 * equity_score
        + 0.15 * routing_score
    )


GRADERS = {
    "task1": grade_task1,
    "task2": grade_task2,
    "task3": grade_task3,
}


def run_grader(task_id: str, state) -> float:
    """
    Run the grader for the given task.
    Accepts either a dict or an object with attribute access.
    Always returns a float strictly in (0.001, 0.999).
    """
    from types import SimpleNamespace
    if isinstance(state, dict):
        state = SimpleNamespace(**state)
    if task_id not in GRADERS:
        raise ValueError(f"Unknown task_id: {task_id!r}. Valid: {list(GRADERS)}")
    try:
        score = GRADERS[task_id](state)
        # Final safety net — should never be needed but guarantees the contract
        return _clamp(score)
    except Exception:
        # If grader crashes for any reason, return a safe mid-range value
        return 0.001