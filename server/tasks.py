"""
tasks.py — Task configurations and deterministic graders.

BUG FIXED (this revision):
  The previous _clamp used 0.999 as the upper bound.
  However:
      round(0.999, 2)     == 1.0    (Python floating-point)
      f"{0.999:.4f}"       == '0.9990'   (fine)

  BUT the live server was running OLD code without _clamp at all,
  returning the raw fraction 1.0 directly when all dangers were caught.
  That raw 1.0 displayed as 1.0000 in the inference table.

  FIX — two changes:
    1. Upper bound changed from 0.999 → 0.990
       round(0.990, 2) == 0.99   ✓  never reaches 1.00
       round(0.990, 4) == 0.9900 ✓  never reaches 1.0000
    2. _safe_float default for disease_burden_index kept at 0.5.
    3. All grader exit paths go through _clamp (no bare literals).
    4. run_grader applies _clamp as final safety net.

VALIDATOR REQUIREMENT:
  Scores must be in the OPEN interval (0, 1) — 0.0 and 1.0 are rejected.
  This version guarantees every score is in [0.001, 0.990].
"""

from dataclasses import dataclass
from typing import Any
import math


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clamp(score) -> float:
    """
    Force score into the open interval (0, 1) as required by the validator.

    Specifically maps to [0.0010, 0.9900]:
      • Lower bound 0.0010 — safely above 0.0
      • Upper bound 0.9900 — safely below 1.0 under ANY display format:
            round(0.990, 2) = 0.99   (not 1.00)
            round(0.990, 4) = 0.99   (not 1.0000)
            f"{0.990:.4f}"  = '0.9900' (not '1.0000')

    Handles: None, Python float NaN/inf, numpy.float64 NaN/inf,
             exact 0.0, exact 1.0, and any out-of-range value.
    """
    try:
        v = float(score)
        if math.isnan(v) or math.isinf(v):
            return 0.0010
        clamped = max(0.0010, min(0.9900, v))
        return round(clamped, 4)
    except (TypeError, ValueError):
        return 0.0010


def _safe_float(value, default: float = 0.0) -> float:
    """Convert value to float safely; return default if None/NaN/invalid."""
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Task configurations
# ─────────────────────────────────────────────────────────────────────────────

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

    All return paths go through _clamp → guaranteed in [0.001, 0.990].
    """
    visit_history = getattr(state, "visit_history", None) or []

    if not visit_history:
        return _clamp(0.0)

    # Collect all danger-sign IDs that existed at the START of each day
    danger_before: set[int] = set()
    for day_log in visit_history:
        danger_before.update(day_log.get("danger_ids_before", []))

    if not danger_before:
        # No danger signs existed — full credit, but clamped below 1.0
        return _clamp(1.0)

    visited_ids: set[int] = set()
    for day_log in visit_history:
        visited_ids.update(day_log.get("visited", []))

    caught = len(danger_before & visited_ids)
    raw_score = caught / len(danger_before)   # could be exactly 1.0
    return _clamp(raw_score)                  # → max 0.990


def grade_task2(state: Any) -> float:
    """
    Score = 0.5 × weighted_coverage  +  0.5 × tb_compliance_rate

    weighted_coverage: visits weighted by household category importance.
    tb_compliance_rate: fraction of 3-day DOTS windows honoured.
    All values sanitized and clamped → guaranteed in [0.001, 0.990].
    """
    weights = {
        "newborn":        3.0,
        "tb_patient":     2.0,
        "high_risk_preg": 2.5,
        "diabetic":       1.5,
        "routine":        1.0,
    }

    all_hh        = getattr(state, "all_households",   None) or []
    visit_history = getattr(state, "visit_history",    None) or []

    visited_ids: set[int] = set()
    for day_log in visit_history:
        visited_ids.update(day_log.get("visited", []))

    total_weight = sum(
        weights.get(h.get("category", "routine"), 1.0) for h in all_hh
    )
    visited_weight = sum(
        weights.get(h.get("category", "routine"), 1.0)
        for h in all_hh
        if h.get("id") in visited_ids
    )
    coverage_score = visited_weight / total_weight if total_weight > 0 else 0.0

    # Sanitize tb_compliance_rate — could be None/NaN if no TB patients
    raw_tb = _safe_float(getattr(state, "tb_compliance_rate", 0.0), default=0.0)
    tb_score = max(0.0, min(1.0, raw_tb))

    combined = 0.5 * coverage_score + 0.5 * tb_score
    return _clamp(combined)


def grade_task3(state: Any) -> float:
    """
    Score = 0.60 × (1 − disease_burden_index)
          + 0.25 × equity_score
          + 0.15 × routing_efficiency_score

    disease_burden_index: weighted average risk across all alive households.
    equity_score:         Gini-based equity across geographic clusters.
    routing_efficiency:   fraction of days where travel < 35% of working time.

    All components sanitized → combined score clamped → [0.001, 0.990].
    """
    all_hh        = getattr(state, "all_households",  None) or []
    visit_history = getattr(state, "visit_history",   None) or []

    # Disease burden: sanitize and clamp to [0, 1] before inverting
    raw_dbi = _safe_float(getattr(state, "disease_burden_index", 0.5), default=0.5)
    raw_dbi = max(0.0, min(1.0, raw_dbi))
    dbi_score = 1.0 - raw_dbi

    # Equity score: Gini coefficient across geo clusters
    cluster_visits: dict[int, int] = {i: 0 for i in range(5)}
    hh_cluster_map = {h.get("id"): h.get("geo_cluster", 4) for h in all_hh}
    for day_log in visit_history:
        for hh_id in day_log.get("visited", []):
            c = hh_cluster_map.get(hh_id, 4)
            cluster_visits[c] = cluster_visits.get(c, 0) + 1

    total_v = sum(cluster_visits.values())
    if total_v > 0:
        n     = len(cluster_visits)
        props = [v / total_v for v in cluster_visits.values()]
        gini  = sum(
            abs(props[i] - props[j]) for i in range(n) for j in range(n)
        )
        gini /= (2 * n * max(sum(props) / n, 1e-9) * n)
        equity_score = max(0.0, min(1.0, 1.0 - gini))
    else:
        equity_score = 0.0

    # Routing efficiency: fraction of days with travel < 35% of 360 min
    DAY_END = 360.0
    efficient_days = sum(
        1 for d in visit_history
        if _safe_float(d.get("total_travel_min", DAY_END), DAY_END) < DAY_END * 0.35
    )
    routing_score = (
        efficient_days / len(visit_history) if visit_history else 0.0
    )

    combined = (
        0.60 * dbi_score
        + 0.25 * equity_score
        + 0.15 * routing_score
    )
    return _clamp(combined)


# ─────────────────────────────────────────────────────────────────────────────
# Registry + public entry point
# ─────────────────────────────────────────────────────────────────────────────

GRADERS = {
    "task1": grade_task1,
    "task2": grade_task2,
    "task3": grade_task3,
}


def run_grader(task_id: str, state) -> float:
    """
    Run the grader for the given task.

    Accepts either a dict or an object with attribute access.
    Applies _clamp as a final safety net regardless of what the grader returns.
    Always returns a float in [0.001, 0.990].
    """
    from types import SimpleNamespace
    if isinstance(state, dict):
        state = SimpleNamespace(**state)
    if task_id not in GRADERS:
        raise ValueError(
            f"Unknown task_id: {task_id!r}. Valid: {list(GRADERS)}"
        )
    try:
        score = GRADERS[task_id](state)
        # Final safety net — guarantees the contract even if a grader slips
        return _clamp(score)
    except Exception:
        return 0.0010