"""
inference.py — Baseline LLM agent for ASHA Village Health OpenEnv.

REQUIRED by hackathon:
  • Named inference.py, placed at project root
  • Uses OpenAI client for all LLM calls
  • Reads credentials from environment variables:
      API_BASE_URL — LLM API endpoint
      MODEL_NAME   — model identifier
      HF_TOKEN     — API key (used as openai api_key)
  • Reads environment URL:
      ENV_URL      — environment server (default http://localhost:8000)
  • Produces reproducible scores for all 3 tasks
  • Runs in < 20 minutes total
"""

import json
import os
import sys
import time
from typing import List

import requests
from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     os.environ.get("OPENAI_API_KEY", ""))
ENV_URL      = os.environ.get("ENV_URL",      "http://localhost:8000")

TEMPERATURE     = 0.0
MAX_TOKENS      = 512
MAX_STEPS       = 32
REQUEST_TIMEOUT = 30

TASK_MAX_DAYS = {"task1": 1, "task2": 7, "task3": 30}

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "placeholder")

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an AI scheduling assistant for Sunita, an ASHA (Accredited Social Health Activist)
worker in rural India. Each day you plan which households she visits and in what order.

SUNITA'S CONSTRAINTS:
- 6 working hours = 360 minutes (7:00 AM to 1:00 PM)
- Maximum 15 visits per day
- She walks between households (no vehicle)

CATEGORY PRIORITY (highest to lowest):
  1. newborn         — CRITICAL: must visit within 48hrs of birth (neonatal sepsis risk)
  2. tb_patient      — HIGH: DOTS protocol requires supervised dose every 3 days
  3. high_risk_preg  — HIGH: weekly monitoring (eclampsia, haemorrhage risk)
  4. diabetic        — MEDIUM: bi-weekly monitoring
  5. routine         — LOWEST: monthly wellness check

TRAVEL TIME FORMULA (approximate):
  speed_kmh = road_quality x 5.0  (paved=5, gravel=3.5, dirt=2, mud=0.8)
  If heavy_rain: speed_kmh x 0.4 for dirt paths (road_quality < 0.4)
  travel_min = (dist_from_asha_home_km / speed_kmh) x 60

DECISION RULES:
  1. If danger_sign_active=true -> visit FIRST regardless of distance
  2. If newborn AND days_since_birth <= 1 -> must visit today
  3. Skip households with road_quality < 0.12 during heavy_rain (unreachable)
  4. Group households in the same geo_cluster to save travel time
  5. If est_visit_duration_min + travel is large -> place it later in the day

RESPONSE FORMAT — return ONLY valid JSON, nothing else:
{"visit_sequence": [id1, id2, id3, ...]}

Order the list by priority: most urgent first, then cluster nearby households.
""".strip()

FALLBACK_ACTION = '{"visit_sequence": []}'

# ── Environment HTTP helpers ──────────────────────────────────────────────────

def env_post(path: str, **kwargs) -> dict:
    resp = requests.post(f"{ENV_URL}{path}", timeout=REQUEST_TIMEOUT, **kwargs)
    resp.raise_for_status()
    return resp.json()


def env_get(path: str) -> dict:
    resp = requests.get(f"{ENV_URL}{path}", timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()

# ── Greedy fallback ───────────────────────────────────────────────────────────

def greedy_sequence(obs: dict) -> List[int]:
    households = obs.get("households", [])
    weather    = obs.get("weather", {}).get("condition", "sunny")
    time_left  = 360 - obs.get("current_time_min", 0)

    CATEGORY_WEIGHT = {
        "newborn":        3.0,
        "tb_patient":     2.0,
        "high_risk_preg": 1.8,
        "diabetic":       1.2,
        "routine":        0.5,
    }

    def estimate_travel_min(h: dict) -> float:
        rq    = max(0.05, h.get("road_quality", 0.5))
        dist  = h.get("dist_from_asha_home_km", 1.0)
        speed = rq * 5.0
        if weather == "heavy_rain" and rq < 0.40:
            speed *= 0.40
        return (dist / max(0.1, speed)) * 60.0

    def score(h: dict) -> float:
        s = h.get("risk_score", 0)
        if h.get("danger_sign_active"):
            s += 1.5
        s += CATEGORY_WEIGHT.get(h.get("category", "routine"), 0.5)
        rq = h.get("road_quality", 0.5)
        if weather == "heavy_rain" and rq < 0.12:
            s -= 5.0
        s -= h.get("dist_from_asha_home_km", 1.0) * 0.03
        return s

    ranked   = sorted(households, key=score, reverse=True)
    selected = []
    elapsed  = 0.0
    for h in ranked:
        if len(selected) >= 15:
            break
        t_travel = estimate_travel_min(h)
        t_visit  = h.get("est_visit_duration_min", 25)
        if elapsed + t_travel + t_visit > time_left:
            continue
        selected.append(h["id"])
        elapsed += t_travel + t_visit

    return selected

# ── LLM prompt builder ────────────────────────────────────────────────────────

def build_prompt(obs: dict, task_id: str, step_num: int) -> str:
    day        = obs.get("day", 0)
    max_days   = TASK_MAX_DAYS.get(task_id, 1)
    season     = obs.get("season", "summer")
    weather    = obs.get("weather", {})
    cond       = weather.get("condition", "sunny")
    temp       = weather.get("temp_celsius", 30)
    rain       = weather.get("rainfall_mm", 0)
    rqm        = weather.get("road_quality_modifier", 1.0)
    time_used  = obs.get("current_time_min", 0)
    energy     = obs.get("asha_energy_pct", 100)
    alerts     = obs.get("new_alerts", [])
    households = obs.get("households", [])

    def display_score(h: dict) -> float:
        s = h.get("risk_score", 0) * 2
        if h.get("danger_sign_active"):
            s += 3
        cats = {"newborn": 4, "tb_patient": 2, "high_risk_preg": 2, "diabetic": 1}
        s += cats.get(h.get("category", "routine"), 0)
        s -= h.get("dist_from_asha_home_km", 1) * 0.02
        return s

    top_hhs = sorted(households, key=display_score, reverse=True)[:30]

    rows = []
    for h in top_hhs:
        rq        = h.get("road_quality", 0.5)
        dist      = h.get("dist_from_asha_home_km", 1.0)
        danger    = "DANGER!" if h.get("danger_sign_active") else ""
        rain_warn = " [RAIN+MUD]" if cond == "heavy_rain" and rq < 0.20 else ""
        rows.append(
            f"  {h['id']:>4}  {h['category']:20s}  risk={h['risk_score']:.2f}  "
            f"days={h['days_since_visit']:>2}  dist={dist:.1f}km  "
            f"road={rq:.2f}  est={h.get('est_visit_duration_min',25):>3}min  "
            f"cluster={h.get('geo_cluster',0)}  {danger}{rain_warn}"
        )

    alert_lines = [
        f"  !! {a['urgency'].upper()}: HH {a['household_id']} - {a['message']}"
        for a in alerts[:5]
    ] or ["  None"]

    time_remaining = 360 - time_used

    return (
        f"Day {day + 1}/{max_days} | Season: {season} | "
        f"Weather: {cond} {temp}C {rain}mm rain\n"
        f"Road quality modifier today: {rqm:.2f} (1.0=normal, <0.5=roads degraded)\n"
        f"Time used: {time_used} min | Time remaining: {time_remaining} min | "
        f"ASHA energy: {energy:.0f}%\n\n"
        f"ALERTS (act on these immediately):\n"
        f"{chr(10).join(alert_lines)}\n\n"
        f"TOP {len(top_hhs)} HOUSEHOLDS BY URGENCY "
        f"(showing {len(top_hhs)} of {len(households)} total):\n"
        f"    ID  Category              Risk  Days  Dist   Road  EstMin  Cluster  Notes\n"
        f"{chr(10).join(rows)}\n\n"
        f"You have {time_remaining} minutes left today.\n"
        f"Plan a route that fits in this time. Group clusters to reduce travel.\n"
        f'Return ONLY JSON: {{"visit_sequence": [id1, id2, ...]}}'
    )

# ── Action parser ─────────────────────────────────────────────────────────────

def parse_action(text: str) -> dict:
    text  = text.strip()
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start:end])
            if "visit_sequence" in obj and isinstance(obj["visit_sequence"], list):
                return obj
        except json.JSONDecodeError:
            pass
    return json.loads(FALLBACK_ACTION)

# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(task_id: str) -> dict:
    print(f"\n{'─'*60}")
    print(f"  TASK: {task_id.upper()}")
    print(f"{'─'*60}")

    # Reset — unwrap observation from wrapper
    reset_raw = env_post("/reset", json={"task_id": task_id})
    obs       = reset_raw.get("observation", reset_raw)

    n_hh     = len(obs.get("households", []))
    max_days = TASK_MAX_DAYS.get(task_id, 1)
    print(
        f"  Households: {n_hh} | Days: {max_days} | "
        f"Season: {obs.get('season')} | "
        f"Weather: {obs.get('weather', {}).get('condition')}"
    )

    total_reward = 0.0
    step_num     = 0

    for step_num in range(MAX_STEPS):
        user_prompt = build_prompt(obs, task_id, step_num)

        # LLM call
        action_obj = None
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            raw        = completion.choices[0].message.content or FALLBACK_ACTION
            action_obj = parse_action(raw)
        except Exception as exc:
            print(f"  [LLM error day {step_num+1}] {exc} — using greedy fallback")
            action_obj = {"visit_sequence": greedy_sequence(obs)}

        visit_seq = action_obj.get("visit_sequence", [])

        # Step — action wrapped correctly for openenv-core
        step_raw = env_post("/step", json={"action": {"visit_sequence": visit_seq}})

        # KEY FIX: read done/reward from TOP-LEVEL response only
        reward = step_raw.get("reward", 0.0)
        done   = step_raw.get("done",   False)

        # Unwrap observation for next loop iteration
        obs  = step_raw.get("observation", step_raw)
        meta = obs.get("metadata", {})
        total_reward += reward

        print(
            f"  Day {step_num+1:>2}/{max_days}  "
            f"visits={meta.get('visits_completed', 0):>2}  "
            f"travel={meta.get('total_travel_min', 0):>4.0f}min  "
            f"energy={meta.get('asha_energy_end', 100):>4.0f}%  "
            f"deaths={meta.get('new_deaths', 0)}  "
            f"weather={obs.get('weather', {}).get('condition', '?'):10s}  "
            f"r={reward:+.3f}"
        )

        if done:
            break

        time.sleep(0.05)

    # Grade
    score = 0.0
    try:
        grade = env_get("/grade")
        score = grade.get("score", 0.0)
    except Exception:
        score = round(max(0.0, min(1.0, total_reward / max(step_num + 1, 1))), 4)

    # Extra metrics from state
    env_state = {}
    try:
        env_state = env_get("/state")
    except Exception:
        pass

    dbi  = env_state.get("disease_burden_index", 0.0)
    tb   = env_state.get("tb_compliance_rate",   0.0)
    dead = env_state.get("preventable_deaths",   0)

    print(f"\n  {'─'*50}")
    print(f"  Score          : {score:.4f}")
    print(f"  Disease burden : {dbi:.4f}")
    print(f"  TB compliance  : {tb:.4f}")
    print(f"  Deaths         : {dead}")
    print(f"  Total reward   : {total_reward:.3f}")

    return {
        "task_id":              task_id,
        "score":                score,
        "disease_burden_index": dbi,
        "tb_compliance_rate":   tb,
        "preventable_deaths":   dead,
        "total_reward":         round(total_reward, 3),
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ASHA Village Health — OpenEnv Baseline Inference")
    print("=" * 60)
    print(f"  Model   : {MODEL_NAME}")
    print(f"  Env URL : {ENV_URL}")
    print(f"  Tasks   : task1 (easy) -> task2 (medium) -> task3 (hard)")

    try:
        health = env_get("/health")
        print(
            f"  Env OK  : {health.get('environment', 'asha-village-health')} "
            f"v{health.get('version', '2.0.0')}"
        )
    except Exception as exc:
        print(f"  [WARN] Cannot reach environment at {ENV_URL}: {exc}")
        print("  Make sure the server is running: uvicorn server.app:app --port 8000")
        sys.exit(1)

    results = []
    for task_id in ["task1", "task2", "task3"]:
        try:
            result = run_episode(task_id)
            results.append(result)
        except Exception as exc:
            print(f"\n  [ERROR] Task {task_id} failed: {exc}")
            results.append({
                "task_id":              task_id,
                "score":                0.0,
                "disease_burden_index": 1.0,
                "tb_compliance_rate":   0.0,
                "preventable_deaths":   -1,
                "total_reward":         0.0,
            })

    print(f"\n{'='*60}")
    print("  FINAL BASELINE SCORES")
    print(f"{'='*60}")
    print(f"  {'Task':<8}  {'Score':>7}  {'DBI':>7}  {'TB%':>7}  {'Deaths':>7}")
    print(f"  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")
    for r in results:
        print(
            f"  {r['task_id']:<8}  {r['score']:>7.4f}  "
            f"{r['disease_burden_index']:>7.4f}  "
            f"{r['tb_compliance_rate']:>7.4f}  "
            f"{r['preventable_deaths']:>7}"
        )

    avg = sum(r["score"] for r in results) / max(len(results), 1)
    print(f"\n  Average score: {avg:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
