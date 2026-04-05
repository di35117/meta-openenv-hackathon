---
title: ASHA Village Health OpenEnv
emoji: 🏥
colorFrom: green
colorTo: green
sdk: docker
app_port: 8000
pinned: false
tags:
  - openenv
---

# ASHA Village Health — OpenEnv

> An RL environment for training AI to help 1 million ASHA community health workers in rural India make better daily visit decisions.

---

## The Real-World Problem

India has **~1 million ASHA (Accredited Social Health Activist) workers**, each covering 1,000 people across ~200 households with only **6 walking hours per day**. Every morning, Sunita — an ASHA worker in Chhattisgarh — must decide which 12–15 of her 200 households to visit today. She does this entirely from memory, with no decision support.

The wrong decision kills people:
- A newborn not visited in the first 48 hours → neonatal sepsis goes undetected
- A TB patient missed for 4+ days → drug-resistant TB begins to develop
- A high-risk pregnancy not seen for 2 weeks → eclampsia strikes without warning

**This environment trains an RL agent to produce Sunita's optimal daily visit list.**

---

## Why RL — Not ML or LLM Prompts

| Approach | Why It Fails |
|----------|-------------|
| Risk scoring (ML) | Scores individual households but cannot solve the joint scheduling problem across 200 households with a 6-hour budget |
| LLM prompting | Reasons about a single day but cannot learn across thousands of episodes that monsoon batching saves 1.4 lives/month |
| **RL** | Discovers multi-week policies through delayed rewards — learns that skipping TB on day 3 causes drug resistance on day 21 |

---

## Dynamic Features

This is a fully dynamic environment. Every factor that constrains a real ASHA worker is modelled:

### Real 2-D Map (10 km × 10 km village)
- 5 geographic clusters: North (5,8), South (5,2), East (8,5), West (2,5), Centre (5,5)
- Each household has real `(x, y)` coordinates scattered around its cluster
- ASHA starts at home `(5, 5)` each morning; position updates after each visit

### Road Quality Network
- Main paved road: y = 5.0
- Secondary road: x = 5.0
- Households near roads: `road_quality` 0.80–0.98 (fast walking)
- Households far from roads: `road_quality` 0.05–0.18 (slow, mud paths)
- Today's weather multiplies every road quality value

### Real Travel Time
```
speed_kmh = road_quality × 5.0
if heavy_rain and road_quality < 0.4:
    speed_kmh × 0.40   ← mud paths become streams
travel_min = (distance_km / speed_kmh) × 60
```
A household 2 km away costs:
- 24 min on paved road in sun
- 300 min on dirt path in heavy rain → effectively unreachable

### 6-Hour Clock
The step loop tracks real minutes. When `elapsed_min + travel + visit > 360`, the day ends — no fixed visit count cap. Rainy days shrink the effective visit window.

### ASHA Fatigue
- Energy starts at 100 each morning
- Walking: −0.15 energy/minute
- Visiting: −0.10 energy/minute
- Below 60% energy → visits take longer and reset risk less
- Below 30% energy → significant quality degradation

### Daily Weather Engine
| Season | Sunny | Cloudy | Rainy | Heavy Rain |
|--------|-------|--------|-------|------------|
| Summer  | 60%  | 28%    | 10%   | 2%        |
| Monsoon | 8%   | 22%    | 42%   | 28%       |
| Winter  | 52%  | 34%    | 11%   | 3%        |

### Seasonal Disease Dynamics
Deterioration rates multiply by season:
- Monsoon: newborns ×1.5, malaria/waterborne ×2.0 for routine households
- Winter: newborns ×1.6 (hypothermia/pneumonia), TB ×1.4

Disease event rate scales with season (monsoon = 3.5× summer).

---

## Action & Observation Spaces

### Action
```json
{ "visit_sequence": [42, 17, 103, 8, ...] }
```
Ordered list of household IDs. First = highest priority. Max 15.

### Observation (per day)
```json
{
  "day": 3,
  "season": "monsoon",
  "weather": { "condition": "heavy_rain", "temp_celsius": 29, "rainfall_mm": 45, "road_quality_modifier": 0.38 },
  "current_time_min": 0,
  "asha_energy_pct": 100.0,
  "asha_home_x": 5.0,
  "asha_home_y": 5.0,
  "households": [
    {
      "id": 42, "category": "newborn", "risk_score": 0.71,
      "days_since_visit": 1, "danger_sign_active": true,
      "geo_cluster": 2, "x": 7.3, "y": 6.1,
      "road_quality": 0.29,
      "dist_from_asha_home_km": 2.4,
      "est_visit_duration_min": 55
    }
  ],
  "new_alerts": [
    { "household_id": 42, "message": "HH 42: new birth — 2.4 km, est 89 min travel (heavy_rain)", "urgency": "critical" }
  ]
}
```

---

## Reward Function

| Signal | Value |
|--------|-------|
| Danger sign caught | +0.40 |
| Correct referral | +0.30 |
| Newborn visited ≤ 48 hrs | +0.25 |
| TB dose supervised on time | +0.15 |
| High-risk category visit | +0.05 |
| Geographic clustering bonus | +0.03–0.05 |
| Preventable death | **−1.00** |
| Wasted visit (stable routine) | −0.05 |

---

## Tasks

| Task | Households | Days | Season | Score Formula |
|------|-----------|------|--------|---------------|
| `task1` — Easy | 30 | 1 | Summer | `danger_caught / total_danger` |
| `task2` — Medium | 100 | 7 | Summer | `0.5×coverage + 0.5×tb_compliance` |
| `task3` — Hard | 200 | 30 | Monsoon | `0.6×(1−DBI) + 0.25×equity + 0.15×routing` |

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/reset?task_id=task1` | Start a new episode |
| POST | `/step` | Execute one day of visits |
| GET  | `/state` | Full internal state |
| GET  | `/grade` | Run grader → score 0.0–1.0 |
| GET  | `/health` | Liveness check |
| GET  | `/docs` | Interactive API docs |

---

## Quick Start

### Local development
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/asha-village-health
cd asha-village-health
pip install -r requirements.txt
uvicorn server.app:app --reload --port 8000
```

### Docker
```bash
docker build -t asha-openenv .
docker run -p 8000:8000 asha-openenv
# Test: curl -X POST http://localhost:8000/reset
```

### Run inference baseline
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="sk-..."
export ENV_URL="http://localhost:8000"
python inference.py
```

---

## Baseline Scores (GPT-4o-mini greedy agent)

| Task | Score | Disease Burden | TB Compliance | Deaths |
|------|-------|----------------|---------------|--------|
| task1 | ~0.85 | ~0.15 | 1.00 | 0 |
| task2 | ~0.45 | ~0.35 | ~0.55 | 0 |
| task3 | ~0.40 | ~0.55 | ~0.25 | ~50 |

A trained RL agent is expected to significantly improve task3 by discovering:
- Monsoon cluster batching (visit east/west clusters only on non-rain days)
- TB 3-day cycle enforcement
- Pregnancy escalation scheduling (daily visits in the final week)

---

## File Structure
```
asha-village-health/
├── models.py                  Pydantic data models
├── inference.py               Baseline LLM agent (hackathon required)
├── openenv.yaml               OpenEnv spec metadata
├── Dockerfile                 Container definition
├── requirements.txt           Python dependencies
├── README.md                  This file
└── server/
    ├── app.py                 FastAPI HTTP server
    ├── household.py           Health state machine (dynamic)
    ├── village.py             Village map + travel + weather
    ├── my_env_environment.py  RL environment: step/reset/state
    └── tasks.py               3 task configs + graders
```

---

## Impact

If a trained policy improves visit efficiency by 15% across 1 million ASHA workers, that is **150,000 additional correct visits per day across India** — newborns reached in time, TB doses supervised, pregnancies caught before emergencies. This environment is the training ground for that policy.
