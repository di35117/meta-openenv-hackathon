import uvicorn
from fastapi import FastAPI
from .my_env_environment import MyEnvironment
from .tasks import run_grader

try:
    from ..models import Action, Observation
except ImportError:
    from models import Action, Observation

app = FastAPI(title="ASHA Village Health OpenEnv")

# Single environment instance — no session routing complexity
_env = MyEnvironment()


@app.post("/reset")
def reset(body: dict = {}):
    task_id = body.get("task_id", "task1")
    obs = _env.reset(task_id=task_id)
    return {"observation": obs.model_dump(), "reward": 0.0, "done": False}


@app.post("/step")
def step(body: dict):
    action_data = body.get("action", body)
    action = Action(**action_data)
    obs = _env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward":      obs.reward,
        "done":        obs.done,
    }


@app.get("/state")
def state():
    return _env.get_full_state()


@app.get("/grade")
def grade():
    state_data = _env.get_full_state()
    score = run_grader(state_data.get("task_id", "task1"), state_data)
    return {
        "score":   score,
        "task_id": state_data.get("task_id", "task1"),
    }


@app.get("/health")
def health():
    return {
        "status":      "healthy",
        "environment": "asha-village-health",
        "version":     "2.0.0",
    }


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()