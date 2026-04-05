import uvicorn
from fastapi import FastAPI
from .my_env_environment import MyEnvironment
from .tasks import run_grader

try:
    from ..models import Action, Observation
except ImportError:
    from models import Action, Observation

app = FastAPI(title="ASHA Village Health OpenEnv")

# One isolated environment per task_id
_envs = {}

def get_env(task_id: str) -> MyEnvironment:
    if task_id not in _envs:
        _envs[task_id] = MyEnvironment()
    return _envs[task_id]


@app.get("/")
def root():
    return {
        "environment": "asha-village-health",
        "version":     "2.0.0",
        "docs":        "/docs",
        "endpoints":   ["/reset", "/step", "/state", "/grade", "/health"],
    }


@app.post("/reset")
def reset(body: dict = {}):
    task_id = body.get("task_id", "task1")
    env = get_env(task_id)
    obs = env.reset(task_id=task_id)
    return {"observation": obs.model_dump(), "reward": 0.0, "done": False}


@app.post("/step")
def step(body: dict):
    action_data = body.get("action", body)
    task_id = body.get("task_id", "task1")
    env = get_env(task_id)
    action = Action(**action_data)
    obs = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward":      obs.reward,
        "done":        obs.done,
    }


@app.get("/state")
def state(task_id: str = "task1"):
    return get_env(task_id).get_full_state()


@app.get("/grade")
def grade(task_id: str = "task1"):
    env = get_env(task_id)
    state_data = env.get_full_state()
    score = run_grader(task_id, state_data)
    return {"score": score, "task_id": task_id}


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