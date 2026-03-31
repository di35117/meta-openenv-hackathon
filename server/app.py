import uvicorn
from openenv.core.env_server import create_app  # <-- THE FIX IS HERE
from .my_env_environment import MyEnvironment

try:
    from ..models import Action, Observation
except ImportError:
    from models import Action, Observation

# Create the FastAPI app
app = create_app(
    MyEnvironment,
    Action,
    Observation,
    max_concurrent_envs=10,
)

# Required by OpenEnv validator for multi-mode deployment
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()