class TaskConfig:
    def __init__(self, id, n_households, max_days, seed):
        self.id = id
        self.n_households = n_households
        self.max_days = max_days
        self.seed = seed

TASKS = {
    "task1": TaskConfig("task1", 30, 1, seed=42),
    "task2": TaskConfig("task2", 100, 7, seed=101),
    "task3": TaskConfig("task3", 200, 30, seed=2026)
}