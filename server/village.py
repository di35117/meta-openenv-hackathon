import random
from .household import Household

class Village:
    def __init__(self, n_households: int, seed: int):
        self.n_households = n_households
        random.seed(seed)
        categories = ["routine", "diabetic", "high_risk_preg", "tb_patient", "newborn"]
        weights = [0.60, 0.15, 0.10, 0.10, 0.05]
        
        self.households = {}
        for i in range(n_households):
            category = random.choices(categories, weights=weights)[0]
            geo_cluster = random.randint(0, 4)
            self.households[i] = Household(id=i, category=category, geo_cluster=geo_cluster)

    @property
    def preventable_deaths(self) -> int:
        return sum(1 for hh in self.households.values() if hh.is_dead)

    def generate_daily_events(self, day: int) -> list:
        new_alerts = []
        for hh in self.households.values():
            if hh.is_dead:
                continue
            if random.random() < 0.02 and not hh.danger_sign_active:
                hh.risk_score = min(1.0, hh.risk_score + 0.4)
                if hh.risk_score > 0.75:
                    hh.danger_sign_active = True
                    if random.random() < 0.5:
                        new_alerts.append({
                            "household_id": hh.id, 
                            "message": f"Sudden deterioration reported in cluster {hh.geo_cluster}", 
                            "urgency": "high"
                        })
            if hh.category == "routine" and random.random() < 0.005:
                hh.category = "newborn"
                hh.risk_score = 0.4
                hh.days_since_visit = 0
                hh.danger_sign_active = False
                new_alerts.append({
                    "household_id": hh.id, 
                    "message": "New birth reported! 48-hour visit required.", 
                    "urgency": "critical"
                })
        return new_alerts

    def compute_dbi(self) -> float:
        alive_hhs = [hh for hh in self.households.values() if not hh.is_dead]
        if not alive_hhs:
            return 1.0 
        return sum(hh.risk_score for hh in alive_hhs) / len(alive_hhs)