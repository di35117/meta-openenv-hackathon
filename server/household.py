import random

class Household:
    DETERIORATION_RATES = {
        "newborn":        0.12,
        "tb_patient":     0.06,
        "high_risk_preg": 0.04,
        "diabetic":       0.02,
        "routine":        0.005,
    }

    def __init__(self, id, category, geo_cluster):
        self.id = id
        self.category = category
        self.geo_cluster = geo_cluster
        self.risk_score = self._initial_risk()
        self.days_since_visit = random.randint(0, 2) 
        self.danger_sign_active = self.risk_score > 0.75
        self.known_to_asha = True
        self._tb_window_counter = 0  
        self.tb_doses_missed = 0
        self.is_dead = False

    def _initial_risk(self):
        base_risks = {"newborn": 0.4, "tb_patient": 0.3, "high_risk_preg": 0.3, "diabetic": 0.2, "routine": 0.05}
        return min(0.8, base_risks[self.category] + random.uniform(0.0, 0.2))

    def tick(self):
        if self.is_dead:
            return

        rate = self.DETERIORATION_RATES.get(self.category, 0.01)
        self.risk_score = min(1.0, self.risk_score + rate)
        self.days_since_visit += 1
        
        if self.risk_score > 0.75:
            self.danger_sign_active = True

        if self.risk_score >= 0.97 and self.days_since_visit >= 3:
            self.is_dead = True

        if self.category == "tb_patient":
            self._tb_window_counter += 1
            if self._tb_window_counter > 0 and self._tb_window_counter % 3 == 0:
                self.tb_doses_missed += 1

    def receive_visit(self) -> dict:
        found = {
            "danger_sign": self.danger_sign_active,
            "risk_before": self.risk_score,
            "referral_needed": self.risk_score > 0.75,
        }
        self.risk_score = max(0.05, self.risk_score * 0.2)
        self.days_since_visit = 0
        self.danger_sign_active = False
        self._tb_window_counter = 0 
        return found