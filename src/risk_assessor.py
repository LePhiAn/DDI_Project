# ============================================
# Advanced Risk Assessment Module
# ============================================

import numpy as np


class RiskAssessor:
    """
    Converts raw probability into structured risk levels
    and clinical-style alerts.
    """

    def __init__(self):
        # You can tune these thresholds later
        self.thresholds = {
            "low": 0.3,
            "moderate": 0.6,
            "high": 0.8
        }

    def classify_risk(self, probability: float) -> str:
        if probability < self.thresholds["low"]:
            return "LOW"
        elif probability < self.thresholds["moderate"]:
            return "MODERATE"
        elif probability < self.thresholds["high"]:
            return "HIGH"
        else:
            return "CRITICAL"

    def confidence_score(self, probability: float) -> float:
        """
        Confidence based on distance from decision boundary (0.5)
        """
        return round(abs(probability - 0.5) * 2, 4)

    def generate_alert(self, probability: float, side_effect: str):
        risk_level = self.classify_risk(probability)
        confidence = self.confidence_score(probability)

        alert_message = {
            "risk_level": risk_level,
            "probability": round(probability, 4),
            "confidence": confidence,
            "side_effect": side_effect,
            "clinical_recommendation": self._clinical_guideline(risk_level)
        }

        return alert_message

    def _clinical_guideline(self, risk_level: str) -> str:
        if risk_level == "LOW":
            return "Monitor patient condition. Interaction risk minimal."
        elif risk_level == "MODERATE":
            return "Consider dosage adjustment or monitoring."
        elif risk_level == "HIGH":
            return "Avoid co-administration if possible."
        else:
            return "Contraindicated combination. Immediate review required."