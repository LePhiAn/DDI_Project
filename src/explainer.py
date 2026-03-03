# ============================================
# Explainability Module for DDI
# ============================================

import numpy as np


class DDIExplainer:
    """
    Provides explainability and aggregated risk analysis
    """

    def aggregate_pair_risk(self, predictions: list):
        """
        predictions: list of risk dict from RiskAssessor
        """

        probabilities = [p["probability"] for p in predictions]

        avg_risk = float(np.mean(probabilities))
        max_risk = float(np.max(probabilities))

        risk_distribution = {
            "LOW": 0,
            "MODERATE": 0,
            "HIGH": 0,
            "CRITICAL": 0
        }

        for p in predictions:
            risk_distribution[p["risk_level"]] += 1

        return {
            "average_risk_score": round(avg_risk, 4),
            "max_risk_score": round(max_risk, 4),
            "risk_distribution": risk_distribution
        }

    def generate_explanation(self, top_predictions: list):
        """
        Generate human-readable explanation
        """

        critical = [p for p in top_predictions if p["risk_level"] == "CRITICAL"]
        high = [p for p in top_predictions if p["risk_level"] == "HIGH"]

        if len(critical) > 0:
            return (
                f"Critical interaction risk detected in {len(critical)} side effects. "
                "Immediate medical review recommended."
            )
        elif len(high) > 0:
            return (
                f"High interaction risk observed in {len(high)} side effects. "
                "Monitoring strongly advised."
            )
        else:
            return (
                "No severe interaction patterns detected. "
                "Standard monitoring sufficient."
            )