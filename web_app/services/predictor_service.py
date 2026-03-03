# ============================================
# Predictor Service
# ============================================

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.inference_engine import InferenceEngine


class PredictorService:

    def __init__(self, model, drug_map, side_map, device, side_to_vn=None):
        self.engine = InferenceEngine(
            model=model,
            drug_map=drug_map,
            side_map=side_map,
            device=device,
            side_to_vn=side_to_vn
        )

    def predict_pair(self, drug1, drug2):
        return self.engine.predict_pair(drug1, drug2)