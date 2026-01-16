import numpy as np
from forensic.ela import ELADetector

class TamperingDetector:
    def __init__(self, ela_weight=1.0, autoencoder_weight=0.0, autoencoder_path=None):
        self.ela_weight = ela_weight
        self.autoencoder_weight = autoencoder_weight
        self.autoencoder_path = autoencoder_path
        self.ela_detector = ELADetector()

    def analyze_image(self, image, include_details=False):
        ela_score = self.ela_detector.compute_ela_score(image)

        # Normalisation simple (0–100)
        tampering_score = min(100, ela_score * self.ela_weight)

        results = {
            "tampering_score": round(tampering_score, 2),
            "ela_score": round(ela_score, 4),
            "is_tampered": tampering_score > 40,
            "confidence_level": "High" if tampering_score > 70 else "Medium" if tampering_score > 40 else "Low"
        }

        if include_details:
            return results
        return tampering_score
