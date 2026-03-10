"""
Core Orchestrator — coordinates all 5 analysis layers and aggregates results.
"""

import time

from layers.metadata import MetadataAnalyzer
from layers.pixel_forensics import PixelForensicsAnalyzer
from layers.frequency_domain import FrequencyDomainAnalyzer
from layers.visual_anomalies import VisualAnomalyAnalyzer
from layers.provenance import ProvenanceAnalyzer


WEIGHTS = {
    "metadata": 0.25,
    "pixel_forensics": 0.20,
    "frequency_domain": 0.20,
    "visual_anomalies": 0.15,
    "provenance": 0.20,
}


class Orchestrator:
    """Coordinates all analysis layers and produces a final verdict."""

    def __init__(self):
        self.analyzers = {
            "metadata": MetadataAnalyzer(),
            "pixel_forensics": PixelForensicsAnalyzer(),
            "frequency_domain": FrequencyDomainAnalyzer(),
            "visual_anomalies": VisualAnomalyAnalyzer(),
            "provenance": ProvenanceAnalyzer(),
        }

    def analyze(self, image_path: str, original_filename: str = "") -> dict:
        start = time.time()

        layer_results = {}
        overrides = []

        for name, analyzer in self.analyzers.items():
            try:
                result = analyzer.analyze(
                    image_path, original_filename=original_filename
                )
            except Exception as e:
                result = {"score": 0, "findings": [f"Error in {name}: {str(e)}"], "error": True}

            layer_results[name] = result

            if result.get("override"):
                overrides.append(result["override"])

        # Calculate weighted score
        weighted_score = 0
        total_weight = 0

        for name, result in layer_results.items():
            if result.get("error"):
                continue
            weighted_score += result["score"] * WEIGHTS[name]
            total_weight += WEIGHTS[name]

        if total_weight > 0:
            final_score = int(weighted_score / total_weight)
        else:
            final_score = 0

        # Apply overrides
        for override in overrides:
            if override.get("force_verdict"):
                final_score = max(final_score, override.get("min_score", final_score))

        final_score = max(0, min(100, final_score))
        verdict = self._score_to_verdict(final_score)
        elapsed = round(time.time() - start, 2)

        report = {
            "verdict": verdict,
            "confidence": final_score,
            "analysis_time_seconds": elapsed,
            "layers": layer_results,
            "original_filename": original_filename,
        }

        # Collect heatmaps
        heatmaps = {}
        for name, result in layer_results.items():
            if result.get("heatmap"):
                heatmaps[name] = result["heatmap"]
        if heatmaps:
            report["heatmaps"] = heatmaps

        # Collect GAN fingerprint
        for name, result in layer_results.items():
            if result.get("gan_fingerprint"):
                report["gan_fingerprint_detected"] = True
                break

        return report

    @staticmethod
    def _score_to_verdict(score: int) -> str:
        if score >= 75:
            return "AI_GENERATED"
        elif score >= 50:
            return "LIKELY_AI"
        elif score >= 25:
            return "UNCERTAIN"
        else:
            return "LIKELY_REAL"
