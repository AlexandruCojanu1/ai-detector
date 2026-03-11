"""
Layer 4 — Visual Anomaly Detection
Modules: Lighting Consistency, Reflection Consistency, Perspective, Edge Coherence
"""

import numpy as np
import cv2


class VisualAnomalyAnalyzer:
    """Analyzes high-level visual anomalies and structural inconsistencies."""

    def analyze(self, image_path: str, **kwargs) -> dict:
        findings = []
        score = 0

        img_cv = cv2.imread(image_path)
        if img_cv is None:
            return {"score": 0, "findings": ["Could not load image"]}

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Edge coherence analysis
        edge_score, edge_findings = self._edge_coherence(gray, img_cv)
        findings.extend(edge_findings)
        score += edge_score
        
        # Color distribution mapping
        color_score, color_findings = self._color_distribution(img_cv)
        findings.extend(color_findings)
        score += color_score
        
        # Blurry background transition (common in diffusion depth-of-field)
        dof_score, dof_findings = self._depth_of_field_analysis(gray)
        findings.extend(dof_findings)
        score += dof_score

        # Compound bonus: multiple visual anomalies together
        ai_indicators = sum([
            dof_score >= 10,    # Shallow DoF triggered
            color_score >= 10,  # Color anomaly triggered
            edge_score >= 10,   # Edge anomaly triggered
        ])
        if ai_indicators >= 2:
            findings.append("Multiple visual anomalies detected — compound bonus applied")
            score += 15

        score = max(0, min(100, score))
        return {"score": score, "findings": findings}

    def _edge_coherence(self, gray: np.ndarray, img_cv: np.ndarray) -> tuple:
        findings = []
        score = 0

        try:
            # Canny edge detection
            edges = cv2.Canny(gray, 100, 200)
            
            # Edge density
            edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
            findings.append(f"Edge density: {edge_density:.4f}")
            
            # AI images often have either too few details (oversmooth) or hallucinated micro-details
            if edge_density < 0.02:
                findings.append("Unnaturally low edge density — possible AI smoothing")
                score += 15
            elif edge_density > 0.15:
                findings.append("Unnaturally high edge density — possible AI detail hallucination")
                score += 15
                
            # Straight line detection (Hough) - AI often struggles with perfectly straight architectural lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
            
            if lines is not None:
                findings.append(f"Strong structural lines detected: {len(lines)}")
                # Many straight lines usually indicate human-made structures which AI often bends
            else:
                findings.append("No strong structural lines detected")

        except Exception as e:
            findings.append(f"Edge coherence error: {str(e)}")

        return score, findings
        
    def _color_distribution(self, img_cv: np.ndarray) -> tuple:
        findings = []
        score = 0
        
        try:
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            s_channel = hsv[:, :, 1]
            v_channel = hsv[:, :, 2]
            
            mean_saturation = np.mean(s_channel)
            findings.append(f"Mean saturation: {mean_saturation:.2f}")
            
            # Midjourney / DALL-E tend to be highly saturated and vibrant
            if mean_saturation > 160:
                findings.append("Extremely high color saturation — typical of AI models (e.g. Midjourney output)")
                score += 20
            elif mean_saturation > 130:
                findings.append("High color saturation — common in AI-generated images")
                score += 15
                
            # Contrast check
            std_value = np.std(v_channel)
            if std_value > 80:
                findings.append("Hyper-contrast detected — common in AI 'cinematic' prompts")
                score += 15
                
        except Exception as e:
            findings.append(f"Color distribution error: {str(e)}")
            
        return score, findings
        
    def _depth_of_field_analysis(self, gray: np.ndarray) -> tuple:
        findings = []
        score = 0
        
        try:
            # Calculate local variance (blur map)
            blur_map = cv2.GaussianBlur(gray, (15, 15), 0)
            diff = cv2.absdiff(gray, blur_map)
            
            # Normalize diff map
            norm_diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
            
            # Threshold to find sharp regions
            _, sharp_mask = cv2.threshold(norm_diff, 50, 255, cv2.THRESH_BINARY)
            
            sharp_ratio = np.sum(sharp_mask > 0) / (gray.shape[0] * gray.shape[1])
            findings.append(f"Sharp area ratio: {sharp_ratio:.4f}")
            
            # AI often fakes shallow depth of field very aggressively
            if sharp_ratio < 0.05:
                findings.append("Extremely shallow depth of field — aggressive background blur (strong AI indicator)")
                score += 30
            elif sharp_ratio < 0.1:
                findings.append("Very shallow depth of field — aggressive background blur (AI prompt characteristic)")
                score += 20
            elif sharp_ratio < 0.15:
                findings.append("Shallow depth of field — possibly AI-generated blur")
                score += 10
                
        except Exception as e:
            findings.append(f"DoF analysis error: {str(e)}")
            
        return score, findings
