"""
Layer 2 — Pixel-Level Forensics
Modules: ELA, Noise Fingerprint, Clone Detection, Micro-Texture Analysis
"""

import io
import base64
import numpy as np
import cv2
from PIL import Image
from scipy import ndimage
from skimage.feature import local_binary_pattern
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class PixelForensicsAnalyzer:
    """Analyzes pixel-level forensic indicators of AI generation."""

    def analyze(self, image_path: str, **kwargs) -> dict:
        findings = []
        score = 0
        heatmap = None

        img_cv = cv2.imread(image_path)
        if img_cv is None:
            return {"score": 0, "findings": ["Could not load image for pixel analysis"]}

        # Error Level Analysis
        ela_score, ela_findings, ela_heatmap = self._error_level_analysis(image_path, img_cv)
        findings.extend(ela_findings)
        score += ela_score
        if ela_heatmap:
            heatmap = ela_heatmap

        # Noise fingerprint
        noise_score, noise_findings = self._noise_fingerprint(img_cv)
        findings.extend(noise_findings)
        score += noise_score

        # Clone detection
        clone_score, clone_findings = self._clone_detection(img_cv)
        findings.extend(clone_findings)
        score += clone_score

        # Micro-texture analysis
        texture_score, texture_findings = self._micro_texture(img_cv)
        findings.extend(texture_findings)
        score += texture_score

        score = max(0, min(100, score))

        result = {"score": score, "findings": findings}
        if heatmap:
            result["heatmap"] = heatmap
        return result

    def _error_level_analysis(self, image_path: str, img_cv: np.ndarray) -> tuple:
        """ELA: re-compress at known quality and measure error distribution."""
        findings = []
        score = 0

        try:
            img_pil = Image.open(image_path).convert("RGB")
            buffer = io.BytesIO()
            img_pil.save(buffer, format="JPEG", quality=95)
            buffer.seek(0)
            resaved = np.array(Image.open(buffer))

            original = np.array(img_pil)

            # Compute ELA
            ela = np.abs(original.astype(np.float64) - resaved.astype(np.float64))

            # Scale for visualization
            ela_scaled = (ela * 20).clip(0, 255).astype(np.uint8)

            # Statistics
            ela_gray = np.mean(ela, axis=2)
            mean_ela = np.mean(ela_gray)
            std_ela = np.std(ela_gray)
            max_ela = np.max(ela_gray)

            # Coefficient of variation
            cv = std_ela / (mean_ela + 1e-10)

            findings.append(
                f"ELA stats — mean: {mean_ela:.2f}, std: {std_ela:.2f}, "
                f"max: {max_ela:.2f}, CV: {cv:.2f}"
            )

            # Evaluate uniformity
            if cv < 0.8:
                findings.append("ELA distribution is abnormally uniform — AI indicator")
                score += 25
            elif cv < 1.2:
                findings.append("ELA distribution is somewhat uniform")
                score += 10
            else:
                findings.append("ELA distribution shows natural variation")
                score -= 5

            # Block-level analysis
            h, w = ela_gray.shape
            block_size = 64
            block_stds = []

            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = ela_gray[y : y + block_size, x : x + block_size]
                    block_stds.append(np.std(block))

            if block_stds:
                block_std_variance = np.std(block_stds)
                findings.append(f"ELA block variance: {block_std_variance:.2f}")
                if block_std_variance < 2.0:
                    findings.append("Very uniform block-level ELA — strong AI indicator")
                    score += 15

            # Generate heatmap
            heatmap = self._generate_heatmap(ela_scaled, "Error Level Analysis")

            return score, findings, heatmap

        except Exception as e:
            return 0, [f"ELA analysis error: {str(e)}"], None

    def _noise_fingerprint(self, img_cv: np.ndarray) -> tuple:
        """Analyze sensor noise patterns."""
        findings = []
        score = 0

        try:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY).astype(np.float64)

            # Extract noise
            denoised = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = gray - denoised

            noise_std = np.std(noise)
            noise_mean = np.mean(np.abs(noise))

            findings.append(f"Noise level — std: {noise_std:.4f}, mean abs: {noise_mean:.4f}")

            if noise_std < 1.5:
                findings.append("Extremely low noise level — no sensor noise detected (AI indicator)")
                score += 20
            elif noise_std < 3.0:
                findings.append("Low noise level — possibly AI or heavy denoising")
                score += 10

            # Quadrant uniformity check
            h, w = gray.shape
            quadrants = [
                noise[: h // 2, : w // 2],
                noise[: h // 2, w // 2 :],
                noise[h // 2 :, : w // 2],
                noise[h // 2 :, w // 2 :],
            ]
            quad_stds = [np.std(q) for q in quadrants]
            quad_variance = np.std(quad_stds)

            findings.append(f"Noise uniformity across quadrants — variance: {quad_variance:.4f}")
            if quad_variance < 0.1:
                findings.append("Noise is perfectly uniform across image — AI indicator")
                score += 15
            elif quad_variance > 0.5:
                findings.append("Natural noise variation detected across quadrants")
                score -= 5

            # Spectral flatness
            noise_fft = np.abs(np.fft.fft2(noise))
            spectral_flat = np.exp(np.mean(np.log(noise_fft + 1e-10))) / (np.mean(noise_fft) + 1e-10)

            findings.append(f"Noise spectral flatness: {spectral_flat:.4f}")
            if spectral_flat > 0.8:
                findings.append("Noise spectrum is flat (white noise) — unusual for sensor noise")
                score += 10

        except Exception as e:
            findings.append(f"Noise analysis error: {str(e)}")

        return score, findings

    def _clone_detection(self, img_cv: np.ndarray) -> tuple:
        """Detect cloned/repeated regions."""
        findings = []
        score = 0

        try:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

            # Resize if too large
            h, w = gray.shape
            scale = 1.0
            if max(h, w) > 800:
                scale = 800 / max(h, w)
                gray = cv2.resize(gray, None, fx=scale, fy=scale)

            # ORB feature detection
            orb = cv2.ORB_create(nfeatures=1000)
            keypoints, descriptors = orb.detectAndCompute(gray, None)

            if descriptors is not None and len(descriptors) > 10:
                # BFMatcher
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                matches = bf.knnMatch(descriptors, descriptors, k=2)

                # Count close matches (excluding self-matches)
                repeated = 0
                for m_list in matches:
                    if len(m_list) >= 2:
                        m, n = m_list[0], m_list[1]
                        if m.queryIdx != m.trainIdx and m.distance < 30:
                            repeated += 1

                findings.append(f"Repeated pattern regions detected: {repeated}")
                if repeated > 50:
                    findings.append("High number of repeated patterns — possible AI texture repetition")
                    score += 15
            else:
                findings.append("Insufficient features for clone detection")

        except Exception as e:
            findings.append(f"Clone detection error: {str(e)}")

        return score, findings

    def _micro_texture(self, img_cv: np.ndarray) -> tuple:
        """Analyze micro-texture patterns using LBP."""
        findings = []
        score = 0

        try:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

            # Resize if too large
            h, w = gray.shape
            scale = 1.0
            if max(h, w) > 600:
                scale = 600 / max(h, w)
                gray = cv2.resize(gray, None, fx=scale, fy=scale)

            # LBP analysis
            radius = 1
            n_points = 8 * radius
            lbp = local_binary_pattern(gray, n_points, radius, method="uniform")

            # Histogram
            n_bins = int(lbp.max() + 1)
            hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)

            # Entropy
            hist_nonzero = hist[hist > 0]
            entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))

            findings.append(f"Texture entropy (LBP): {entropy:.4f}")
            if entropy < 2.0:
                findings.append("Very low texture entropy — overly smooth (AI indicator)")
                score += 20
            elif entropy < 2.5:
                findings.append("Low texture entropy — possibly AI smoothing")
                score += 10
            else:
                findings.append("Normal texture complexity")

            # Patch-level variance
            h2, w2 = gray.shape
            patch_size = 64
            patch_entropies = []
            for y in range(0, h2 - patch_size, patch_size):
                for x in range(0, w2 - patch_size, patch_size):
                    patch = gray[y : y + patch_size, x : x + patch_size]
                    p_lbp = local_binary_pattern(patch, n_points, radius, method="uniform")
                    p_hist, _ = np.histogram(p_lbp, bins=n_bins, range=(0, n_bins), density=True)
                    p_nz = p_hist[p_hist > 0]
                    p_ent = -np.sum(p_nz * np.log2(p_nz))
                    patch_entropies.append(p_ent)

            if patch_entropies:
                ent_var = np.std(patch_entropies)
                findings.append(f"Texture entropy variance across patches: {ent_var:.4f}")
                if ent_var < 0.3:
                    findings.append("Uniform texture across image — AI indicator")
                    score += 10

        except Exception as e:
            findings.append(f"Texture analysis error: {str(e)}")

        return score, findings

    def _generate_heatmap(self, data: np.ndarray, title: str) -> str:
        """Generate a base64-encoded heatmap image."""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            if len(data.shape) == 3:
                ax.imshow(data)
            else:
                ax.imshow(data, cmap="hot")
            ax.set_title(title, fontsize=12)
            ax.axis("off")
            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8")
        except Exception:
            return None
