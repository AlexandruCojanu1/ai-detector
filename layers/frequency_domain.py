"""
Layer 3 — Frequency Domain Analysis
Modules: FFT Spectral, DCT Coefficient, Wavelet Analysis
"""

import io
import base64
import numpy as np
import cv2
import pywt
from scipy import fftpack
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class FrequencyDomainAnalyzer:
    """Analyzes frequency domain characteristics for AI generation artifacts."""

    def analyze(self, image_path: str, **kwargs) -> dict:
        findings = []
        score = 0
        heatmap = None
        gan_fingerprint = False

        img_cv = cv2.imread(image_path)
        if img_cv is None:
            return {"score": 0, "findings": ["Could not load image"]}

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY).astype(np.float64)

        # FFT Analysis
        fft_score, fft_findings, fft_heatmap, fft_gan = self._fft_analysis(gray)
        findings.extend(fft_findings)
        score += fft_score
        if fft_heatmap:
            heatmap = fft_heatmap
        if fft_gan:
            gan_fingerprint = True

        # DCT Analysis
        dct_score, dct_findings = self._dct_analysis(gray)
        findings.extend(dct_findings)
        score += dct_score

        # Wavelet Analysis
        wav_score, wav_findings = self._wavelet_analysis(gray)
        findings.extend(wav_findings)
        score += wav_score

        score = max(0, min(100, score))
        
        result = {
            "score": score,
            "findings": findings,
            "gan_fingerprint": gan_fingerprint
        }
        if heatmap:
            result["heatmap"] = heatmap
            
        return result

    def _fft_analysis(self, gray: np.ndarray) -> tuple:
        findings = []
        score = 0
        gan_detected = False

        try:
            # 2D FFT
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.log1p(np.abs(f_shift))
            
            # Normalize
            magnitude_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-10)

            h, w = magnitude_norm.shape
            cy, cx = h // 2, w // 2
            max_radius = min(cy, cx)

            # Radial profile to detect GAN periodic artifacts
            radial_profile = []
            for r in range(1, max_radius, 2):
                mask = np.zeros_like(magnitude_norm, dtype=bool)
                y, x = np.ogrid[0:h, 0:w]
                dist = np.sqrt((y - cy)**2 + (x - cx)**2)
                mask[(dist >= r - 1) & (dist < r + 1)] = True
                radial_profile.append(np.mean(magnitude_norm[mask]))

            # Find peaks in radial profile (periodic artifacts)
            peaks = 0
            for i in range(1, len(radial_profile) - 1):
                if radial_profile[i] > radial_profile[i - 1] and radial_profile[i] > radial_profile[i + 1]:
                    if radial_profile[i] > np.mean(radial_profile) + 2 * np.std(radial_profile):
                        peaks += 1

            findings.append(f"FFT Radial peaks found: {peaks}")

            if peaks > 3:
                findings.append("Periodic artifacts detected in frequency domain (GAN Fingerprint)")
                score += 30
                gan_detected = True
            elif peaks > 1:
                findings.append("Minor frequency anomalies detected")
                score += 10

            # Generate spectrum heatmap
            heatmap = self._generate_spectrum_heatmap(magnitude_norm, "FFT Power Spectrum")

            return score, findings, heatmap, gan_detected

        except Exception as e:
            return 0, [f"FFT analysis error: {str(e)}"], None, False

    def _dct_analysis(self, gray: np.ndarray) -> tuple:
        findings = []
        score = 0

        try:
            # Block-wise DCT analysis
            h, w = gray.shape
            h_adj = h - (h % 8)
            w_adj = w - (w % 8)
            gray_adj = gray[:h_adj, :w_adj]
            
            # Perform 2D DCT on the whole image 
            dct = fftpack.dct(fftpack.dct(gray_adj.T, norm='ortho').T, norm='ortho')
            
            # Extract high frequency components
            high_freq_mask = np.zeros_like(dct)
            high_freq_mask[h_adj//2:, w_adj//2:] = 1
            high_freq_power = np.sum(np.abs(dct * high_freq_mask)) / (np.sum(np.abs(dct)) + 1e-10)
            
            findings.append(f"High frequency DCT power ratio: {high_freq_power:.4f}")
            
            # AI models often struggle with high-frequency details leading to lower power
            if high_freq_power < 0.05:
                findings.append("Very low high-frequency content — typical of AI generation")
                score += 15
            elif high_freq_power < 0.1:
                findings.append("Lower than normal high-frequency content")
                score += 5

        except Exception as e:
            findings.append(f"DCT analysis error: {str(e)}")

        return score, findings

    def _wavelet_analysis(self, gray: np.ndarray) -> tuple:
        findings = []
        score = 0

        try:
            # 2D Discrete Wavelet Transform
            coeffs2 = pywt.dwt2(gray, 'haar')
            LL, (LH, HL, HH) = coeffs2
            
            # Calculate energy in detail sub-bands
            lh_energy = np.sum(LH**2) / (np.sum(gray**2) + 1e-10)
            hl_energy = np.sum(HL**2) / (np.sum(gray**2) + 1e-10)
            hh_energy = np.sum(HH**2) / (np.sum(gray**2) + 1e-10)
            
            total_detail_energy = lh_energy + hl_energy + hh_energy
            
            findings.append(f"Wavelet detail energy: {total_detail_energy:.6f}")
            
            # AI models generate unnaturally smooth images lacking true high-frequency detail energy
            if total_detail_energy < 0.005:
                findings.append("Extremely low detail energy in wavelet sub-bands — AI indicator")
                score += 20
            elif total_detail_energy < 0.01:
                findings.append("Low detail energy in wavelet sub-bands")
                score += 10
                
            # Check for grid artifacts (mismatch between H and V energy)
            energy_ratio = min(lh_energy, hl_energy) / (max(lh_energy, hl_energy) + 1e-10)
            if energy_ratio < 0.5:
                findings.append("Asymmetric horizontal/vertical detail energy — possible grid artifacts")
                score += 10

        except Exception as e:
            findings.append(f"Wavelet analysis error: {str(e)}")

        return score, findings

    def _generate_spectrum_heatmap(self, data: np.ndarray, title: str) -> str:
        try:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(data, cmap="viridis")
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
