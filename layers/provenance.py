"""
Layer 5 — Digital Provenance & Watermark Detection
Modules: SynthID Detection, Invisible Watermark Scanner, Provenance Chain
"""

import numpy as np
import cv2


class ProvenanceAnalyzer:
    """Detects invisible watermarks and digital provenance signatures."""

    def analyze(self, image_path: str, **kwargs) -> dict:
        findings = []
        score = 0
        heatmap = None
        gan_fingerprint = False

        img_cv = cv2.imread(image_path)
        if img_cv is None:
            return {"score": 0, "findings": ["Could not load image"]}

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Invisible Watermark / Steganography check (LSB analysis)
        lsb_score, lsb_findings = self._lsb_analysis(gray)
        findings.extend(lsb_findings)
        score += lsb_score

        # Google SynthID / Spectral Watermark Check
        synth_score, synth_findings, synth_gan = self._spectral_watermark_check(gray)
        findings.extend(synth_findings)
        score += synth_score
        if synth_gan:
            gan_fingerprint = True

        score = max(0, min(100, score))
        
        return {
            "score": score,
            "findings": findings,
            "gan_fingerprint": gan_fingerprint
        }

    def _lsb_analysis(self, gray: np.ndarray) -> tuple:
        findings = []
        score = 0

        try:
            # Extract Least Significant Bit (LSB) plane
            lsb = gray & 1
            
            # Calculate LSB entropy
            hist, _ = np.histogram(lsb, bins=2, range=(0, 2), density=True)
            hist_nz = hist[hist > 0]
            entropy = -np.sum(hist_nz * np.log2(hist_nz))
            
            findings.append(f"LSB Entropy: {entropy:.4f}")
            
            # Completely random LSB (entropy ~ 1.0) can sometimes hide watermarks
            # but AI models often have very pure LSB plains
            if entropy < 0.1:
                findings.append("Unnaturally pure LSB plane — AI models often generate 'clean' 8-bit values without analog noise")
                score += 15
            elif entropy < 0.5:
                findings.append("Low LSB noise — unusual for camera sensors")
                score += 5
                
            # Check for grid patterns in LSB
            h, w = lsb.shape
            if h > 128 and w > 128:
                lsb_crop = lsb[:128, :128]
                if np.sum(lsb_crop) / (128*128) > 0.4 and np.sum(lsb_crop) / (128*128) < 0.6:
                    findings.append("Dense LSB usage — possible invisible watermark/steganography payload")
                    # No score increment as it could be noise, just noted

        except Exception as e:
            findings.append(f"LSB analysis error: {str(e)}")

        return score, findings

    def _spectral_watermark_check(self, gray: np.ndarray) -> tuple:
        findings = []
        score = 0
        gan_detected = False

        try:
            # FFT for checking spectral signatures (e.g. SynthID pattern)
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-10)
            
            h, w = magnitude_spectrum.shape
            cy, cx = h // 2, w // 2
            
            # Mask out the DC component and low frequencies
            y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
            mask = x**2 + y**2 <= 40**2
            magnitude_spectrum[mask] = 0
            
            # Find brightest points in remaining spectrum
            max_val = np.max(magnitude_spectrum)
            mean_val = np.mean(magnitude_spectrum[magnitude_spectrum > 0])
            std_val = np.std(magnitude_spectrum[magnitude_spectrum > 0])
            
            peaks = np.sum(magnitude_spectrum > mean_val + 4 * std_val)
            
            findings.append(f"High-frequency spectral peaks: {peaks}")
            
            # Invisible watermarks often inject highly localized frequency peaks
            if peaks > 10 and peaks < 100:
                findings.append("Anomalous spectral peaks detected — possible invisible AI watermark (e.g. Google SynthID)")
                score += 30
                gan_detected = True

        except Exception as e:
            findings.append(f"Spectral analysis error: {str(e)}")

        return score, findings, gan_detected
