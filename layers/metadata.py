"""
Layer 1 — Metadata Analysis
Modules: EXIF Inspector, XMP Deep Scan, Format Origin Check, C2PA Check
"""

import os
import struct
import xml.etree.ElementTree as ET

import exifread
from PIL import Image

CAMERA_FIELDS = [
    "Make", "Model", "ExposureTime", "FNumber", "ISOSpeedRatings",
    "FocalLength", "Flash", "MeteringMode", "WhiteBalance",
    "DigitalZoomRatio", "SceneCaptureType", "LensModel",
]

AI_GENERATOR_SIGNATURES = [
    "midjourney", "stable diffusion", "dall-e", "dalle", "gemini",
    "firefly", "leonardo", "ideogram", "flux", "playground",
    "craiyon", "nightcafe", "artbreeder", "deep dream",
]


class MetadataAnalyzer:
    """Analyzes image metadata for signs of AI generation."""

    def analyze(self, image_path: str, **kwargs) -> dict:
        findings = []
        score = 0
        original_filename = kwargs.get("original_filename", "")
        override = None

        try:
            # EXIF analysis
            exif_score, exif_findings = self._analyze_exif(image_path)
            findings.extend(exif_findings)
            score += exif_score

            # XMP analysis
            xmp_score, xmp_findings, xmp_override = self._analyze_xmp(image_path)
            findings.extend(xmp_findings)
            score += xmp_score
            if xmp_override:
                override = xmp_override

            # Format origin check
            fmt_score, fmt_findings = self._format_origin_check(image_path)
            findings.extend(fmt_findings)
            score += fmt_score

            # Filename analysis
            fn_score, fn_findings, fn_override = self._filename_analysis(
                original_filename or os.path.basename(image_path)
            )
            findings.extend(fn_findings)
            score += fn_score
            if fn_override:
                override = fn_override

        except Exception as e:
            findings.append(f"Metadata analysis error: {str(e)}")

        score = max(0, min(100, score))
        result = {"score": score, "findings": findings}
        if override:
            result["override"] = override
        return result

    def _analyze_exif(self, image_path: str) -> tuple:
        findings = []
        score = 0

        with open(image_path, "rb") as f:
            tags = exifread.process_file(f, details=False)

        if not tags:
            findings.append("No EXIF metadata found — typical of AI-generated images")
            score += 25
            return score, findings

        # Check for camera hardware fields
        camera_fields_found = []
        camera_fields_missing = []

        for field in CAMERA_FIELDS:
            exif_key = f"EXIF {field}"
            image_key = f"Image {field}"
            if exif_key in tags or image_key in tags:
                camera_fields_found.append(field)
            else:
                camera_fields_missing.append(field)

        if len(camera_fields_found) == 0:
            findings.append("No camera hardware fields in EXIF — not from a real camera")
            score += 25
        elif len(camera_fields_missing) > 4:
            findings.append(
                f"Missing {len(camera_fields_missing)}/{len(CAMERA_FIELDS)} "
                f"camera fields: {', '.join(camera_fields_missing)}"
            )
            score += 15
        else:
            findings.append(
                f"Found camera fields: {', '.join(camera_fields_found)} — "
                f"consistent with real camera"
            )
            score -= 10

        return score, findings

    def _analyze_xmp(self, image_path: str) -> tuple:
        findings = []
        score = 0
        override = None

        with open(image_path, "rb") as f:
            data = f.read()

        xmp_start = data.find(b"<x:xmpmeta")
        xmp_end = data.find(b"</x:xmpmeta>")

        if xmp_start == -1 or xmp_end == -1:
            findings.append("No XMP metadata block found")
            score += 5
            return score, findings, override

        xmp_data = data[xmp_start : xmp_end + 12].decode("utf-8", errors="replace")

        # Check for AI-related software
        xmp_lower = xmp_data.lower()
        for sig in AI_GENERATOR_SIGNATURES:
            if sig in xmp_lower:
                findings.append(f"XMP metadata contains AI generator signature: '{sig}'")
                score += 30
                override = {"force_verdict": True, "min_score": 85}
                break

        # Check creator tool
        if "CreatorTool" in xmp_data:
            try:
                ct_start = xmp_data.index('CreatorTool="') + len('CreatorTool="')
                ct_end = xmp_data.index('"', ct_start)
                creator_tool = xmp_data[ct_start:ct_end]
                findings.append(f"Creator tool: {creator_tool}")
            except ValueError:
                pass

        # Check for conversion history
        if "converted from image/png to image/jpeg" in xmp_lower:
            findings.append("Image was converted from PNG to JPEG — PNG is common for AI-generated images")
            score += 10

        # Check RawFileName for AI signatures
        if "RawFileName" in xmp_data:
            try:
                rf_start = xmp_data.index('RawFileName="') + len('RawFileName="')
                rf_end = xmp_data.index('"', rf_start)
                raw_filename = xmp_data[rf_start:rf_end]
                findings.append(f"Original file name in XMP: {raw_filename}")

                raw_lower = raw_filename.lower()
                for sig in AI_GENERATOR_SIGNATURES:
                    if sig in raw_lower:
                        findings.append(f"Original filename contains AI signature: '{sig}'")
                        score += 25
                        override = {"force_verdict": True, "min_score": 85}
                        break
            except ValueError:
                pass

        # Check for Firefly / generative AI tools
        if "firefly" in xmp_lower or "fill_method" in xmp_lower:
            findings.append("Adobe Firefly AI tool usage detected in XMP")
            score += 10

        return score, findings, override

    def _format_origin_check(self, image_path: str) -> tuple:
        findings = []
        score = 0

        try:
            img = Image.open(image_path)
            fmt = img.format
            findings.append(f"Image format: {fmt}")

            if fmt == "PNG":
                findings.append("PNG format — common for AI generators")
                score += 5

            # Check DPI
            dpi = img.info.get("dpi")
            if dpi:
                findings.append(f"DPI: {dpi}")
                if dpi == (72, 72) or dpi == (96, 96):
                    findings.append("Standard screen DPI — not typical of cameras")
                    score += 5

            # Check dimensions
            w, h = img.size
            findings.append(f"Dimensions: {w}x{h}")

        except Exception as e:
            findings.append(f"Format check error: {str(e)}")

        return score, findings

    def _filename_analysis(self, filename: str) -> tuple:
        findings = []
        score = 0
        override = None

        fn_lower = filename.lower()
        for sig in AI_GENERATOR_SIGNATURES:
            if sig in fn_lower:
                findings.append(f"Filename contains AI generator name: '{sig}'")
                score += 20
                override = {"force_verdict": True, "min_score": 80}
                break

        # Check for common AI image naming patterns
        if any(pattern in fn_lower for pattern in ["generated_image", "gen_image", "ai_image"]):
            findings.append("Filename follows AI image naming convention")
            score += 15

        return score, findings, override
