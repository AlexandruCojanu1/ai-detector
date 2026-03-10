"""
AI Image Detector — Professional Forensic Analysis
Main application entry point.
"""

import os
import shutil
import tempfile

from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core.orchestrator import Orchestrator

app = FastAPI(title="AI Image Detector")

app.mount("/static", StaticFiles(directory="frontend"), name="static")

orchestrator = Orchestrator()


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("frontend/index.html", "r") as f:
        return f.read()


@app.post("/analyze")
async def analyze_image(file: UploadFile):
    """Analyze an uploaded image for AI generation indicators."""
    try:
        suffix = os.path.splitext(file.filename)[1] or ".jpg"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        try:
            result = orchestrator.analyze(tmp_path, original_filename=file.filename)
            return JSONResponse(content=result)
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
