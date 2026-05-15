#!/usr/bin/env python3
"""
StemForge — API Server v2
Endpoints asincroni con job queue per generazione e stem splitting.
"""

import uuid
import asyncio
import shutil
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="StemForge API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS: dict[str, dict] = {}
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

executor = ThreadPoolExecutor(max_workers=2)


# ── Models ────────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str
    duration: int = 30
    backend: str = "stable-audio"
    split_stems: bool = True
    steps: int = 100


class SplitRequest(BaseModel):
    file_id: str


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


# ── Generate ──────────────────────────────────────────────────────────────────

def _run_generate(job_id: str, req: GenerateRequest):
    try:
        JOBS[job_id]["status"] = "generating"
        from multi_generator import generate

        out_dir = OUTPUT_DIR / job_id
        mix_path = generate(
            prompt=req.prompt,
            duration=req.duration,
            backend=req.backend,
            output_dir=out_dir,
            split_stems=req.split_stems,
            steps=req.steps,
        )

        stems = {}
        stems_dir = out_dir / "stems"
        if stems_dir.exists():
            JOBS[job_id]["status"] = "splitting"
            for f in stems_dir.glob("*.wav"):
                stems[f.stem] = f"/files/{job_id}/stems/{f.name}"

        JOBS[job_id].update({
            "status": "done",
            "full_mix": f"/files/{job_id}/full_mix.wav",
            "stems": stems,
        })

    except Exception as e:
        JOBS[job_id].update({"status": "error", "error": str(e)})


@app.post("/generate")
async def generate_music(req: GenerateRequest):
    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {"status": "pending", "prompt": req.prompt}
    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, _run_generate, job_id, req)
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS[job_id]


# ── Stem Split ────────────────────────────────────────────────────────────────

def _run_split(job_id: str, audio_path: Path):
    try:
        JOBS[job_id]["status"] = "splitting"
        from stem_splitter import separate

        stems_dir = OUTPUT_DIR / job_id / "stems"
        separate(audio_path, stems_dir)

        stems = {}
        for f in stems_dir.glob("*.wav"):
            stems[f.stem] = f"/files/{job_id}/stems/{f.name}"

        JOBS[job_id].update({"status": "done", "stems": stems})

    except Exception as e:
        JOBS[job_id].update({"status": "error", "error": str(e)})


@app.post("/split")
async def split_audio(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())[:8]
    upload_path = UPLOAD_DIR / f"{job_id}_{file.filename}"

    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    JOBS[job_id] = {"status": "pending", "filename": file.filename}
    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, _run_split, job_id, upload_path)
    return {"job_id": job_id}


# ── File serving ──────────────────────────────────────────────────────────────

@app.get("/files/{job_id}/{filename}")
def serve_file(job_id: str, filename: str):
    path = OUTPUT_DIR / job_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(path), media_type="audio/wav")


@app.get("/files/{job_id}/stems/{stem_name}")
def serve_stem(job_id: str, stem_name: str):
    path = OUTPUT_DIR / job_id / "stems" / stem_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Stem not found")
    return FileResponse(str(path), media_type="audio/wav")


# ── Fine-tune ─────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    dataset_dir: str
    genre: str = "custom"
    epochs: int = 10
    lr: float = 1e-4
    batch_size: int = 4


def _run_train(job_id: str, req: TrainRequest):
    try:
        JOBS[job_id]["status"] = "training"
        from trainer import train_lora

        checkpoint_path = train_lora(
            dataset_dir=Path(req.dataset_dir),
            genre=req.genre,
            epochs=req.epochs,
            lr=req.lr,
            batch_size=req.batch_size,
            output_dir=Path("checkpoints") / req.genre,
        )
        JOBS[job_id].update({
            "status": "done",
            "checkpoint": str(checkpoint_path),
        })
    except Exception as e:
        JOBS[job_id].update({"status": "error", "error": str(e)})


@app.post("/train")
async def train_model(req: TrainRequest):
    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {"status": "pending", "genre": req.genre}
    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, _run_train, job_id, req)
    return {"job_id": job_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=58432, log_level="info")
