"""FastAPI server for video generation"""
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import uuid
from pathlib import Path

from src.models.ltx import LTXVideoGenerator

app = FastAPI(title="Magima Kids Video Generation API")

# Global generator (loaded once)
generator: LTXVideoGenerator = None

# Job storage (in production, use Redis or similar)
jobs = {}


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 704
    height: int = 480
    num_frames: int = 65
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    seed: Optional[int] = None


class JobStatus(BaseModel):
    job_id: str
    status: str
    video_path: Optional[str] = None
    error: Optional[str] = None


@app.on_event("startup")
async def startup():
    """Load model on startup"""
    global generator
    generator = LTXVideoGenerator()
    generator.load()


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": generator is not None}


@app.post("/generate", response_model=JobStatus)
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    """Start video generation job"""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "video_path": None, "error": None}

    background_tasks.add_task(run_generation, job_id, request)

    return JobStatus(job_id=job_id, status="processing")


@app.get("/job/{job_id}", response_model=JobStatus)
async def get_job(job_id: str):
    """Get job status"""
    if job_id not in jobs:
        return JobStatus(job_id=job_id, status="not_found")

    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        video_path=job["video_path"],
        error=job["error"],
    )


async def run_generation(job_id: str, request: GenerateRequest):
    """Run generation in background"""
    try:
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        video_path = output_dir / f"{job_id}.mp4"

        frames = generator.generate(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_frames=request.num_frames,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
        )

        generator.save_video(frames, str(video_path))

        jobs[job_id] = {
            "status": "completed",
            "video_path": str(video_path),
            "error": None,
        }
    except Exception as e:
        jobs[job_id] = {
            "status": "failed",
            "video_path": None,
            "error": str(e),
        }


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
