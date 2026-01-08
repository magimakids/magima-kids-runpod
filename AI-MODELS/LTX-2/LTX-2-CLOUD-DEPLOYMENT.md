# LTX-2 Cloud Deployment Guide for Magima Kids

A comprehensive guide for deploying the LTX-2 video generation model on cloud servers for Magima AI systems.

---

## Table of Contents
1. [Overview](#1-overview)
2. [Speed Priority Guide](#2-speed-priority-guide) ⚡ **START HERE**
3. [Hardware Requirements](#3-hardware-requirements)
4. [Cloud Provider Recommendations](#4-cloud-provider-recommendations)
5. [Model Selection Guide](#5-model-selection-guide)
6. [Step-by-Step Cloud Setup](#6-step-by-step-cloud-setup)
7. [Installation](#7-installation)
8. [Running Inference](#8-running-inference)
9. [API Setup for Magima Apps](#9-api-setup-for-magima-apps)
10. [Cost Optimization](#10-cost-optimization)
11. [COPPA & Privacy Compliance](#11-coppa--privacy-compliance)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Overview

### What is LTX-2?

LTX-2 is a DiT-based (Diffusion Transformer) audio-video foundation model developed by Lightricks. It generates synchronized video and audio within a single model.

**Key Capabilities:**
- Text-to-video generation
- Image-to-video generation
- Video-to-video transformation
- Audio generation (text-to-audio, video-to-audio)
- Multi-modal generation (text + image + audio to video)
- Up to 4K resolution, 50 FPS, 20 seconds per clip
- Multi-language support: English, German, Spanish, French, Japanese, Korean, Chinese, Italian, Portuguese

### Model Variants

| Model | File Size | VRAM Required | Notes |
|-------|-----------|---------------|-------|
| ltx-2-19b-dev | 40 GB | 48GB+ | Full model, fully trainable |
| ltx-2-19b-dev-fp8 | 25 GB | ~24GB | FP8 quantized, 2x faster |
| ltx-2-19b-dev-fp4 | 19 GB | ~16GB | NVFP4 quantized, lowest VRAM |
| ltx-2-19b-distilled | 40 GB | 48GB+ | 8-step inference, CFG=1 |
| **ltx-2-19b-distilled-fp8** | **25 GB** | **~16-20GB** | **RECOMMENDED for 720p** |
| ltx-2-spatial-upscaler | 950 MB | +2GB | 2x resolution upscaling |
| ltx-2-temporal-upscaler | 250 MB | +1GB | 2x FPS upscaling |

### License Considerations

LTX-2 uses the **LTX-2 Community License Agreement**:
- Free for commercial use if your organization's revenue is **under $10,000,000 annually**
- Contact Lightricks for enterprise licensing if above this threshold
- Full license: https://github.com/Lightricks/LTX-2/blob/main/LICENSE

---

## 2. Speed Priority Guide

> **For Magima Kids: Speed matters!** Kids have short attention spans. Every second counts. This section shows you how to get the fastest generation without overspending.

### Speed Benchmark Summary

| Configuration | GPU | 4-sec 720p Video | Cost/hr |
|--------------|-----|------------------|---------|
| 🐢 Slowest | A4000 + Full Model | ~90 seconds | $0.35 |
| 🚶 Moderate | A4000 + Distilled FP8 | ~35 seconds | $0.35 |
| 🏃 Fast | A10 + Distilled FP8 | ~18 seconds | $0.44 |
| ⚡ **Fastest Budget** | **RTX 4090 + Distilled FP8** | **~12 seconds** | **$0.35-0.69** |
| 🚀 Fastest (overkill) | A100 + Distilled FP8 | ~8 seconds | $1.64 |

### The Speed Formula

```
Speed = (Distilled Model) + (FP8 Quantization) + (Right GPU) + (Optimal Settings)
```

### Maximum Speed Configuration

**Model:** `ltx-2-19b-distilled-fp8`
- Distilled = 8 steps instead of 20-50 (3-5x faster)
- FP8 = 2x faster than BF16

**GPU:** RTX 4090 (24GB) on TensorDock @ $0.35/hr
- Best speed-per-dollar ratio
- Consumer architecture optimized for inference

**Settings for Fastest 720p:**
```bash
python -m ltx_pipelines.inference \
    --model_path ./models/ltx-2-19b-distilled-fp8.safetensors \
    --prompt "Your prompt" \
    --width 1280 \
    --height 736 \
    --num_frames 25 \                 # 3 seconds (shorter = faster)
    --num_inference_steps 8 \         # Distilled model uses 8
    --guidance_scale 1.0 \            # Distilled uses CFG=1
    --output_path ./output.mp4
```

### Speed vs Quality Trade-offs

| Setting | Faster | Slower | Impact on Quality |
|---------|--------|--------|-------------------|
| Resolution | 540p | 1080p | Lower = faster, minimal quality loss for kids content |
| Frame count | 17 (2 sec) | 49 (6 sec) | Shorter clips = faster |
| Inference steps | 8 | 20+ | Distilled @ 8 is optimized |
| Model | Distilled FP8 | Full BF16 | Distilled is 90%+ quality |

### Real-World Speed Targets for Kids Apps

| Use Case | Target Time | Recommended Setup |
|----------|-------------|-------------------|
| Quick reaction video | < 15 sec | RTX 4090 + 540p + 17 frames |
| Story segment | < 25 sec | RTX 4090 + 720p + 25 frames |
| Pre-generated library | < 45 sec | A10 + 720p + 33 frames |

### Speed Tips

1. **Batch during off-hours:** Pre-generate common videos at night
2. **Use shorter clips:** Kids prefer 2-4 second clips anyway
3. **540p is fine:** On phone screens, 540p looks great
4. **Warm up the model:** First inference is slow, subsequent ones are faster
5. **Keep GPU warm:** Don't let pod sleep between requests

### Warm-Up Script (Run Once After Boot)

```python
# warmup.py - Run this when server starts
import subprocess

# Generate a tiny test video to warm up GPU/model
subprocess.run([
    "python", "-m", "ltx_pipelines.inference",
    "--model_path", "./models/ltx-2-19b-distilled-fp8.safetensors",
    "--prompt", "test",
    "--width", "256",
    "--height", "256",
    "--num_frames", "9",
    "--output_path", "/tmp/warmup.mp4"
])
print("Model warmed up and ready!")
```

---

## 3. Hardware Requirements

### For 720p HD Video Generation (Recommended for Magima)

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| GPU | RTX A4000 (16GB) | A10 (24GB) |
| VRAM | 16GB | 24GB |
| System RAM | 32GB | 64GB |
| Storage | 50GB SSD | 100GB NVMe SSD |
| CUDA | 12.7+ | 12.7+ |
| Python | 3.12+ | 3.12+ |
| PyTorch | 2.7+ | 2.7+ |

### VRAM Usage by Resolution

Using `ltx-2-19b-distilled-fp8`:

| Resolution | VRAM Usage | Generation Speed |
|------------|------------|------------------|
| 540p (960x540) | ~12GB | ~15 sec/clip |
| 720p (1280x720) | ~16GB | ~25 sec/clip |
| 1080p (1920x1080) | ~24GB | ~45 sec/clip |

---

## 4. Cloud Provider Recommendations

### Budget-Friendly Providers (Recommended)

#### RunPod (Top Recommendation)
- **Website:** https://runpod.io
- **Why:** Easy setup, persistent storage, good GPU selection
- **Pricing:**

| GPU | VRAM | On-Demand | Spot |
|-----|------|-----------|------|
| RTX A4000 | 16GB | $0.35/hr | $0.19/hr |
| A10 | 24GB | $0.44/hr | $0.28/hr |
| RTX 4090 | 24GB | $0.69/hr | $0.34/hr |
| A100 40GB | 40GB | $1.64/hr | $0.89/hr |

**Monthly Cost Estimate (8 hours/day usage):**
- RTX A4000: ~$84/month (on-demand) or ~$46/month (spot)
- A10: ~$106/month (on-demand) or ~$67/month (spot)

#### TensorDock
- **Website:** https://tensordock.com
- **Pricing:**

| GPU | VRAM | Price/hr |
|-----|------|----------|
| RTX 4090 | 24GB | $0.35/hr |
| A100 40GB | 40GB | $0.75/hr |
| H100 SXM5 | 80GB | $2.25/hr |

#### Thunder Compute
- **Website:** https://thundercompute.com
- **Pricing:**

| GPU | VRAM | Price/hr |
|-----|------|----------|
| T4 | 16GB | $0.29/hr |
| A100 40GB | 40GB | $0.66/hr |

### Provider Comparison Summary

| Provider | Best For | Setup Ease | Min GPU |
|----------|----------|------------|---------|
| **RunPod** | Beginners, flexibility | Easy | A4000 |
| TensorDock | Budget RTX 4090 | Medium | 4090 |
| Thunder | Cheapest A100 | Medium | T4 |

---

## 5. Model Selection Guide

### For Magima Kids (720p, Budget)

**Recommended Model:** `ltx-2-19b-distilled-fp8`

**Why this model?**
1. **Distilled** = Only 8 inference steps (vs 20-50 for full model) = 3-5x faster
2. **FP8** = 40% less VRAM, 2x faster than BF16
3. **25GB file** = Fits on budget GPUs with room for processing
4. **Quality** = Minimal quality loss for 720p output

### Model Decision Tree

```
Need to train/fine-tune?
├── Yes → ltx-2-19b-dev (full BF16 model)
└── No → Need highest quality?
         ├── Yes → ltx-2-19b-dev-fp8
         └── No → ltx-2-19b-distilled-fp8 (FASTEST)
                  └── Very limited VRAM (<16GB)?
                       └── ltx-2-19b-dev-fp4
```

---

## 6. Step-by-Step Cloud Setup

### Option A: RunPod Setup (Recommended)

#### Step 1: Create Account
1. Go to https://runpod.io
2. Sign up with email or GitHub
3. Add payment method (credit card or crypto)
4. Add credits ($25-50 to start)

#### Step 2: Create a Pod
1. Click **"+ GPU Pod"**
2. Select GPU: **A10 24GB** (best value for 720p)
3. Select template: **RunPod PyTorch 2.4**
4. Configure storage:
   - Container Disk: 20GB (temporary)
   - **Volume Disk: 50GB** (persistent - stores your model)
5. Click **"Deploy"**

#### Step 3: Connect to Pod
1. Wait for pod to start (1-2 minutes)
2. Click **"Connect"** → **"Start Web Terminal"**
3. Or use SSH: `ssh root@<pod-ip> -p <port> -i ~/.ssh/id_rsa`

#### Step 4: Verify GPU
```bash
nvidia-smi
```
Should show your A10 with 24GB VRAM.

---

## 7. Installation

### Lightweight Installation (Recommended)

This approach downloads only what you need (~30GB total instead of 400GB+).

#### Step 1: Install System Dependencies
```bash
# Update system
apt update && apt install -y git-lfs python3.12 python3.12-venv

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

#### Step 2: Clone Repository (Code Only)
```bash
# Clone WITHOUT large model files
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Lightricks/LTX-2.git
cd LTX-2
```

#### Step 3: Install Python Dependencies
```bash
# Create virtual environment and install deps
uv sync
source .venv/bin/activate
```

#### Step 4: Download Only the Model You Need
```bash
# Install huggingface CLI
pip install huggingface-hub

# Download ONLY the distilled FP8 model (~25GB)
mkdir -p models
huggingface-cli download Lightricks/LTX-2 \
    ltx-2-19b-distilled-fp8.safetensors \
    --local-dir ./models \
    --local-dir-use-symlinks False
```

#### Step 5: Verify Installation
```bash
# Check model file exists
ls -lh models/ltx-2-19b-distilled-fp8.safetensors
# Should show ~25GB file

# Test Python imports
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### Storage Summary

| Component | Size |
|-----------|------|
| LTX-2 code repo | ~500MB |
| Python dependencies | ~3GB |
| ltx-2-19b-distilled-fp8 model | ~25GB |
| **Total** | **~30GB** |

---

## 8. Running Inference

### Important Constraints

Before generating video, remember these rules:
- **Width & Height:** Must be divisible by 32 (e.g., 1280x720 ✓, 1290x720 ✗)
- **Frame Count:** Must be divisible by 8, plus 1 (e.g., 9, 17, 25, 33, 41 frames)

### Common Resolutions

| Name | Dimensions | Aspect Ratio |
|------|------------|--------------|
| 540p | 960 x 544 | 16:9 |
| 720p | 1280 x 736 | ~16:9 |
| 1080p | 1920 x 1088 | ~16:9 |

### Basic Inference Command

```bash
cd LTX-2

# Activate environment
source .venv/bin/activate

# Run inference
python -m ltx_pipelines.inference \
    --model_path ./models/ltx-2-19b-distilled-fp8.safetensors \
    --prompt "A friendly cartoon character waving hello, colorful, kid-friendly animation style" \
    --width 1280 \
    --height 736 \
    --num_frames 33 \
    --output_path ./output/video.mp4 \
    --num_inference_steps 8 \
    --guidance_scale 1.0
```

### Parameters Explained

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--width` | 1280 | Video width (must be /32) |
| `--height` | 736 | Video height (must be /32) |
| `--num_frames` | 33 | Frame count (must be /8 + 1) |
| `--num_inference_steps` | 8 | Denoising steps (distilled=8) |
| `--guidance_scale` | 1.0 | CFG scale (distilled=1.0) |
| `--seed` | 42 | Random seed for reproducibility |

### Image-to-Video Generation

```bash
python -m ltx_pipelines.inference \
    --model_path ./models/ltx-2-19b-distilled-fp8.safetensors \
    --image_path ./input/character.png \
    --prompt "The character starts dancing happily" \
    --width 1280 \
    --height 736 \
    --num_frames 33 \
    --output_path ./output/video.mp4
```

---

## 9. API Setup for Magima Apps

### Simple FastAPI Wrapper

Create `api_server.py`:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import uuid
import os
from pathlib import Path

app = FastAPI(title="Magima LTX-2 Video API")

# Configuration
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_PATH = "./models/ltx-2-19b-distilled-fp8.safetensors"

class VideoRequest(BaseModel):
    prompt: str
    width: int = 1280
    height: int = 736
    num_frames: int = 33
    seed: int = None

class VideoResponse(BaseModel):
    video_id: str
    status: str
    video_url: str = None

@app.post("/generate", response_model=VideoResponse)
async def generate_video(request: VideoRequest):
    # Validate dimensions
    if request.width % 32 != 0 or request.height % 32 != 0:
        raise HTTPException(400, "Width and height must be divisible by 32")
    if (request.num_frames - 1) % 8 != 0:
        raise HTTPException(400, "num_frames must be divisible by 8, plus 1")

    # Generate unique ID
    video_id = str(uuid.uuid4())[:8]
    output_path = OUTPUT_DIR / f"{video_id}.mp4"

    # Build command
    cmd = [
        "python", "-m", "ltx_pipelines.inference",
        "--model_path", MODEL_PATH,
        "--prompt", request.prompt,
        "--width", str(request.width),
        "--height", str(request.height),
        "--num_frames", str(request.num_frames),
        "--output_path", str(output_path),
        "--num_inference_steps", "8",
        "--guidance_scale", "1.0"
    ]

    if request.seed:
        cmd.extend(["--seed", str(request.seed)])

    # Run generation
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return VideoResponse(
            video_id=video_id,
            status="completed",
            video_url=f"/videos/{video_id}.mp4"
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(500, f"Generation failed: {e.stderr.decode()}")

@app.get("/videos/{video_id}.mp4")
async def get_video(video_id: str):
    from fastapi.responses import FileResponse
    video_path = OUTPUT_DIR / f"{video_id}.mp4"
    if not video_path.exists():
        raise HTTPException(404, "Video not found")
    return FileResponse(video_path, media_type="video/mp4")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": os.path.exists(MODEL_PATH)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Running the API Server

```bash
# Install FastAPI dependencies
pip install fastapi uvicorn python-multipart

# Run server
python api_server.py
```

### Testing the API

```bash
# Health check
curl http://localhost:8000/health

# Generate video
curl -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "A cute animated cat playing with a ball, colorful cartoon style",
        "width": 1280,
        "height": 736,
        "num_frames": 33
    }'
```

### Connecting from Magima Apps

```swift
// Swift example for iOS
struct VideoRequest: Codable {
    let prompt: String
    let width: Int
    let height: Int
    let numFrames: Int

    enum CodingKeys: String, CodingKey {
        case prompt, width, height
        case numFrames = "num_frames"
    }
}

func generateVideo(prompt: String) async throws -> URL {
    let url = URL(string: "https://your-cloud-server.com/generate")!
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")

    let body = VideoRequest(prompt: prompt, width: 1280, height: 736, numFrames: 33)
    request.httpBody = try JSONEncoder().encode(body)

    let (data, _) = try await URLSession.shared.data(for: request)
    let response = try JSONDecoder().decode(VideoResponse.self, from: data)

    return URL(string: response.videoUrl)!
}
```

---

## 10. Cost Optimization

### Strategy 1: Use Spot Instances

Spot instances are 40-60% cheaper but can be interrupted.

**RunPod Spot Pricing:**
- A10: $0.28/hr (vs $0.44 on-demand) = 36% savings
- A4000: $0.19/hr (vs $0.35 on-demand) = 46% savings

**Best for:** Pre-generating content library (can restart if interrupted)

### Strategy 2: Auto-Shutdown

Set up automatic shutdown when idle:

```bash
# Add to crontab
crontab -e

# Shutdown if no API requests in 30 minutes
*/5 * * * * /path/to/check_idle.sh
```

`check_idle.sh`:
```bash
#!/bin/bash
LAST_REQUEST=$(stat -c %Y /tmp/last_api_request 2>/dev/null || echo 0)
NOW=$(date +%s)
IDLE_TIME=$((NOW - LAST_REQUEST))

if [ $IDLE_TIME -gt 1800 ]; then  # 30 minutes
    echo "Idle for 30+ minutes, shutting down..."
    # For RunPod: use their API to stop pod
    curl -X POST "https://api.runpod.io/v2/pod/$POD_ID/stop"
fi
```

### Strategy 3: Pre-generate Content Library

For predictable content, generate during off-peak hours:

```python
# batch_generate.py
prompts = [
    "Happy birthday animation with confetti",
    "Thank you message with hearts",
    "Good morning sunshine animation",
    # ... add more templates
]

for i, prompt in enumerate(prompts):
    os.system(f'''
        python -m ltx_pipelines.inference \
            --prompt "{prompt}" \
            --output_path ./library/template_{i}.mp4 \
            # ... other params
    ''')
```

### Monthly Cost Estimates

| Usage Pattern | GPU | Hours/Day | Monthly Cost |
|---------------|-----|-----------|--------------|
| Light (pre-gen only) | A4000 Spot | 2 | ~$12 |
| Moderate | A4000 On-demand | 4 | ~$42 |
| Heavy | A10 On-demand | 8 | ~$106 |
| Production | A10 + Spot mix | 12 | ~$130 |

---

## 11. COPPA & Privacy Compliance

### Why Self-Hosted Matters for Kids Apps

By running LTX-2 on your own cloud server:

1. **No data sent to third-party AI providers**
   - Prompts stay on your server
   - Generated content stays under your control
   - No external API logging of children's requests

2. **Full data control**
   - You decide what to log and retain
   - Can implement strict data deletion policies
   - GDPR/COPPA compliant by design

3. **Content moderation**
   - Pre-screen all prompts before generation
   - Block inappropriate content generation
   - Audit trail under your control

### Recommended Privacy Practices

#### Input Sanitization
```python
BLOCKED_TERMS = ["violence", "scary", "adult", ...]  # Expand this list

def sanitize_prompt(prompt: str) -> str:
    prompt_lower = prompt.lower()
    for term in BLOCKED_TERMS:
        if term in prompt_lower:
            raise ValueError(f"Inappropriate content detected")
    return prompt
```

#### Minimal Logging
```python
# Log only what's necessary
import logging

logger = logging.getLogger("magima")

def log_request(video_id: str, prompt_hash: str):
    # Don't log the actual prompt, just a hash
    logger.info(f"Generated video {video_id}, prompt_hash={prompt_hash}")
```

#### Data Retention
```bash
# Cron job to delete videos older than 24 hours
0 * * * * find /outputs -name "*.mp4" -mtime +1 -delete
```

### COPPA Compliance Checklist

- [ ] No collection of personal information from children
- [ ] Parental consent mechanism if collecting any data
- [ ] Clear privacy policy describing AI video generation
- [ ] Data minimization - don't store prompts containing child info
- [ ] Secure data transmission (HTTPS)
- [ ] Regular security audits

---

## 12. Troubleshooting

### Common Issues

#### "CUDA out of memory"
```
Solution 1: Use FP4 model instead
huggingface-cli download Lightricks/LTX-2 ltx-2-19b-dev-fp4.safetensors ...

Solution 2: Reduce resolution
--width 960 --height 544

Solution 3: Reduce frame count
--num_frames 17  # instead of 33
```

#### "Model file not found"
```bash
# Verify model exists
ls -la ./models/

# Re-download if needed
huggingface-cli download Lightricks/LTX-2 ltx-2-19b-distilled-fp8.safetensors --local-dir ./models
```

#### "Width/height not divisible by 32"
```
Common valid sizes:
540p: 960 x 544
720p: 1280 x 736
1080p: 1920 x 1088

NOT valid: 1280 x 720 (720 % 32 = 16, not 0)
```

#### "Slow generation speed"
```bash
# Check GPU utilization
watch -n 1 nvidia-smi

# If GPU < 90% utilized, check:
1. Using distilled model? (8 steps vs 20-50)
2. FP8 quantization enabled?
3. No CPU offloading happening?
```

#### API not accessible from outside
```bash
# Ensure firewall allows port 8000
ufw allow 8000

# For RunPod: Check "Expose HTTP Ports" in pod settings
# Add port 8000 to exposed ports list
```

---

## Quick Reference Card

### Essential Commands

```bash
# Start environment
cd LTX-2 && source .venv/bin/activate

# Generate 720p video (4 seconds, 24fps)
python -m ltx_pipelines.inference \
    --model_path ./models/ltx-2-19b-distilled-fp8.safetensors \
    --prompt "Your prompt here" \
    --width 1280 --height 736 --num_frames 33 \
    --output_path ./output.mp4

# Start API server
python api_server.py

# Check GPU status
nvidia-smi
```

### Valid Frame Counts
9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97 ...

### Formula: (8 × n) + 1

---

## Resources

- **LTX-2 GitHub:** https://github.com/Lightricks/LTX-2
- **Model Weights:** https://huggingface.co/Lightricks/LTX-2
- **Prompting Guide:** https://ltx.video/blog/how-to-prompt-for-ltx-2
- **RunPod Docs:** https://docs.runpod.io
- **ComfyUI Integration:** https://blog.comfy.org/p/ltx-2-open-source-audio-video-ai

---

*Document created for Magima AI Cloud Systems - January 2026*
