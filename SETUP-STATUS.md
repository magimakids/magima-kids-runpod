# Magima Cloud AI - Setup Status

**Last Updated:** January 7, 2026

---

## Current Status: Setting Up RunPod MCP Integration

### What We're Building
- Cloud-based AI video generation for Magima Kids apps
- Using LTX-2 model for video generation
- RunPod for GPU cloud hosting
- COPPA-compliant self-hosted approach

---

## Completed Steps

### 1. Documentation Created
- **File:** `/Users/mr.daniel/Code/MAGIMA-CLOUD-AI/AI-MODELS/LTX-2/LTX-2-CLOUD-DEPLOYMENT.md`
- Comprehensive 700+ line guide covering:
  - Hardware requirements
  - Cloud provider comparison (RunPod recommended)
  - Speed optimization for kids (12-25 sec per video)
  - Installation commands
  - FastAPI wrapper for Magima apps
  - COPPA/privacy compliance
  - Cost optimization

### 2. RunPod Account Setup
- Account created at runpod.io
- ~$10 credit added
- API Key created: `rpa_M54T48AYB4RNJLUTRPUI2ML6FFWXMEAOD5NK4SW81fqfos`
  - **NOTE:** Regenerate this key later since it's in chat history!

### 3. RunPod Infrastructure Created
- **Network Volume:** `Magima-Kids-1-dev`
  - Size: 150 GB
  - Location: US-IL-1 (Illinois)
  - Cost: ~$10.50/month
  - Purpose: Persistent storage for models (shared across pods)

- **Pod Ready to Deploy:**
  - GPU: RTX 4090 (24GB VRAM)
  - Template: RunPod PyTorch 2.4.0
  - Location: US-IL-1 (same as volume)
  - Cost: $0.59/hr
  - Status: NOT YET STARTED (waiting on MCP setup)

### 4. RunPod MCP Integration (In Progress)
- **Config File:** `~/.claude/mcp.json`
- RunPod MCP server added with API key
- Package: `@runpod/mcp-server@latest`
- **Status:** Need to restart Claude Code to test if it works

---

## Architecture Decisions

### GPU Choice: RTX 4090
- 24GB VRAM - plenty for LTX-2 FP8 model
- ~15 seconds per 4-sec 720p video
- $0.59/hr - good speed/cost balance
- User also has a 4090 at home for local testing

### Model Choice: ltx-2-19b-distilled-fp8
- Distilled = 8 inference steps (5x faster than full model)
- FP8 = 40% less VRAM, 2x faster
- ~25GB file size
- Best balance of speed + quality for kids apps

### Scaling Strategy
- **Phase 1 (Now):** Single pod for development/testing
- **Phase 2 (Classes):** RunPod Serverless for auto-scaling
  - 5 students can generate videos simultaneously
  - Pay per inference (~$0.007 per video)
  - Auto-scales to zero when idle
- **Phase 3 (Production):**
  - Pre-generated content library (80%)
  - Serverless for on-demand (20%)
  - CDN for delivery

### Multi-Model Architecture (Future)
| Pod | Purpose | GPU | Cost |
|-----|---------|-----|------|
| Pod 1 | Video Gen (LTX-2) | RTX 4090 | $0.59/hr |
| Pod 2 | Image Tools (bg removal, eraser) | RTX 4000 Ada | $0.26/hr |
| Pod 3 | Other models (TBD) | L4 | $0.39/hr |

---

## Next Steps

### Immediate (Resume Here)
1. **Restart Claude Code** - MCP config was just updated
2. **Test RunPod MCP** - Check if `mcp__runpod` tools are available
3. **Start the pod** - Either via MCP or RunPod web UI
4. **Install LTX-2** - Commands ready in deployment doc

### LTX-2 Installation Commands (Once in Terminal)
```bash
# Go to persistent volume
cd /workspace

# Check GPU
nvidia-smi

# Install dependencies
apt update && apt install -y git-lfs
pip install huggingface-hub

# Clone LTX-2 (code only, no large files)
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Lightricks/LTX-2.git
cd LTX-2

# Install Python packages
pip install -e packages/ltx-video

# Download ONLY the FP8 model (~25GB)
mkdir -p models
huggingface-cli download Lightricks/LTX-2 \
    ltx-2-19b-distilled-fp8.safetensors \
    --local-dir ./models \
    --local-dir-use-symlinks False

# Test inference
python -m ltx_pipelines.inference \
    --model_path ./models/ltx-2-19b-distilled-fp8.safetensors \
    --prompt "A friendly cartoon character waving hello" \
    --width 1280 --height 736 --num_frames 33 \
    --output_path ./test_output.mp4
```

### After LTX-2 Works
1. Convert to RunPod Serverless for classroom use
2. Set up API endpoint for Magima apps
3. Test with local 4090 for comparison

---

## Key Files

| File | Purpose |
|------|---------|
| `/Users/mr.daniel/Code/MAGIMA-CLOUD-AI/AI-MODELS/LTX-2/LTX-2-CLOUD-DEPLOYMENT.md` | Full deployment documentation |
| `/Users/mr.daniel/Code/MAGIMA-CLOUD-AI/AI-MODELS/LTX-2/LTX-2/README.md` | Original LTX-2 model README |
| `~/.claude/mcp.json` | Claude MCP server config (includes RunPod) |
| This file | Setup status and resume point |

---

## RunPod MCP Config (for reference)

Location: `~/.claude/mcp.json`

```json
{
  "mcpServers": {
    "runpod": {
      "command": "npx",
      "args": ["-y", "@runpod/mcp-server@latest"],
      "env": {
        "RUNPOD_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

---

## Cost Summary

| Item | Cost |
|------|------|
| Network Volume (150GB) | $10.50/month |
| RTX 4090 Pod | $0.59/hour |
| Estimated testing today | $2-5 |
| Estimated monthly (8hr/day) | ~$150 |
| Serverless per video | ~$0.007 |

---

## Resume Prompt for New Claude Session

Copy this to continue:

```
We're setting up LTX-2 video generation for Magima Kids on RunPod.

Read /Users/mr.daniel/Code/MAGIMA-CLOUD-AI/SETUP-STATUS.md for full context.

Quick status:
- RunPod account ready, volume created (Magima-Kids-1-dev, 150GB, US-IL-1)
- RTX 4090 pod ready to deploy
- Just configured RunPod MCP in ~/.claude/mcp.json
- Need to test if mcp__runpod tools are now available

Try listing your available RunPod MCP tools, then let's start the pod and install LTX-2!
```

---

## Notes

- User is in California but using US-IL-1 datacenter (4090 availability)
- Speed is priority - kids have short attention spans
- Eventually want smaller/faster models for iterative creative process
- May run locally on user's 4090 for instant iteration
- COPPA compliance is important - self-hosted approach helps
