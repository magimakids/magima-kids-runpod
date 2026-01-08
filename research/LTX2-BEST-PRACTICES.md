# LTX-2 Best Practices for High-Quality Video Generation

**Last Updated:** January 2026

## Model Selection

### Available Models

| Model | File Size | VRAM | Steps | Best For |
|-------|-----------|------|-------|----------|
| **ltx-2-19b-dev.safetensors** | 43GB | 48GB+ | Variable (20-50) | **Highest quality** |
| ltx-2-19b-dev-fp8.safetensors | 27GB | ~24GB | Variable | Speed + quality balance |
| ltx-2-19b-distilled.safetensors | 43GB | 48GB+ | Locked to 8 | Fast, lower quality |
| ltx-2-19b-distilled-fp8.safetensors | 27GB | ~16-20GB | Locked to 8 | Budget GPUs |

### Decision Tree

```
Need highest quality?
├── Yes → ltx-2-19b-dev (full BF16)
└── No → Need speed?
         ├── Yes → ltx-2-19b-distilled-fp8 (8 steps, fast)
         └── No → ltx-2-19b-dev-fp8 (balance)
```

### For Kids Content (Quality Priority)
**Recommended:** `ltx-2-19b-dev.safetensors` (full model)
- Variable inference steps (30-50 for quality)
- Best character rendering
- Smoothest motion

---

## Resolution Constraints

### Critical Rules
1. **Width & Height must be divisible by 32**
2. **Frame count must be (8 × n) + 1**

### Valid Resolutions

| Name | Dimensions | Notes |
|------|------------|-------|
| 540p | 960 × 544 | Fast, mobile-friendly |
| 720p | 1280 × 736 | Good balance |
| **1080p** | **1920 × 1088** | Best quality (NOT 1920x1080!) |
| 1440p | 2560 × 1440 | High-end |

### Valid Frame Counts

| Frames | Duration @ 24fps |
|--------|------------------|
| 25 | ~1 second |
| 49 | ~2 seconds |
| 73 | ~3 seconds |
| 97 | ~4 seconds |
| **121** | **~5 seconds** |
| 145 | ~6 seconds |
| 217 | ~9 seconds |

---

## Inference Settings

### Optimal Settings for Full Model

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--num-inference-steps` | 30-40 | 20 = fast, 50 = best quality |
| `--guidance-scale` | 7.5 | Standard. Higher = more prompt-adherent |
| `--fps` | 24 | Standard film rate |

### Speed vs Quality Tradeoffs

| Setting | Faster | Slower | Quality Impact |
|---------|--------|--------|----------------|
| Steps | 20 | 50 | Major |
| Resolution | 720p | 1080p | Moderate |
| Frames | 49 | 121 | Minor |

### NVIDIA Recommendations
- **20 steps is sweet spot** for speed/quality balance
- **30+ steps** for higher fidelity when time allows
- **1080p is max practical quality** - higher often doesn't help

---

## Prompt Engineering

### CRITICAL: Use `--enhance-prompt`

The `--enhance-prompt` flag uses Gemma to expand your prompt into film-direction style. This is **essential for quality**.

**Requires:** `unsloth/gemma-3-12b-it-bnb-4bit` (7.4GB, 4-bit quantized)

### Film-Style Prompt Template

```
"An animated scene, Pixar style 3D, [YOUR SCENE].
Shot on 35mm lens f/2.0, shallow depth of field,
soft key light 45 degrees, cinematic color grading,
avoid high-frequency patterns, smooth textures."
```

### Example Prompts for Kids Content

**Good:**
```
"An animated scene, Pixar style 3D, A cute colorful cartoon letter A
with big friendly eyes bouncing happily up and down.
Shot on 35mm lens f/2.0, soft key light 45 degrees,
vibrant saturated colors, smooth textures, children's animation style."
```

**Bad:**
```
"Letter A bouncing"
```

### Prompt Tips

1. **"3D animated style"** or **"Pixar quality"** improves character rendering
2. **"vibrant colors, soft lighting"** good for kids content
3. **Be specific about actions** ("bouncing up and down happily")
4. **Add technical details**: lens, lighting, camera movement
5. **Add guardrails**: "avoid high-frequency patterns", "smooth textures"

---

## Negative Prompts

### Always Include

```
"blurry, low quality, artifacts, watermark, text, logo, distorted,
morphing, flickering, inconsistent, jittery motion, dark, scary,
violent, inappropriate content"
```

### For Kids Content (Extended)

```
"blurry, low quality, artifacts, watermark, text, logo, distorted,
morphing, flickering, inconsistent, jittery motion, dark, scary,
violent, blood, weapons, adult content, horror, nightmare,
realistic human faces, uncanny valley"
```

---

## Performance Optimization

### Model Warmup

First inference is slow (model loading). Run a tiny warmup generation on server start:

```python
# warmup.py
# Generate 256x256, 9 frames to warm up GPU
```

### Batch Processing Tips

1. **Keep GPU warm** - don't let pod sleep between requests
2. **Batch similar resolutions** - avoids recompilation
3. **Pre-generate common content** during off-hours

### Memory Management

- Enable **weight streaming** if VRAM is tight
- Use **FP8 model** if under 48GB VRAM
- Reduce **frames first**, then resolution, then FPS

---

## Quality Checklist for Kids Content

- [ ] Using full model (`ltx-2-19b-dev.safetensors`)
- [ ] Using `--enhance-prompt` flag
- [ ] Resolution is 1920x1088 (1080p)
- [ ] Using 30+ inference steps
- [ ] Film-style prompt with lens/lighting details
- [ ] Negative prompt includes all blocklist terms
- [ ] Content is age-appropriate

---

## Expected Performance by GPU

| GPU | 5sec 1080p (30 steps) | Model Load |
|-----|----------------------|------------|
| H100 80GB | ~60-90 sec | ~10 sec |
| RTX 6000 Ada | ~120-150 sec | ~15 sec |
| RTX 4090 | ~150-180 sec | ~15 sec |
| A100 80GB | ~100-130 sec | ~12 sec |

---

## Sources

- [NVIDIA RTX Guide](https://www.nvidia.com/en-us/geforce/news/rtx-ai-video-generation-guide/)
- [LTX-2 Official](https://ltx.io/model/ltx-2)
- [Apatero Tips & Tricks](https://apatero.com/blog/ltx-2-tips-and-tricks-ai-video-generation-2025)
- [GitHub - Lightricks/LTX-2](https://github.com/Lightricks/LTX-2)
- Our experiments: `docs/LTX2-EXPERIMENTS.md`
