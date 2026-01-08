# LTX-2 Video Generation Experiments

**Started:** January 7, 2026
**Hardware:** RunPod RTX 4090 (24GB VRAM)
**Model:** ltx-2-19b-distilled-fp8 + 4-bit quantized Gemma

---

## Setup Summary

| Component | Details |
|-----------|---------|
| Pod ID | `elfm76kspb7lus` |
| GPU | RTX 4090 (24GB VRAM) |
| Video Model | ltx-2-19b-distilled-fp8.safetensors (26GB) |
| Text Encoder | unsloth/gemma-3-12b-it-bnb-4bit (7.4GB) |
| Spatial Upscaler | ltx-2-spatial-upscaler-x2-1.0.safetensors (950MB) |
| Cost | $0.59/hour |

### Key Finding: Gemma Quantization
- Original Gemma (23GB) caused OOM on 4090
- Switched to 4-bit quantized version (7.4GB)
- Works perfectly, no quality loss noticed

---

## Experiments

### Experiment 1: Low Quality Test
**File:** `test_video.mp4`

| Setting | Value |
|---------|-------|
| Resolution | 512x768 |
| Frames | 33 (~1 sec) |
| Prompt | "A friendly cartoon cat waving hello" |
| Generation Time | ~5-6 seconds |
| File Size | 188KB |

**Result:** Works, but quality is "mid" - good for quick tests only.

---

### Experiment 2: Low Quality Magima Style
**File:** `magima_letter_A.mp4`

| Setting | Value |
|---------|-------|
| Resolution | 512x768 |
| Frames | 33 (~1 sec) |
| Prompt | "A colorful animated letter A bouncing happily, cute cartoon style for children, bright colors, playful animation, educational kids content" |
| Generation Time | ~5-6 seconds |
| File Size | 129KB |

**Result:** Same quality issues as test.

---

### Experiment 3: Higher Quality
**File:** `magima_letter_A_hq.mp4`

| Setting | Value |
|---------|-------|
| Resolution | 704x1280 (upscaled to 1408x2560) |
| Frames | 65 (~2 sec) |
| Prompt | "A cute colorful cartoon letter A character with big eyes and a smile, bouncing up and down happily, 3D animated style, Pixar quality, vibrant colors, soft lighting, kids educational content" |
| Generation Time | ~12 seconds |
| File Size | 371KB |

**Result:** Much better! Acceptable quality for kids content.

---

### Experiment 11-12: Distilled + Enhanced Prompt (BREAKTHROUGH)
**Key Discovery:** `--enhance-prompt` flag uses Gemma to expand prompts automatically!

**File:** `exp11_distilled_filmprompt.mp4`, `exp12_distilled_4sec.mp4`

| Setting | Value |
|---------|-------|
| Resolution | 704x1280 (upscaled to 1408x2560) |
| Frames | 65-97 (2.7-4 sec) |
| Prompt Style | Film-direction style with camera/lens details |
| Flags | `--enhance-prompt --enable-fp8` |
| Generation Time | 83-84 seconds |

**Result:** Significant quality improvement! The enhanced prompt + film-style prompting produces much better results.

---

### Experiment 16-18: Resolution/Length Optimization (BEST RESULT)

| Exp | Resolution | Frames | Length | Gen Time | Result |
|-----|------------|--------|--------|----------|--------|
| 16 | 512x768 | 121 | 5 sec | ~25s | ✓ Good |
| 17 | 512x768 | 241 | 10 sec | - | ✗ OOM |
| 18 | 512x768 | 217 | 9 sec | ~30s | ✓ **BEST** |

**BEST CONFIG:** 512x768, 217 frames, `--enhance-prompt` = **9 sec video in 30 seconds!**

Output: 1024x1536 after 2x upscale. Quality is good, slightly soft but acceptable for kids content.

---

## Technical Notes

### Resolution Requirements
- Height and width must be divisible by 64
- Spatial upscaler doubles resolution (704x1280 → 1408x2560)

**Valid resolutions:**
| Name | Base | After Upscale |
|------|------|---------------|
| Small | 512x768 | 1024x1536 |
| Medium | 704x1280 | 1408x2560 |
| Large | 768x1344 | 1536x2688 |

### Frame Count
- 33 frames ≈ 1 second @ 30fps
- 65 frames ≈ 2 seconds @ 30fps
- 97 frames ≈ 3 seconds @ 30fps

### Prompt Engineering Tips
- "3D animated style" helps quality
- "Pixar quality" improves character rendering
- "vibrant colors, soft lighting" good for kids content
- Be specific about actions ("bouncing up and down")
- **Use `--enhance-prompt` flag** - Gemma auto-expands prompts significantly
- **Film-style prompts work best:** Include lens (35mm f/2.0), lighting (soft key light 45°), camera movement
- Add guardrails: "avoid high-frequency patterns", "smooth textures"

### Performance Limits (RTX 4090, 24GB VRAM)

| Resolution | Max Length | Gen Time | Final Output |
|------------|------------|----------|--------------|
| 704x1280 | 4 sec (97 frames) | ~84s | 1408x2560 |
| 512x768 | 9 sec (217 frames) | ~30s | 1024x1536 |
| 384x640 | 10+ sec (241 frames) | ~78s | 768x1280 |

### Key Findings
- **Distilled pipeline locked to 8 steps** - `--num-inference-steps` flag is ignored
- **FP4 model doesn't exist for distilled** - Only dev architecture has FP4, incompatible
- **Lower res + upscaler = sharper** - 480p upscaled beats native 720p in blind tests
- **Spatial upscaler is required** - Distilled pipeline won't run without it
- **Quality pipeline (ti2vid_one_stage)** - Supports more steps but ~6x slower

---

## Command Template (Recommended)

```bash
cd /workspace/LTX-2 && .venv/bin/python -m ltx_pipelines.distilled \
  --checkpoint-path ./models/ltx-2-19b-distilled-fp8.safetensors \
  --spatial-upsampler-path ./models/ltx-2-spatial-upscaler-x2-1.0.safetensors \
  --gemma-root ./models/gemma \
  --prompt "An animated scene, Pixar style 3D, [YOUR SCENE]. Shot on 35mm lens f/2.0, shallow depth of field, soft key light 45 degrees, cinematic color grading, avoid high-frequency patterns, smooth textures." \
  --output-path /workspace/OUTPUT_NAME.mp4 \
  --enable-fp8 \
  --enhance-prompt \
  --height 512 --width 768 \
  --num-frames 217
```

**Note:** 512x768 with 217 frames = 9 seconds of video, outputs at 1024x1536 after upscaling.

---

## Cost Analysis

| Duration | Videos (12s each) | Cost |
|----------|-------------------|------|
| 1 hour | ~300 videos | $0.59 |
| Per video | 1 | ~$0.002 |

Very cost effective for batch generation!

---

## TODO / Next Experiments

**Completed:**
- [x] Test longer videos (97+ frames) - found 217 frames max at 512x768
- [x] Experiment with --num-inference-steps - locked to 8 for distilled
- [x] Try without spatial upscaler - not possible, it's required
- [x] Research FP4 model - doesn't exist for distilled pipeline

**Still To Try:**
- [ ] Try even lower res (384x576) with lens prompting for sharper upscale
- [ ] Test image-to-video (--image flag)
- [ ] Try LoRA for style consistency
- [ ] Install ffmpeg on pod for post-processing sharpening
- [ ] Test quality pipeline (ti2vid_one_stage) with more steps for comparison
- [ ] Benchmark different prompts for kids content

---

## Files Generated

| File | Resolution | Frames | Length | Quality |
|------|------------|--------|--------|---------|
| test_video.mp4 | 512x768 | 33 | 1s | Low |
| magima_letter_A.mp4 | 512x768 | 33 | 1s | Low |
| magima_letter_A_hq.mp4 | 704x1280 | 65 | 2.7s | Good |
| exp11_distilled_filmprompt.mp4 | 704x1280 | 65 | 2.7s | Good |
| exp12_distilled_4sec.mp4 | 704x1280 | 97 | 4s | Good |
| exp14_8sec_lowres.mp4 | 512x768 | 193 | 8s | Good |
| exp15_10sec_verylow.mp4 | 384x640 | 241 | 10s | Good |
| exp16_512_5sec.mp4 | 512x768 | 121 | 5s | Good |
| exp18_512_9sec.mp4 | 512x768 | 217 | 9s | **Best** |
