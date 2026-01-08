#!/usr/bin/env python3
"""
Warmup script for LTX-2 Video Generator

First inference is always slow due to model loading and CUDA compilation.
Run this script after starting a pod to warm up the GPU before generating real content.
"""
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def warmup():
    """Run minimal generation to warm up GPU and model"""
    print("=" * 50)
    print("LTX-2 Model Warmup")
    print("=" * 50)
    print("")
    print("This will run a tiny test generation to warm up the GPU.")
    print("First inference is always slow - subsequent ones will be faster.")
    print("")

    start_time = time.time()

    from src.models.ltx import LTXVideoGenerator

    # Initialize without prompt enhancement for faster warmup
    print("Loading model (this may take a minute)...")
    generator = LTXVideoGenerator(use_prompt_enhancement=False)
    generator.load()

    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.1f}s")
    print("")

    # Run tiny generation
    print("Running warmup generation (256x256, 9 frames)...")
    gen_start = time.time()

    frames = generator.generate(
        prompt="test warmup",
        width=256,
        height=256,
        num_frames=9,
        num_inference_steps=4,
        enhance_prompt=False,
    )

    gen_time = time.time() - gen_start
    total_time = time.time() - start_time

    print("")
    print("=" * 50)
    print("Warmup complete!")
    print("=" * 50)
    print(f"  Model load time: {load_time:.1f}s")
    print(f"  Test generation: {gen_time:.1f}s")
    print(f"  Total warmup:    {total_time:.1f}s")
    print("")
    print("GPU is now warm. Future generations will be faster.")
    print("")
    print("Try generating a real video:")
    print("  python -m src.pipelines.generate -p 'A colorful letter bouncing'")
    print("")


if __name__ == "__main__":
    warmup()
