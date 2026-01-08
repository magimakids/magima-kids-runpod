"""Video generation pipeline"""
import argparse
from pathlib import Path
from datetime import datetime

from src.models.ltx import LTXVideoGenerator


def generate_video(
    prompt: str,
    output_dir: str = "outputs",
    width: int = 704,
    height: int = 480,
    num_frames: int = 65,
    steps: int = 30,
    seed: int = None,
):
    """Generate a video from a text prompt"""

    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename from timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_file = output_path / f"video_{timestamp}.mp4"

    # Initialize and run generator
    generator = LTXVideoGenerator()
    generator.load()

    print(f"\nGenerating video...")
    print(f"  Prompt: {prompt}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Frames: {num_frames}")
    print(f"  Steps: {steps}")

    frames = generator.generate(
        prompt=prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=steps,
        seed=seed,
    )

    generator.save_video(frames, str(video_file))

    return str(video_file)


def main():
    parser = argparse.ArgumentParser(description="Generate video from text prompt")
    parser.add_argument("--prompt", "-p", required=True, help="Text prompt for video generation")
    parser.add_argument("--output", "-o", default="outputs", help="Output directory")
    parser.add_argument("--width", "-W", type=int, default=704, help="Video width")
    parser.add_argument("--height", "-H", type=int, default=480, help="Video height")
    parser.add_argument("--frames", "-f", type=int, default=65, help="Number of frames")
    parser.add_argument("--steps", "-s", type=int, default=30, help="Inference steps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    video_path = generate_video(
        prompt=args.prompt,
        output_dir=args.output,
        width=args.width,
        height=args.height,
        num_frames=args.frames,
        steps=args.steps,
        seed=args.seed,
    )

    print(f"\nDone! Video saved to: {video_path}")


if __name__ == "__main__":
    main()
