"""Video generation pipeline with presets and prompt enhancement"""
import argparse
from pathlib import Path
from datetime import datetime

from src.models.ltx import LTXVideoGenerator, PRESETS


def generate_video(
    prompt: str,
    output_dir: str = "outputs",
    preset: str = None,
    width: int = None,
    height: int = None,
    num_frames: int = None,
    steps: int = None,
    guidance: float = None,
    seed: int = None,
    enhance_prompt: bool = None,
    film_template: bool = None,
    negative_prompt: str = None,
    no_enhance: bool = False,
):
    """Generate a video from a text prompt

    Args:
        prompt: Text description of the video
        output_dir: Directory to save output
        preset: Use a preset configuration (kids-1080p-5sec, kids-720p-5sec, fast-test)
        width: Video width (overrides preset)
        height: Video height (overrides preset)
        num_frames: Number of frames (overrides preset)
        steps: Inference steps (overrides preset)
        guidance: Guidance scale (overrides preset)
        seed: Random seed
        enhance_prompt: Use Gemma to enhance prompt (overrides preset)
        film_template: Wrap prompt in film-style template (overrides preset)
        negative_prompt: Custom negative prompt
        no_enhance: Disable prompt enhancement
    """
    # Start with preset defaults
    config = {}
    if preset and preset in PRESETS:
        config = PRESETS[preset].copy()
        print(f"Using preset: {preset}")
    else:
        # Default to kids-1080p-5sec
        config = PRESETS["kids-1080p-5sec"].copy()

    # Override with explicit arguments
    if width is not None:
        config["width"] = width
    if height is not None:
        config["height"] = height
    if num_frames is not None:
        config["num_frames"] = num_frames
    if steps is not None:
        config["num_inference_steps"] = steps
    if guidance is not None:
        config["guidance_scale"] = guidance
    if enhance_prompt is not None:
        config["enhance_prompt"] = enhance_prompt
    if film_template is not None:
        config["use_film_template"] = film_template
    if no_enhance:
        config["enhance_prompt"] = False

    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename from timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_file = output_path / f"video_{timestamp}.mp4"

    # Initialize generator
    generator = LTXVideoGenerator(
        use_prompt_enhancement=config.get("enhance_prompt", True)
    )
    generator.load()

    print(f"\n{'='*50}")
    print(f"Generating video")
    print(f"{'='*50}")
    print(f"Prompt: {prompt}")
    print(f"Resolution: {config['width']}x{config['height']}")
    print(f"Frames: {config['num_frames']} (~{config['num_frames']/24:.1f}s)")
    print(f"Steps: {config['num_inference_steps']}")
    print(f"Enhance prompt: {config.get('enhance_prompt', True)}")
    print(f"Film template: {config.get('use_film_template', False)}")
    print(f"{'='*50}\n")

    frames = generator.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=config["width"],
        height=config["height"],
        num_frames=config["num_frames"],
        num_inference_steps=config["num_inference_steps"],
        guidance_scale=config["guidance_scale"],
        seed=seed,
        enhance_prompt=config.get("enhance_prompt", True),
        use_film_template=config.get("use_film_template", False),
    )

    generator.save_video(frames, str(video_file))

    return str(video_file)


def main():
    parser = argparse.ArgumentParser(
        description="Generate video from text prompt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  kids-1080p-5sec  1920x1088, 121 frames, 40 steps, enhanced (default)
  kids-720p-5sec   1280x736, 121 frames, 30 steps, enhanced
  fast-test        512x512, 25 frames, 20 steps, no enhancement

Examples:
  # Quick test
  python -m src.pipelines.generate -p "a cat playing" --preset fast-test

  # High quality kids content
  python -m src.pipelines.generate -p "A colorful letter A bouncing happily"

  # Custom settings
  python -m src.pipelines.generate -p "A dancing robot" -W 1280 -H 736 -s 30
        """
    )
    parser.add_argument("--prompt", "-p", required=True, help="Text prompt for video generation")
    parser.add_argument("--output", "-o", default="outputs", help="Output directory")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), help="Use preset configuration")
    parser.add_argument("--width", "-W", type=int, help="Video width (divisible by 32)")
    parser.add_argument("--height", "-H", type=int, help="Video height (divisible by 32)")
    parser.add_argument("--frames", "-f", type=int, help="Number of frames (8n+1)")
    parser.add_argument("--steps", "-s", type=int, help="Inference steps (20-50)")
    parser.add_argument("--guidance", "-g", type=float, help="Guidance scale")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--enhance-prompt", action="store_true", help="Enable prompt enhancement")
    parser.add_argument("--no-enhance", action="store_true", help="Disable prompt enhancement")
    parser.add_argument("--film-template", action="store_true", help="Wrap in film-style template")
    parser.add_argument("--negative-prompt", "-n", help="Custom negative prompt")

    args = parser.parse_args()

    video_path = generate_video(
        prompt=args.prompt,
        output_dir=args.output,
        preset=args.preset,
        width=args.width,
        height=args.height,
        num_frames=args.frames,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        enhance_prompt=args.enhance_prompt if args.enhance_prompt else None,
        film_template=args.film_template,
        negative_prompt=args.negative_prompt,
        no_enhance=args.no_enhance,
    )

    print(f"\nDone! Video saved to: {video_path}")


if __name__ == "__main__":
    main()
