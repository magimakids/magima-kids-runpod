"""LTX-Video model wrapper"""
import sys
from pathlib import Path

# Add LTX-Video to path when running on RunPod
LTX_PATH = Path(__file__).parent.parent.parent / "models" / "LTX-Video"
if LTX_PATH.exists():
    sys.path.insert(0, str(LTX_PATH))


class LTXVideoGenerator:
    """Wrapper for LTX-Video generation"""

    def __init__(self, model_path: str = None):
        self.model_path = model_path or str(
            Path(__file__).parent.parent.parent / "models" / "ltx-video-weights"
        )
        self.pipeline = None

    def load(self):
        """Load the model pipeline"""
        # Import here to avoid loading on import
        import torch
        from diffusers import DiffusionPipeline

        print(f"Loading LTX-Video from {self.model_path}...")
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        print("Model loaded!")
        return self

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 704,
        height: int = 480,
        num_frames: int = 65,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int = None,
    ) -> list:
        """Generate video frames from prompt"""
        import torch

        if self.pipeline is None:
            self.load()

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        return output.frames[0]

    def save_video(self, frames: list, output_path: str, fps: int = 24):
        """Save frames as video"""
        import imageio

        imageio.mimsave(output_path, frames, fps=fps)
        print(f"Video saved to {output_path}")
        return output_path
