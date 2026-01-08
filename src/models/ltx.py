"""LTX-2 Video model wrapper with prompt enhancement"""
import sys
from pathlib import Path
from typing import Optional

# Add LTX-2 to path when running on RunPod
LTX_PATH = Path(__file__).parent.parent.parent / "models" / "LTX-2"
if LTX_PATH.exists():
    sys.path.insert(0, str(LTX_PATH))

# Default paths
DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "ltx-2-19b-dev.safetensors"
DEFAULT_GEMMA_PATH = Path(__file__).parent.parent.parent / "models" / "gemma"

# Default negative prompt for kids content
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, artifacts, watermark, text, logo, distorted, "
    "morphing, flickering, inconsistent, jittery motion, dark, scary, "
    "violent, blood, weapons, adult content, horror, nightmare"
)

# Film-style prompt template
FILM_PROMPT_TEMPLATE = """An animated scene, Pixar style 3D, {scene}.
Shot on 35mm lens f/2.0, shallow depth of field,
soft key light 45 degrees, cinematic color grading,
avoid high-frequency patterns, smooth textures, vibrant colors."""


class PromptEnhancer:
    """Enhance prompts using Gemma model"""

    def __init__(self, model_path: str = None):
        self.model_path = model_path or str(DEFAULT_GEMMA_PATH)
        self.model = None
        self.tokenizer = None

    def load(self):
        """Load Gemma model for prompt enhancement"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            print(f"Loading Gemma from {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            print("Gemma loaded!")
        except Exception as e:
            print(f"Warning: Could not load Gemma for prompt enhancement: {e}")
            self.model = None
        return self

    def enhance(self, prompt: str) -> str:
        """Enhance a prompt using Gemma"""
        if self.model is None:
            print("Gemma not loaded, using original prompt")
            return prompt

        system_prompt = """You are a professional cinematographer. Expand the following video prompt
into a detailed film-direction style description. Include:
- Visual style (3D animated, Pixar quality, etc.)
- Camera details (lens, depth of field)
- Lighting (soft key light, etc.)
- Motion description
- Color palette
Keep it concise but descriptive. Output only the enhanced prompt, nothing else."""

        full_prompt = f"{system_prompt}\n\nOriginal prompt: {prompt}\n\nEnhanced prompt:"

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
        )
        enhanced = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the enhanced prompt part
        if "Enhanced prompt:" in enhanced:
            enhanced = enhanced.split("Enhanced prompt:")[-1].strip()

        print(f"Enhanced prompt: {enhanced[:100]}...")
        return enhanced


class LTXVideoGenerator:
    """Wrapper for LTX-2 Video generation with prompt enhancement"""

    def __init__(
        self,
        model_path: str = None,
        gemma_path: str = None,
        use_prompt_enhancement: bool = True,
    ):
        self.model_path = model_path or str(DEFAULT_MODEL_PATH)
        self.gemma_path = gemma_path or str(DEFAULT_GEMMA_PATH)
        self.use_prompt_enhancement = use_prompt_enhancement
        self.pipeline = None
        self.prompt_enhancer = None

    def load(self):
        """Load the model pipeline and optional prompt enhancer"""
        import torch

        # Load prompt enhancer first (uses less VRAM)
        if self.use_prompt_enhancement:
            self.prompt_enhancer = PromptEnhancer(self.gemma_path)
            self.prompt_enhancer.load()

        # Load LTX-2 pipeline
        print(f"Loading LTX-2 from {self.model_path}...")

        # Try to use the LTX-2 pipeline if available
        try:
            # Import from LTX-2 package if installed
            from ltx_pipelines import LTXPipeline
            self.pipeline = LTXPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
            ).to("cuda")
        except ImportError:
            # Fallback to diffusers
            from diffusers import DiffusionPipeline
            self.pipeline = DiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
            ).to("cuda")

        print("LTX-2 loaded!")
        return self

    def warmup(self):
        """Run a tiny generation to warm up GPU/model"""
        print("Warming up model...")
        if self.pipeline is None:
            self.load()

        # Tiny generation: 256x256, 9 frames
        _ = self.generate(
            prompt="test",
            width=256,
            height=256,
            num_frames=9,
            num_inference_steps=4,
            enhance_prompt=False,
        )
        print("Warmup complete!")

    def generate(
        self,
        prompt: str,
        negative_prompt: str = None,
        width: int = 1920,
        height: int = 1088,
        num_frames: int = 121,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        enhance_prompt: bool = True,
        use_film_template: bool = False,
    ) -> list:
        """Generate video frames from prompt

        Args:
            prompt: Text description of the video
            negative_prompt: What to avoid (defaults to kids-safe blocklist)
            width: Video width (must be divisible by 32)
            height: Video height (must be divisible by 32)
            num_frames: Number of frames (must be 8n+1)
            num_inference_steps: Denoising steps (20-50, higher = better quality)
            guidance_scale: CFG scale (7.5 is standard)
            seed: Random seed for reproducibility
            enhance_prompt: Whether to use Gemma to enhance prompt
            use_film_template: Whether to wrap prompt in film-style template
        """
        import torch

        if self.pipeline is None:
            self.load()

        # Validate dimensions
        if width % 32 != 0 or height % 32 != 0:
            raise ValueError(f"Width ({width}) and height ({height}) must be divisible by 32")
        if (num_frames - 1) % 8 != 0:
            raise ValueError(f"num_frames ({num_frames}) must be (8 × n) + 1")

        # Apply film template if requested
        if use_film_template:
            prompt = FILM_PROMPT_TEMPLATE.format(scene=prompt)

        # Enhance prompt if enabled
        if enhance_prompt and self.prompt_enhancer and self.prompt_enhancer.model:
            prompt = self.prompt_enhancer.enhance(prompt)

        # Use default negative prompt if not provided
        if negative_prompt is None:
            negative_prompt = DEFAULT_NEGATIVE_PROMPT

        # Set up generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        print(f"\nGenerating video:")
        print(f"  Resolution: {width}x{height}")
        print(f"  Frames: {num_frames} (~{num_frames/24:.1f}s @ 24fps)")
        print(f"  Steps: {num_inference_steps}")
        print(f"  Prompt: {prompt[:80]}...")

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

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"Video saved to {output_path}")
        return output_path


# Presets for common use cases
PRESETS = {
    "kids-1080p-5sec": {
        "width": 1920,
        "height": 1088,
        "num_frames": 121,
        "num_inference_steps": 40,
        "guidance_scale": 7.5,
        "enhance_prompt": True,
        "use_film_template": True,
    },
    "kids-720p-5sec": {
        "width": 1280,
        "height": 736,
        "num_frames": 121,
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "enhance_prompt": True,
        "use_film_template": True,
    },
    "fast-test": {
        "width": 512,
        "height": 512,
        "num_frames": 25,
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "enhance_prompt": False,
        "use_film_template": False,
    },
}
