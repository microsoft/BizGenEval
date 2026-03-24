"""
Lightweight wrapper for local diffusion models.

Currently supports Qwen-Image and Z-Image as examples.
To add a new model, implement a load branch in load_model() and a
generation method, then register it in generate_image().
"""

import os
import torch
from PIL import Image


QWEN_IMAGE_ASPECT_RATIOS = {
    "1:1": (1328, 1328), "16:9": (1664, 928), "9:16": (928, 1664),
    "4:3": (1472, 1104), "3:4": (1104, 1472), "3:2": (1584, 1056), "2:3": (1056, 1584),
}


class GenerationModels:
    """
    Unified interface for local image generation models.

    Example:
        model = GenerationModels("Qwen/Qwen-Image", device="cuda")
        image = model.generate_image("a cat", save_path="out.png", seed=42)
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.pipe = None
        self.load_model(model_name)

    def load_model(self, model_name: str):
        print(f"Loading model {model_name} ...")
        self._clear_model()

        if model_name in ("Qwen/Qwen-Image-2512", "Qwen/Qwen-Image"):
            from diffusers import DiffusionPipeline
            self.pipe = DiffusionPipeline.from_pretrained(
                model_name, torch_dtype=torch.bfloat16,
            )
            self.pipe.enable_model_cpu_offload()

        elif model_name in ("Tongyi-MAI/Z-Image-Turbo", "Tongyi-MAI/Z-Image"):
            from diffusers import ZImagePipeline
            self.pipe = ZImagePipeline.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False,
            )
            self.pipe.to(self.device)

        else:
            raise ValueError(
                f"Model '{model_name}' is not supported. "
                f"Supported: Qwen/Qwen-Image-2512, Qwen/Qwen-Image, "
                f"Tongyi-MAI/Z-Image-Turbo, Tongyi-MAI/Z-Image"
            )

    def generate_image(
        self, prompt: str, save_path: str = None, **kwargs,
    ) -> Image.Image:
        if self.model_name in ("Qwen/Qwen-Image-2512", "Qwen/Qwen-Image"):
            image = self._qwen_image_generation(prompt, **kwargs)
        elif self.model_name in ("Tongyi-MAI/Z-Image-Turbo", "Tongyi-MAI/Z-Image"):
            image = self._zimage_generation(prompt, **kwargs)
        else:
            raise ValueError(f"Model '{self.model_name}' is not supported.")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            image.save(save_path)
        return image

    def _qwen_image_generation(
        self, prompt: str,
        aspect_ratio: str = "16:9",
        negative_prompt: str = " ",
        num_inference_steps: int = 50,
        true_cfg_scale: float = 4.0,
        seed: int = 42,
        **kwargs,
    ) -> Image.Image:
        width = kwargs.pop("width", None)
        height = kwargs.pop("height", None)
        if width is None or height is None:
            width, height = QWEN_IMAGE_ASPECT_RATIOS.get(aspect_ratio, (1328, 1328))
        return self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width, height=height,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).images[0]

    def _zimage_generation(
        self, prompt: str,
        height: int = 2048, width: int = 2048,
        num_inference_steps: int = 9,
        guidance_scale: float = 0.0,
        seed: int = 42,
        **kwargs,
    ) -> Image.Image:
        return self.pipe(
            prompt=prompt,
            height=height, width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]

    def _clear_model(self):
        self.pipe = None
