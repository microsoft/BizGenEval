"""
Image generation demo with example models (Qwen-Image, Z-Image).

Loads a model, reads prompts from a JSONL dataset, generates images,
and saves them to disk. Supports dynamic resolution via resolution_utils.

Usage:
    python -m generation.image_generation \
        --data_path <prompts.jsonl> \
        --save_dir outputs/generated_images \
        --resolution_mode dynamic_max_pixels \
        --skip_existing

See config/generation_config.yaml for model configuration.
"""

import sys
sys.path.append(".")
import os
import json
import yaml
import argparse

from generation.models import GenerationModels
from generation.resolution_utils import resolve_resolution


def parse_args():
    parser = argparse.ArgumentParser(description="Image Generation Demo")
    parser.add_argument("--config_path", type=str, default="config/generation_config.yaml", help="Path to model configuration YAML file.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to JSONL file containing prompts.")
    parser.add_argument("--save_dir", type=str, default="outputs/generated_images", help="Directory to save generated images.")
    parser.add_argument("--only_models", type=str, nargs="*", default=None, help="Only run these model names from config.")
    parser.add_argument("--seeds", type=int, nargs="*", default=[42], help="Random seeds for generation.")
    parser.add_argument("--resolution_mode", type=str, default="config", choices=["config", "dynamic_original", "dynamic_max_pixels"], help="Resolution mode.")
    parser.add_argument("--max_pixel_size", type=int, default=None, help="Max pixels for dynamic resolution modes.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip if output image already exists (resume).")
    return parser.parse_args()


def load_data(path: str):
    """Load prompts from JSONL or JSON file."""
    items = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for fname, prompt in data.items():
            items.append({"id": fname, "prompt": prompt})
    else:
        raise ValueError(f"Unsupported data file format: {path}")
    return items


def infer_filename(item: dict) -> str:
    """Infer output filename from dataset item."""
    for k in ("reference_image", "reference image", "image_path"):
        ref = item.get(k)
        if isinstance(ref, str) and ref.strip():
            return os.path.basename(ref.strip())
    domain = str(item.get("domain", "")).strip()
    dimension = str(item.get("dimension", "")).strip()
    idx = item.get("id", "0")
    return f"{domain}_{dimension}_{idx}.png"


def main():
    args = parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    resolution_config = config.pop("resolution", None) or {}
    max_pixel_size = args.max_pixel_size or resolution_config.get("max_pixel_size")
    stride = resolution_config.get("stride")

    data_items = load_data(args.data_path)
    print(f"Loaded {len(data_items)} items from {args.data_path}")

    for model_name, model_config in config.items():
        if not isinstance(model_config, dict) or "model" not in model_config:
            continue
        if args.only_models is not None and model_name not in args.only_models:
            continue

        print(f"\n=== Model: {model_name} ===")
        gen_model = GenerationModels(model_config["model"], device="cuda")

        for seed in args.seeds:
            run_config = dict(model_config)
            run_config["seed"] = seed

            for item in data_items:
                prompt = item.get("prompt") or item.get("description", "")
                fname = infer_filename(item)

                run_config.update(resolve_resolution(
                    args.resolution_mode, item, model_config, model_config["model"],
                    max_pixel_size=max_pixel_size, stride=stride,
                ))

                save_path = os.path.join(args.save_dir, model_name, fname)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                if args.skip_existing and os.path.isfile(save_path):
                    print(f"  [SKIP] {fname} (exists)")
                    continue

                print(f"  Generating: {fname} (h={run_config.get('height')}, w={run_config.get('width')})")
                gen_model.generate_image(prompt=prompt, save_path=save_path, **run_config)

        gen_model._clear_model()
        print(f"Finished model: {model_name}")


if __name__ == "__main__":
    main()
