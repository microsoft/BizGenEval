"""
Dynamic resolution configuration for image generation models.

Supports three modes:
- config:             Use height/width or aspect_ratio from model config YAML.
- dynamic_original:   Use reference_image_wh as-is, snap to model-supported resolution + stride.
- dynamic_max_pixels: Scale reference_image_wh to fit max_pixel_size, then snap.

Example:
    from resolution_utils import resolve_resolution

    resolution = resolve_resolution(
        mode="dynamic_max_pixels",
        item={"reference_image_wh": "1920x1080"},
        model_config={"model": "black-forest-labs/FLUX.2-dev"},
        model_name="black-forest-labs/FLUX.2-dev",
        max_pixel_size=2048 * 2048,
    )
    # resolution -> {"height": 1080, "width": 1920}  (snapped to stride)
"""
from __future__ import annotations

import re
from typing import List, Tuple, Optional

ASPECT_RATIO_LABELS = ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]

DEFAULT_STRIDE = 32

GPT_IMAGE_FIXED_SIZES = [
    (1024, 1024),
    (1536, 1024),
    (1024, 1536),
]

MODEL_RESOLUTION_META = {
    "gpt-image-1.5": {
        "resolution_type": "fixed_sizes",
        "stride": None,
        "supported_sizes_list": GPT_IMAGE_FIXED_SIZES,
        "max_pixels_default": 1536 * 1024,
    },
    "gpt_image_1.5": {
        "resolution_type": "fixed_sizes",
        "stride": None,
        "supported_sizes_list": GPT_IMAGE_FIXED_SIZES,
        "max_pixels_default": 1536 * 1024,
    },
    "black-forest-labs/FLUX.2-dev": {
        "resolution_type": "flexible",
        "stride": 16,
        "max_pixels_default": 2048 * 2048,
        "supported_aspect_ratios": None,
    },
    "black-forest-labs/FLUX.1-dev": {
        "resolution_type": "flexible",
        "stride": 8,
        "max_pixels_default": 2048 * 2048,
        "supported_aspect_ratios": None,
    },
    "stabilityai/stable-diffusion-3.5-large": {
        "resolution_type": "flexible",
        "stride": 16,
        "max_pixels_default": 1024 * 1024,
        "supported_aspect_ratios": None,
    },
}


def _parse_reference_wh(reference_image_wh: Optional[str]) -> Optional[Tuple[int, int]]:
    """Parse 'WxH' string -> (width, height)."""
    if not reference_image_wh or not isinstance(reference_image_wh, str):
        return None
    parts = re.split(r"[xX*×]", reference_image_wh.strip())
    if len(parts) != 2:
        return None
    try:
        w, h = int(parts[0].strip()), int(parts[1].strip())
        if w > 0 and h > 0:
            return (w, h)
    except ValueError:
        pass
    return None


def _aspect_ratio_from_wh(w: int, h: int) -> str:
    """Return the closest standard aspect ratio label for (w, h)."""
    if h == 0:
        return "1:1"
    ratio = w / h
    candidates = {
        "1:1": 1.0, "16:9": 16 / 9, "9:16": 9 / 16,
        "4:3": 4 / 3, "3:4": 3 / 4, "3:2": 3 / 2, "2:3": 2 / 3,
    }
    return min(candidates.items(), key=lambda x: abs(x[1] - ratio))[0]


def _scale_to_max_pixels(w: int, h: int, max_pixels: int) -> Tuple[int, int]:
    if w <= 0 or h <= 0:
        return (w, h)
    if w * h <= max_pixels:
        return (w, h)
    scale = (max_pixels / (w * h)) ** 0.5
    return (max(1, int(w * scale)), max(1, int(h * scale)))


def _round_to_stride(x: int, stride: int) -> int:
    return round(x / stride) * stride


def _snap_to_stride(w: int, h: int, stride: int) -> Tuple[int, int]:
    return (
        max(stride, _round_to_stride(w, stride)),
        max(stride, _round_to_stride(h, stride)),
    )


def _closest_fixed_size(
    w_ref: int, h_ref: int,
    supported_list: List[Tuple[int, int]],
    max_pixels: Optional[int] = None,
) -> Tuple[int, int]:
    if not supported_list:
        return (1024, 1024)
    if h_ref <= 0:
        return supported_list[0]
    target_ratio = w_ref / h_ref
    best, best_diff = None, float("inf")
    for (w, h) in supported_list:
        if h == 0:
            continue
        if max_pixels is not None and w * h > max_pixels:
            continue
        diff = abs(w / h - target_ratio)
        if diff < best_diff:
            best_diff = diff
            best = (w, h)
    return best or supported_list[0]


def get_model_meta(model_name: str) -> dict:
    """Look up resolution metadata for a model. Falls back to flexible/stride-32."""
    for key, meta in MODEL_RESOLUTION_META.items():
        if key in model_name or model_name in key:
            return dict(meta)
    return {
        "resolution_type": "flexible",
        "stride": DEFAULT_STRIDE,
        "max_pixels_default": 2048 * 2048,
        "supported_aspect_ratios": None,
    }


def resolve_resolution(
    mode: str,
    item: dict,
    model_config: dict,
    model_name: str,
    *,
    max_pixel_size: Optional[int] = None,
    stride: Optional[int] = None,
) -> dict:
    """
    Resolve image generation resolution.

    Args:
        mode: One of "config", "dynamic_original", "dynamic_max_pixels".
        item: Dataset row (may contain reference_image_wh, aspect_ratio).
        model_config: Model config from YAML (may contain height, width, aspect_ratio).
        model_name: Model identifier string.
        max_pixel_size: Override max pixel count for dynamic modes.
        stride: Override stride for rounding.

    Returns:
        dict with "height" and "width" (and optionally "aspect_ratio").
    """
    meta = get_model_meta(model_name)
    resolution_type = meta.get("resolution_type", "flexible")
    model_stride = meta.get("stride") or stride or DEFAULT_STRIDE
    default_max = meta.get("max_pixels_default", 1024 * 1024)
    max_pixels = max_pixel_size if max_pixel_size is not None else default_max

    if mode == "config":
        out = {}
        if "height" in model_config and "width" in model_config:
            out["height"], out["width"] = model_config["height"], model_config["width"]
        if "aspect_ratio" in model_config:
            out["aspect_ratio"] = model_config["aspect_ratio"]
        return out if out else {"height": 2048, "width": 2048}

    ref_wh = _parse_reference_wh(item.get("reference_image_wh") or item.get("wh"))
    if not ref_wh:
        ar = item.get("aspect_ratio") or model_config.get("aspect_ratio") or "1:1"
        ratio_map = {
            "1:1": 1.0, "16:9": 16 / 9, "9:16": 9 / 16,
            "4:3": 4 / 3, "3:4": 3 / 4, "3:2": 3 / 2, "2:3": 2 / 3,
        }
        r = ratio_map.get(ar, 1.0) or 1.0
        h0 = max(1, int((max_pixels / r) ** 0.5))
        w0 = int(h0 * r)
        ref_wh = (w0, h0)

    w_ref, h_ref = ref_wh

    if mode == "dynamic_original":
        target_w, target_h = w_ref, h_ref
        if target_w * target_h > max_pixels:
            target_w, target_h = _scale_to_max_pixels(target_w, target_h, max_pixels)
    else:
        target_w, target_h = _scale_to_max_pixels(w_ref, h_ref, max_pixels)

    if resolution_type == "fixed_sizes" and "supported_sizes_list" in meta:
        w_final, h_final = _closest_fixed_size(
            target_w, target_h, meta["supported_sizes_list"], max_pixels=None,
        )
        return {"height": h_final, "width": w_final}

    w_final, h_final = _snap_to_stride(target_w, target_h, model_stride)
    while w_final * h_final > max_pixels and (w_final > model_stride or h_final > model_stride):
        if w_final >= h_final:
            w_final = max(model_stride, w_final - model_stride)
        else:
            h_final = max(model_stride, h_final - model_stride)
        w_final, h_final = _snap_to_stride(w_final, h_final, model_stride)

    return {"height": h_final, "width": w_final}
