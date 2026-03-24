import os
import time
import json
import base64
import yaml
import mimetypes
import requests
from io import BytesIO
from PIL import Image
from typing import Optional, Tuple, Dict, Any


def load_config(config_path: str) -> dict:
    """Load evaluation config from YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _try_convert_raster_to_png_bytes(image_data: bytes) -> bytes | None:
    """
    Best-effort: convert arbitrary raster bytes (e.g. webp/avif) into PNG bytes via PIL.
    Returns PNG bytes on success; None on failure.
    """
    try:
        im = Image.open(BytesIO(image_data))
        im.load()
        # Avoid palette/LA/etc compatibility surprises
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGBA")
        buf = BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def _normalize_mime_type(mime_type: str | None, default: str = "image/png") -> str:
    """
    Normalize MIME type for data URL usage.
    - Drops parameters like "; charset=utf-8"
    - Strips whitespace
    - Lower-cases
    """
    if not mime_type:
        return default
    mt = str(mime_type).split(";", 1)[0].strip().lower()
    return mt or default

def _load_image_bytes_and_mime(path_url_or_pil) -> Tuple[bytes, str]:
    """
    Load an image from local path / URL / PIL / bytes and return (raw_bytes, mime_type).
    Gemini inline_data expects base64 of raw bytes, not a data URL string.

    Notes:
    - Best-effort converts webp/avif to PNG to reduce provider-specific image parsing failures.
    - Returns mime_type without parameters (e.g. "image/png").
    """
    image_data = None
    mime_type = "image/png"

    if isinstance(path_url_or_pil, Image.Image):
        buf = BytesIO()
        path_url_or_pil.save(buf, format="PNG")
        image_data = buf.getvalue()
        mime_type = "image/png"

    elif isinstance(path_url_or_pil, BytesIO):
        image_data = path_url_or_pil.getvalue()
        mime_type = "image/png"

    elif isinstance(path_url_or_pil, bytes):
        image_data = path_url_or_pil
        mime_type = "image/png"

    elif isinstance(path_url_or_pil, str):
        if os.path.exists(path_url_or_pil):
            with open(path_url_or_pil, "rb") as f:
                image_data = f.read()
            mt, _ = mimetypes.guess_type(path_url_or_pil)
            mime_type = _normalize_mime_type(mt, default="image/png")
        elif path_url_or_pil.startswith("http"):
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/115.0.0.0 Safari/537.36"
            }
            resp = requests.get(path_url_or_pil, headers=headers)
            if resp.status_code != 200:
                raise ValueError(f"Failed to download image: {resp.status_code}")
            image_data = resp.content
            mime_type = _normalize_mime_type(resp.headers.get("Content-Type", None), default="image/png")
            if not mime_type.startswith("image/"):
                raise ValueError(f"URL did not return an image (Content-Type={resp.headers.get('Content-Type', None)})")
        else:
            raise ValueError("Unsupported string format: must be file path or URL")
    else:
        raise TypeError("Unsupported input type for load_image")

    assert isinstance(image_data, (bytes, bytearray))

    # Convert formats that frequently fail in provider vision endpoints.
    if mime_type in {"image/webp", "image/avif"}:
        converted = _try_convert_raster_to_png_bytes(bytes(image_data))
        if converted is not None:
            image_data = converted
            mime_type = "image/png"

    return bytes(image_data), mime_type


def _to_gemini_inline_data_part(path_url_or_pil) -> Dict[str, Any]:
    image_data, mime_type = _load_image_bytes_and_mime(path_url_or_pil)
    b64 = base64.b64encode(image_data).decode("utf-8")
    return {"inline_data": {"mime_type": mime_type, "data": b64}}



# ---------------------------------------------------------------------------
# Provider: Google (google-genai SDK)
# ---------------------------------------------------------------------------

def _create_google_client(api_key: str):
    from google import genai
    return genai.Client(api_key=api_key)


def _request_google(client, model, img_path, user_prompt, system_prompt,
                    *, max_retries, sleep_time, timeout, debug):
    from google.genai import types
    image_part = _to_gemini_inline_data_part(img_path)
    
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.0,
        http_options=types.HttpOptions(timeout=timeout * 1000),
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model, contents=[image_part, user_prompt], config=config,
            )
            text = response.text or ""
            if debug:
                print(f"  [DEBUG] Google response ({len(text)} chars): {text[:200]}", flush=True)
            return text, True
        except Exception as e:
            err_str = str(e)
            if _is_retryable(err_str) and attempt < max_retries:
                wait = sleep_time * attempt
                if debug:
                    print(f"  [DEBUG] Attempt {attempt}/{max_retries} failed: {err_str[:120]}. Retrying in {wait}s ...", flush=True)
                time.sleep(wait)
                continue
            return f"google error after {attempt} attempts: {err_str[:300]}", False
    return "max retries exceeded", False


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_RETRYABLE_KEYWORDS = [
    "429", "rate", "quota", "500", "503", "504",
    "unavailable", "overloaded", "timeout", "timed out", "deadline",
    "ssl", "connection", "reset", "broken pipe",
]

def _is_retryable(err_str: str) -> bool:
    lower = err_str.lower()
    return any(kw in lower for kw in _RETRYABLE_KEYWORDS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_gemini_client(config: dict):
    """Create an official Google Gemini client.

    Supported provider:
      - "google" (default): uses google-genai SDK, needs GEMINI_API_KEY or GOOGLE_API_KEY
    """
    provider = (config.get("provider") or "google").strip().lower()
    if provider and provider != "google":
        raise ValueError(f"Unsupported Gemini provider: {provider}. Only 'google' is supported.")

    api_key = (
        config.get("api_key")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY", "")
    )
    if not api_key:
        raise ValueError(
            "Gemini API key not found. Set it in config YAML (api_key), "
            "or via GEMINI_API_KEY / GOOGLE_API_KEY environment variable."
        )
    client = _create_google_client(api_key)
    client._provider = "google"
    return client


def request_gemini_i2t(
    client,
    model: str,
    img_path: str,
    user_prompt: str,
    system_prompt: str,
    *,
    max_retries: int = 5,
    sleep_time: int = 5,
    timeout: int = 600,
    debug: bool = False,
) -> tuple[str, bool]:
    """Send an image + text prompt to Gemini and return (response_text, success).

    Uses the official Google Gemini SDK client.
    """
    kwargs = dict(max_retries=max_retries, sleep_time=sleep_time, timeout=timeout, debug=debug)
    return _request_google(client, model, img_path, user_prompt, system_prompt, **kwargs)
