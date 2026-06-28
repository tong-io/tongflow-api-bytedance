from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tongflow.node_slots import NodeSlots
from tongflow.protocol import Asset, asset
from tongflow.slots import node_slot
from tongflow.models.text_gen_video import TextGenVideoInput, TextGenVideoOutput
from tongflow.models.image_gen_video import ImageGenVideoInput, ImageGenVideoOutput
from tongflow.models.image_image_gen_video import (
    ImageImageGenVideoInput,
    ImageImageGenVideoOutput,
)
from tongflow.models.audio_image_gen_video import (
    AudioImageGenVideoInput,
    AudioImageGenVideoOutput,
)
from tongflow.models.images_gen_video import (
    ImagesGenVideoInput,
    ImagesGenVideoOutput,
)

# Plugin logs go to stderr — stdout is reserved for the ABI JSON response.
# Level can be tuned via `TONGFLOW_PLUGIN_LOG_LEVEL` (default INFO).
logging.basicConfig(
    level=os.environ.get("TONGFLOW_PLUGIN_LOG_LEVEL", "INFO").upper(),
    stream=sys.stderr,
    format="[bytedance] %(levelname)s %(message)s",
)
log = logging.getLogger("tongflow.plugins.bytedance")


# Volcengine Ark (Doubao Seedance 2.0). The model is a plugin-internal knob,
# not an ABI field — overridable via env. Default to the lowest-cost tier.
DEFAULT_MODEL = "doubao-seedance-2-0-mini-260615"
DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_RESOLUTION = "720p"  # Mini supports only 480p / 720p
DEFAULT_RATIO = "adaptive"
DEFAULT_POLL_TIMEOUT_S = 600.0
POLL_INTERVAL_S = 10.0

# Ark ratio buckets (width / height) used to snap an explicit width×height.
_RATIOS: Dict[str, float] = {
    "21:9": 21 / 9,
    "16:9": 16 / 9,
    "4:3": 4 / 3,
    "1:1": 1.0,
    "3:4": 3 / 4,
    "9:16": 9 / 16,
}


def _require_api_key() -> str:
    api_key = os.environ.get("ARK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ARK_API_KEY is not set. Create one in the Volcengine Ark console "
            "and add it in TongFlow Settings."
        )
    return api_key


def _resolve_model() -> str:
    return (os.environ.get("SEEDANCE_MODEL") or "").strip() or DEFAULT_MODEL


def _base_url() -> str:
    return (os.environ.get("SEEDANCE_BASE_URL") or "").strip() or DEFAULT_BASE_URL


def _resolution() -> str:
    return (os.environ.get("SEEDANCE_RESOLUTION") or "").strip() or DEFAULT_RESOLUTION


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _poll_timeout() -> float:
    raw = (os.environ.get("SEEDANCE_POLL_TIMEOUT_S") or "").strip()
    try:
        return float(raw) if raw else DEFAULT_POLL_TIMEOUT_S
    except ValueError:
        return DEFAULT_POLL_TIMEOUT_S


def _ratio_from_wh(width: int | None, height: int | None) -> str:
    """Snap an explicit width×height to the nearest Ark ratio bucket.

    Falls back to ``SEEDANCE_RATIO`` (default ``adaptive``) when either side is
    missing.
    """
    if not width or not height or height <= 0:
        return (os.environ.get("SEEDANCE_RATIO") or "").strip() or DEFAULT_RATIO
    target = width / height
    return min(_RATIOS, key=lambda r: abs(_RATIOS[r] - target))


def _data_url(a: Asset, *, default_mime: str) -> str:
    """Build a base64 data URL for an Asset (Ark accepts these for image/audio)."""
    mime = (a.mime or default_mime).strip() or default_mime
    return f"data:{mime};base64,{a.bytesBase64}"


# ── Ark HTTP helpers ───────────────────────────────────────────────────────


def _request(method: str, path: str, body: Dict[str, Any] | None = None) -> Dict[str, Any]:
    url = _base_url().rstrip("/") + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {
        "Authorization": f"Bearer {_require_api_key()}",
        "Content-Type": "application/json",
    }
    log.info("%s %s", method, path)
    req = Request(url, data=data, headers=headers, method=method)
    try:
        resp = urlopen(req, timeout=180)  # noqa: S310
    except HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
        log.error("HTTP %s on %s\nresponse body: %s", e.code, path, err_body)
        raise RuntimeError(f"HTTP {e.code} from Ark: {err_body or e.reason}") from e
    except URLError as e:
        log.error("network error contacting %s: %s", path, e.reason)
        raise RuntimeError(f"Network error: {e.reason}") from e
    raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw) if raw.strip() else {}


def _create_task(content: List[Dict[str, Any]], **top_params: Any) -> str:
    body: Dict[str, Any] = {
        "model": _resolve_model(),
        "content": content,
        "generate_audio": _env_bool("SEEDANCE_GENERATE_AUDIO", True),
        "resolution": _resolution(),
        "watermark": _env_bool("SEEDANCE_WATERMARK", False),
    }
    for key, val in top_params.items():
        if val is not None:
            body[key] = val
    obj = _request("POST", "/contents/generations/tasks", body)
    task_id = obj.get("id")
    if not isinstance(task_id, str) or not task_id:
        raise RuntimeError(f"Ark create-task returned no id: {obj}")
    return task_id


def _poll_task(task_id: str) -> str:
    """Poll until succeeded, then return the video URL."""
    deadline = time.monotonic() + _poll_timeout()
    while True:
        obj = _request("GET", f"/contents/generations/tasks/{task_id}")
        status = obj.get("status")
        if status == "succeeded":
            video_url = (obj.get("content") or {}).get("video_url")
            if not isinstance(video_url, str) or not video_url:
                raise RuntimeError(f"Ark task {task_id} succeeded without video_url")
            return video_url
        if status == "failed":
            err = obj.get("error") or {}
            msg = err.get("message") if isinstance(err, dict) else err
            raise RuntimeError(f"Ark task {task_id} failed: {msg or 'unknown error'}")
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Ark task {task_id} did not finish within "
                f"{int(_poll_timeout())}s (last status: {status})"
            )
        time.sleep(POLL_INTERVAL_S)


def _download(url: str) -> bytes:
    """Download the produced mp4 (Ark video URLs are valid for ~24h only)."""
    log.info("downloading result video")
    try:
        resp = urlopen(url, timeout=180)  # noqa: S310
    except HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} downloading video: {e.reason}") from e
    except URLError as e:
        raise RuntimeError(f"Network error downloading video: {e.reason}") from e
    return resp.read()


def _generate(content: List[Dict[str, Any]], **top_params: Any) -> Asset:
    task_id = _create_task(content, **top_params)
    video_url = _poll_task(task_id)
    return asset(_download(video_url), mime="video/mp4")


def _duration_seconds(value: float | None) -> int | None:
    if value is None:
        return None
    return int(math.floor(value))


# ── ABI slot handlers ──────────────────────────────────────────────────────


@node_slot(NodeSlots.TEXT_GEN_VIDEO)
def text_gen_video(input: TextGenVideoInput) -> TextGenVideoOutput:
    text = (input.text or "").strip()
    if not text:
        return TextGenVideoOutput(success=False, error="Missing text prompt")
    content = [{"type": "text", "text": text}]
    video = _generate(
        content,
        ratio=_ratio_from_wh(input.width, input.height),
        duration=_duration_seconds(input.duration),
    )
    return TextGenVideoOutput(success=True, video=video)


@node_slot(NodeSlots.IMAGE_GEN_VIDEO)
def image_gen_video(input: ImageGenVideoInput) -> ImageGenVideoOutput:
    text = (input.text or "").strip()
    if not text:
        return ImageGenVideoOutput(success=False, error="Missing text prompt")
    content = [
        {"type": "text", "text": text},
        {
            "type": "image_url",
            "image_url": {"url": _data_url(input.image, default_mime="image/png")},
            "role": "first_frame",
        },
    ]
    video = _generate(
        content,
        ratio=_ratio_from_wh(input.width, input.height),
        duration=_duration_seconds(input.duration),
    )
    return ImageGenVideoOutput(success=True, video=video)


@node_slot(NodeSlots.IMAGE_IMAGE_GEN_VIDEO)
def image_image_gen_video(input: ImageImageGenVideoInput) -> ImageImageGenVideoOutput:
    text = (input.text or "").strip()
    if not text:
        return ImageImageGenVideoOutput(success=False, error="Missing text prompt")
    content = [
        {"type": "text", "text": text},
        {
            "type": "image_url",
            "image_url": {"url": _data_url(input.image, default_mime="image/png")},
            "role": "first_frame",
        },
        {
            "type": "image_url",
            "image_url": {"url": _data_url(input.end_image, default_mime="image/png")},
            "role": "last_frame",
        },
    ]
    video = _generate(
        content,
        ratio=_ratio_from_wh(input.width, input.height),
        duration=_duration_seconds(input.duration),
    )
    return ImageImageGenVideoOutput(success=True, video=video)


@node_slot(NodeSlots.AUDIO_IMAGE_GEN_VIDEO)
def audio_image_gen_video(input: AudioImageGenVideoInput) -> AudioImageGenVideoOutput:
    content: List[Dict[str, Any]] = []
    text = (input.text or "").strip()
    if text:
        content.append({"type": "text", "text": text})
    content.append(
        {
            "type": "image_url",
            "image_url": {"url": _data_url(input.image, default_mime="image/png")},
            "role": "reference_image",
        }
    )
    content.append(
        {
            "type": "audio_url",
            "audio_url": {"url": _data_url(input.audio, default_mime="audio/mpeg")},
            "role": "reference_audio",
        }
    )
    # No ABI duration field here — let the model pick (audio total must be <= 15s).
    video = _generate(content, ratio=_ratio_from_wh(input.width, input.height))
    return AudioImageGenVideoOutput(success=True, video=video)


@node_slot(NodeSlots.IMAGES_GEN_VIDEO)
def images_gen_video(input: ImagesGenVideoInput) -> ImagesGenVideoOutput:
    # Free multi-image reference fusion: every image is a Seedance multimodal
    # reference (role=reference_image), combined with the text into a new video.
    text = (input.text or "").strip()
    if not text:
        return ImagesGenVideoOutput(success=False, error="Missing text prompt")
    content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    for img in input.images or []:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _data_url(img, default_mime="image/png")},
                "role": "reference_image",
            }
        )
    video = _generate(
        content,
        ratio=_ratio_from_wh(input.width, input.height),
        duration=_duration_seconds(input.duration),
    )
    return ImagesGenVideoOutput(success=True, video=video)


# Runtime dispatcher. The @node_slot wrapper accepts a raw dict here (it
# deep-constructs the typed BaseModel internally) and dumps the BaseModel
# return to a dict. `Any` reflects the I/O boundary, not the plugin contract.
_SLOT_HANDLERS: Dict[str, Any] = {
    NodeSlots.TEXT_GEN_VIDEO: text_gen_video,
    NodeSlots.IMAGE_GEN_VIDEO: image_gen_video,
    NodeSlots.IMAGE_IMAGE_GEN_VIDEO: image_image_gen_video,
    NodeSlots.AUDIO_IMAGE_GEN_VIDEO: audio_image_gen_video,
    NodeSlots.IMAGES_GEN_VIDEO: images_gen_video,
}


def _write(out: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(out, ensure_ascii=False))
    sys.stdout.flush()


def main() -> int:
    try:
        raw = sys.stdin.read()
        req = json.loads(raw) if raw.strip() else {}
        prompt = req.get("prompt") if isinstance(req, dict) else {}
        if not isinstance(prompt, dict):
            prompt = {}
        slot = str(req.get("nodeSlot") or "") if isinstance(req, dict) else ""

        handler = _SLOT_HANDLERS.get(slot)
        if handler is None:
            raise RuntimeError(f"unsupported nodeSlot: {slot!r}")
        out = handler(prompt)
    except Exception as e:  # noqa: BLE001 — surfaced as ABI failure
        _write({"success": False, "error": str(e)})
        return 1

    _write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
