from __future__ import annotations

import base64
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
from tongflow.models.gen_text import GenTextInput, GenTextOutput
from tongflow.models.split_text import SplitTextInput, SplitTextOutput
from tongflow.models.combine_text import CombineTextInput, CombineTextOutput
from tongflow.models.image_gen_text import ImageGenTextInput, ImageGenTextOutput
from tongflow.models.image_describe import ImageDescribeInput, ImageDescribeOutput
from tongflow.models.video_gen_text import VideoGenTextInput, VideoGenTextOutput
from tongflow.models.video_describe import VideoDescribeInput, VideoDescribeOutput
from tongflow.models.image_gen import ImageGenInput, ImageGenOutput
from tongflow.models.image_edit import ImageEditInput, ImageEditOutput
from tongflow.models.image_fusion import ImageFusionInput, ImageFusionOutput
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
from tongflow.models.refs_gen_video import (
    RefsGenVideoInput,
    RefsGenVideoOutput,
)
from tongflow.models.drop_video import DropVideoInput, DropVideoOutput
from tongflow.models.arrange_group import ArrangeGroupInput, ArrangeGroupOutput
from tongflow.llm_batch_handlers import arrange_group_output, drop_video_output


# ── Per-node model picker ───────────────────────────────────────────────────
# Pure dict literal read by the platform scanner via AST. First entry = default.
# Model ids verified 2026-07; Ark dated snapshots (-25xxxx/-26xxxx) rotate and
# each must be enabled on the account — re-check the live Ark model list before
# release. A user may also paste an `ep-...` endpoint id, which is passed through
# unchanged. ASR / TTS live on Volcengine's separate speech service (different
# key + transport) and are intentionally out of scope here.
# NOTE: the scanner reads this by AST without importing the module, so every
# value MUST be a pure list-of-string literal — no variable references, no
# shared aliases. The repetition below is intentional (Doubao LLM / Doubao
# vision / Seedream / Seedance families).
TONGFLOW_SLOT_MODELS = {
    "gen-text": ["doubao-seed-1-6-250615", "doubao-seed-2-0-pro-260215", "doubao-seed-1-6-flash-250615", "doubao-1-5-pro-32k", "doubao-1-5-lite-32k"],
    "split-text": ["doubao-seed-1-6-250615", "doubao-seed-2-0-pro-260215", "doubao-seed-1-6-flash-250615", "doubao-1-5-pro-32k", "doubao-1-5-lite-32k"],
    "combine-text": ["doubao-seed-1-6-250615", "doubao-seed-2-0-pro-260215", "doubao-seed-1-6-flash-250615", "doubao-1-5-pro-32k", "doubao-1-5-lite-32k"],
    "image-gen-text": ["doubao-seed-1-6-250615", "doubao-seed-1-6-vision-250815", "doubao-seed-2-0-pro-260215", "doubao-1-5-vision-pro-32k"],
    "image-describe": ["doubao-seed-1-6-250615", "doubao-seed-1-6-vision-250815", "doubao-seed-2-0-pro-260215", "doubao-1-5-vision-pro-32k"],
    "video-gen-text": ["doubao-seed-1-6-250615", "doubao-seed-1-6-vision-250815", "doubao-seed-2-0-pro-260215", "doubao-1-5-vision-pro-32k"],
    "video-describe": ["doubao-seed-1-6-250615", "doubao-seed-1-6-vision-250815", "doubao-seed-2-0-pro-260215", "doubao-1-5-vision-pro-32k"],
    "image-gen": ["doubao-seedream-4-0-250828", "doubao-seedream-4-5-251128", "doubao-seedream-5-0-260128"],
    "image-edit": ["doubao-seedream-4-0-250828", "doubao-seedream-4-5-251128", "doubao-seedream-5-0-260128"],
    "image-fusion": ["doubao-seedream-4-0-250828", "doubao-seedream-4-5-251128", "doubao-seedream-5-0-260128"],
    "text-gen-video": ["doubao-seedance-2-0-mini-260615", "doubao-seedance-2-0-260128", "doubao-seedance-2-0-fast-260128"],
    "image-gen-video": ["doubao-seedance-2-0-mini-260615", "doubao-seedance-2-0-260128", "doubao-seedance-2-0-fast-260128"],
    "image-image-gen-video": ["doubao-seedance-2-0-mini-260615", "doubao-seedance-2-0-260128", "doubao-seedance-2-0-fast-260128"],
    "audio-image-gen-video": ["doubao-seedance-2-0-mini-260615", "doubao-seedance-2-0-260128", "doubao-seedance-2-0-fast-260128"],
    "images-gen-video": ["doubao-seedance-2-0-mini-260615", "doubao-seedance-2-0-260128", "doubao-seedance-2-0-fast-260128"],
    "refs-gen-video": ["doubao-seedance-2-0-mini-260615", "doubao-seedance-2-0-260128", "doubao-seedance-2-0-fast-260128"],
}

# Slots this plugin is the default implementation of.
TONGFLOW_DEFAULT_SLOTS = [
    "text-gen-video",
    "image-gen-video",
    "image-image-gen-video",
    "images-gen-video",
]

# Set from the request envelope's top-level `model` field in main().
_REQUEST_MODEL: str = ""

# Plugin logs go to stderr — stdout is reserved for the ABI JSON response.
logging.basicConfig(
    level=os.environ.get("TONGFLOW_PLUGIN_LOG_LEVEL", "INFO").upper(),
    stream=sys.stderr,
    format="[bytedance] %(levelname)s %(message)s",
)
log = logging.getLogger("tongflow.plugins.bytedance")


DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_RESOLUTION = "720p"  # Mini supports only 480p / 720p
DEFAULT_RATIO = "adaptive"
DEFAULT_POLL_TIMEOUT_S = 600.0
POLL_INTERVAL_S = 10.0

# Ark ratio buckets (width / height) used to snap an explicit width×height.
_RATIOS: Dict[str, float] = {
    "21:9": 21 / 9, "16:9": 16 / 9, "4:3": 4 / 3, "1:1": 1.0, "3:4": 3 / 4, "9:16": 9 / 16,
}


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def _active_model(slot: str, env_override: str = "") -> str:
    """Resolve the model for a slot: per-node pick > legacy env override >
    list default. An `ep-...` endpoint id is accepted verbatim."""
    models = TONGFLOW_SLOT_MODELS[slot]
    if _REQUEST_MODEL:
        if _REQUEST_MODEL.startswith("ep-"):
            return _REQUEST_MODEL
        if _REQUEST_MODEL not in models:
            raise RuntimeError(
                f"unknown model {_REQUEST_MODEL!r} for {slot}; available: {', '.join(models)}"
            )
        return _REQUEST_MODEL
    if env_override:
        return env_override
    return models[0]


def _require_api_key() -> str:
    api_key = os.environ.get("ARK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ARK_API_KEY is not set. Create one in the Volcengine Ark console "
            "and add it in TongFlow Settings."
        )
    return api_key


def _base_url() -> str:
    return _env("SEEDANCE_BASE_URL") or DEFAULT_BASE_URL


def _resolution() -> str:
    return _env("SEEDANCE_RESOLUTION") or DEFAULT_RESOLUTION


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _poll_timeout() -> float:
    raw = _env("SEEDANCE_POLL_TIMEOUT_S")
    try:
        return float(raw) if raw else DEFAULT_POLL_TIMEOUT_S
    except ValueError:
        return DEFAULT_POLL_TIMEOUT_S


def _ratio_from_wh(width: int | None, height: int | None) -> str:
    if not width or not height or height <= 0:
        return _env("SEEDANCE_RATIO") or DEFAULT_RATIO
    target = width / height
    return min(_RATIOS, key=lambda r: abs(_RATIOS[r] - target))


def _data_url(a: Asset, *, default_mime: str) -> str:
    mime = (a.mime or default_mime).strip() or default_mime
    return f"data:{mime};base64,{a.bytesBase64}"


# ── Ark HTTP helpers ───────────────────────────────────────────────────────


def _request(method: str, path: str, body: Dict[str, Any] | None = None, *, timeout: int = 180) -> Dict[str, Any]:
    url = _base_url().rstrip("/") + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {
        "Authorization": f"Bearer {_require_api_key()}",
        "Content-Type": "application/json",
    }
    log.info("%s %s", method, path)
    req = Request(url, data=data, headers=headers, method=method)
    try:
        resp = urlopen(req, timeout=timeout)  # noqa: S310
    except HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
        log.error("HTTP %s on %s\nresponse body: %s", e.code, path, err_body)
        raise RuntimeError(f"HTTP {e.code} from Ark: {err_body or e.reason}") from e
    except URLError as e:
        raise RuntimeError(f"Network error: {e.reason}") from e
    raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw) if raw.strip() else {}


def _download(url: str) -> bytes:
    try:
        resp = urlopen(url, timeout=180)  # noqa: S310
    except HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} downloading media: {e.reason}") from e
    except URLError as e:
        raise RuntimeError(f"Network error downloading media: {e.reason}") from e
    return resp.read()


# ── Doubao chat (LLM + vision) ─────────────────────────────────────────────


def _chat(*, model: str, messages: List[Dict[str, Any]], json_mode: bool = False) -> str:
    body: Dict[str, Any] = {"model": model, "messages": messages}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    obj = _request("POST", "/chat/completions", body)
    choices = obj.get("choices") or []
    if not choices:
        raise RuntimeError("Ark chat response missing choices")
    content = ((choices[0] or {}).get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Ark chat response missing message.content")
    return content.strip()


def _chat_text(*, model: str, user_message: str, json_mode: bool = False) -> str:
    return _chat(
        model=model,
        messages=[
            {"role": "system", "content": "You are a versatile assistant. Follow the user's instructions strictly and respond in the same language as the user."},
            {"role": "user", "content": user_message},
        ],
        json_mode=json_mode,
    )


# ── Seedream image (/images/generations) ────────────────────────────────────


def _generate_image(*, model: str, prompt: str, images: List[Asset]) -> Asset:
    body: Dict[str, Any] = {"model": model, "prompt": prompt}
    size = _env("SEEDANCE_IMAGE_SIZE")
    if size:
        body["size"] = size
    if len(images) == 1:
        body["image"] = _data_url(images[0], default_mime="image/png")
    elif len(images) > 1:
        body["image"] = [_data_url(i, default_mime="image/png") for i in images]
        body["sequential_image_generation"] = "disabled"
    obj = _request("POST", "/images/generations", body, timeout=300)
    data = obj.get("data") or []
    if not data:
        raise RuntimeError(f"Ark image response missing data: {obj}")
    item = data[0] or {}
    b64 = item.get("b64_json")
    if isinstance(b64, str) and b64:
        return asset(base64.b64decode(b64), mime="image/png")
    url = item.get("url")
    if isinstance(url, str) and url:
        return asset(_download(url), mime="image/png")
    raise RuntimeError("Ark image response missing url/b64_json")


# ── Seedance video (async task) ─────────────────────────────────────────────


def _create_task(model: str, content: List[Dict[str, Any]], **top_params: Any) -> str:
    body: Dict[str, Any] = {
        "model": model,
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
                f"Ark task {task_id} did not finish within {int(_poll_timeout())}s (last status: {status})"
            )
        time.sleep(POLL_INTERVAL_S)


def _generate_video(model: str, content: List[Dict[str, Any]], **top_params: Any) -> Asset:
    task_id = _create_task(model, content, **top_params)
    video_url = _poll_task(task_id)
    return asset(_download(video_url), mime="video/mp4")


def _duration_seconds(value: float | None) -> int | None:
    if value is None:
        return None
    return int(math.floor(value))


# ── Text slots ──────────────────────────────────────────────────────────────


@node_slot(NodeSlots.GEN_TEXT)
def gen_text(input: GenTextInput) -> GenTextOutput:
    user_message = (
        f"{input.userPrompt or ''}\n\nUser input: {input.text}\n\n"
        "Note: output only the requested answer. Do not include any other content."
    )
    answer = _chat_text(model=_active_model("gen-text", _env("DOUBAO_MODEL")), user_message=user_message)
    return GenTextOutput(success=True, text=answer)


def _parse_split_texts(raw: str) -> list[str]:
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if "\n" in s:
            _, _, s = s.partition("\n")
        s = s.strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    obj = json.loads(s)
    if isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict):
        raw_items = obj.get("texts")
        items = raw_items if isinstance(raw_items, list) else None
    else:
        items = None
    if items is None or not all(isinstance(x, str) for x in items):
        raise ValueError("LLM did not return a JSON array of strings")
    cleaned = [x.strip() for x in items if x.strip()]
    if not cleaned:
        raise ValueError("LLM returned an empty split")
    return cleaned


@node_slot(NodeSlots.SPLIT_TEXT)
def split_text(input: SplitTextInput) -> SplitTextOutput:
    instruction = (input.userPrompt or "").strip() or "Split into natural, coherent segments."
    user_message = (
        f"Split the following text into multiple segments according to this instruction:\n{instruction}\n\n"
        'Return ONLY a JSON object of the form {"texts": ["segment 1", "segment 2", ...]} — no prose, no markdown. '
        f"Preserve the original wording; do not summarize.\n\nTEXT:\n{input.text}"
    )
    raw = _chat_text(
        model=_active_model("split-text", _env("DOUBAO_MODEL")), user_message=user_message, json_mode=True
    )
    try:
        texts = _parse_split_texts(raw)
    except (ValueError, json.JSONDecodeError) as e:
        return SplitTextOutput(success=False, error=str(e))
    return SplitTextOutput(success=True, texts=texts)


@node_slot(NodeSlots.COMBINE_TEXT)
def combine_text(input: CombineTextInput) -> CombineTextOutput:
    joined = "\n\n".join(input.texts)
    user_message = (
        f"{input.userPrompt or ''}\n\nUser input: {joined}\n\n"
        "Note: output only the requested answer. Do not include any other content."
    )
    answer = _chat_text(model=_active_model("combine-text", _env("DOUBAO_MODEL")), user_message=user_message)
    return CombineTextOutput(success=True, text=answer)


# ── Vision → text slots ─────────────────────────────────────────────────────


@node_slot(NodeSlots.IMAGE_GEN_TEXT)
def image_gen_text(input: ImageGenTextInput) -> ImageGenTextOutput:
    content: List[Dict[str, Any]] = [{"type": "text", "text": input.text}]
    if input.image is not None:
        content.append({"type": "image_url", "image_url": {"url": _data_url(input.image, default_mime="image/png")}})
    messages: List[Dict[str, Any]] = []
    if input.system:
        messages.append({"role": "system", "content": input.system})
    messages.append({"role": "user", "content": content})
    text = _chat(model=_active_model("image-gen-text", _env("DOUBAO_VISION_MODEL")), messages=messages)
    return ImageGenTextOutput(success=True, text=text)


@node_slot(NodeSlots.IMAGE_DESCRIBE)
def image_describe(input: ImageDescribeInput) -> ImageDescribeOutput:
    instruction = (input.userPrompt or "").strip() or (input.text or "").strip() or "Describe this image in detail."
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": _data_url(input.image, default_mime="image/png")}},
            ],
        }
    ]
    text = _chat(model=_active_model("image-describe", _env("DOUBAO_VISION_MODEL")), messages=messages)
    return ImageDescribeOutput(success=True, text=text)


@node_slot(NodeSlots.VIDEO_GEN_TEXT)
def video_gen_text(input: VideoGenTextInput) -> VideoGenTextOutput:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": input.text},
                {"type": "video_url", "video_url": {"url": _data_url(input.video, default_mime="video/mp4")}},
            ],
        }
    ]
    text = _chat(model=_active_model("video-gen-text", _env("DOUBAO_VISION_MODEL")), messages=messages)
    return VideoGenTextOutput(success=True, text=text)


@node_slot(NodeSlots.VIDEO_DESCRIBE)
def video_describe(input: VideoDescribeInput) -> VideoDescribeOutput:
    instruction = (input.text or "").strip() or "Describe this video in detail."
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "video_url", "video_url": {"url": _data_url(input.video, default_mime="video/mp4")}},
            ],
        }
    ]
    text = _chat(model=_active_model("video-describe", _env("DOUBAO_VISION_MODEL")), messages=messages)
    return VideoDescribeOutput(success=True, text=text)


# ── Seedream image slots ────────────────────────────────────────────────────


@node_slot(NodeSlots.IMAGE_GEN)
def image_gen(input: ImageGenInput) -> ImageGenOutput:
    prompt = (input.text or "").strip()
    if not prompt:
        return ImageGenOutput(success=False, error="image-gen requires a text prompt")
    image = _generate_image(model=_active_model("image-gen", _env("SEEDREAM_MODEL")), prompt=prompt, images=[])
    return ImageGenOutput(success=True, image=image)


@node_slot(NodeSlots.IMAGE_EDIT)
def image_edit(input: ImageEditInput) -> ImageEditOutput:
    image = _generate_image(
        model=_active_model("image-edit", _env("SEEDREAM_MODEL")), prompt=input.text, images=[input.image]
    )
    return ImageEditOutput(success=True, image=image)


@node_slot(NodeSlots.IMAGE_FUSION)
def image_fusion(input: ImageFusionInput) -> ImageFusionOutput:
    images = list(input.images or [])
    if not images:
        return ImageFusionOutput(success=False, error="image-fusion requires at least one input image")
    image = _generate_image(
        model=_active_model("image-fusion", _env("SEEDREAM_MODEL")), prompt=input.text, images=images
    )
    return ImageFusionOutput(success=True, image=image)


# ── Seedance video slots ────────────────────────────────────────────────────


@node_slot(NodeSlots.TEXT_GEN_VIDEO)
def text_gen_video(input: TextGenVideoInput) -> TextGenVideoOutput:
    text = (input.text or "").strip()
    if not text:
        return TextGenVideoOutput(success=False, error="Missing text prompt")
    video = _generate_video(
        _active_model("text-gen-video", _env("SEEDANCE_MODEL")),
        [{"type": "text", "text": text}],
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
        {"type": "image_url", "image_url": {"url": _data_url(input.image, default_mime="image/png")}, "role": "first_frame"},
    ]
    video = _generate_video(
        _active_model("image-gen-video", _env("SEEDANCE_MODEL")),
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
        {"type": "image_url", "image_url": {"url": _data_url(input.image, default_mime="image/png")}, "role": "first_frame"},
        {"type": "image_url", "image_url": {"url": _data_url(input.end_image, default_mime="image/png")}, "role": "last_frame"},
    ]
    video = _generate_video(
        _active_model("image-image-gen-video", _env("SEEDANCE_MODEL")),
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
        {"type": "image_url", "image_url": {"url": _data_url(input.image, default_mime="image/png")}, "role": "reference_image"}
    )
    content.append(
        {"type": "audio_url", "audio_url": {"url": _data_url(input.audio, default_mime="audio/mpeg")}, "role": "reference_audio"}
    )
    video = _generate_video(
        _active_model("audio-image-gen-video", _env("SEEDANCE_MODEL")),
        content,
        ratio=_ratio_from_wh(input.width, input.height),
    )
    return AudioImageGenVideoOutput(success=True, video=video)


@node_slot(NodeSlots.IMAGES_GEN_VIDEO)
def images_gen_video(input: ImagesGenVideoInput) -> ImagesGenVideoOutput:
    text = (input.text or "").strip()
    if not text:
        return ImagesGenVideoOutput(success=False, error="Missing text prompt")
    content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    for img in input.images or []:
        content.append(
            {"type": "image_url", "image_url": {"url": _data_url(img, default_mime="image/png")}, "role": "reference_image"}
        )
    video = _generate_video(
        _active_model("images-gen-video", _env("SEEDANCE_MODEL")),
        content,
        ratio=_ratio_from_wh(input.width, input.height),
        duration=_duration_seconds(input.duration),
    )
    return ImagesGenVideoOutput(success=True, video=video)


@node_slot(NodeSlots.REFS_GEN_VIDEO)
def refs_gen_video(input: RefsGenVideoInput) -> RefsGenVideoOutput:
    text = (input.text or "").strip()
    if not text:
        return RefsGenVideoOutput(success=False, error="Missing text prompt")
    images = input.images or []
    videos = input.videos or []
    audios = input.audios or []
    # Ark constraint mirrors the ABI contract: audio can never be the sole
    # reference, and the caps are 9 images / 3 videos / 3 audio clips.
    if not images and not videos:
        return RefsGenVideoOutput(
            success=False,
            error="At least one reference image or video is required; audio cannot be the only reference",
        )
    if len(images) > 9 or len(videos) > 3 or len(audios) > 3 or len(images) + len(videos) + len(audios) > 12:
        return RefsGenVideoOutput(
            success=False,
            error="Too many references: up to 9 images, 3 videos and 3 audio clips (12 files total)",
        )
    content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    for img in images:
        content.append(
            {"type": "image_url", "image_url": {"url": _data_url(img, default_mime="image/png")}, "role": "reference_image"}
        )
    for vid in videos:
        content.append(
            {"type": "video_url", "video_url": {"url": _data_url(vid, default_mime="video/mp4")}, "role": "reference_video"}
        )
    for aud in audios:
        content.append(
            {"type": "audio_url", "audio_url": {"url": _data_url(aud, default_mime="audio/mpeg")}, "role": "reference_audio"}
        )
    video = _generate_video(
        _active_model("refs-gen-video", _env("SEEDANCE_MODEL")),
        content,
        ratio=_ratio_from_wh(input.width, input.height),
        duration=_duration_seconds(input.duration),
    )
    return RefsGenVideoOutput(success=True, video=video)


# ── Deterministic batch slots (no model) ────────────────────────────────────


@node_slot(NodeSlots.DROP_VIDEO)
def drop_video(input: DropVideoInput) -> DropVideoOutput:
    result = drop_video_output(input.model_dump())
    return DropVideoOutput.model_construct(**result)


@node_slot(NodeSlots.ARRANGE_GROUP)
def arrange_group(input: ArrangeGroupInput) -> ArrangeGroupOutput:
    result = arrange_group_output(input.model_dump())
    return ArrangeGroupOutput.model_construct(**result)


# Runtime dispatcher. The @node_slot wrapper accepts a raw dict here (it
# deep-constructs the typed BaseModel internally) and dumps the BaseModel
# return to a dict. `Any` reflects the I/O boundary, not the plugin contract.
_SLOT_HANDLERS: Dict[str, Any] = {
    NodeSlots.GEN_TEXT: gen_text,
    NodeSlots.SPLIT_TEXT: split_text,
    NodeSlots.COMBINE_TEXT: combine_text,
    NodeSlots.IMAGE_GEN_TEXT: image_gen_text,
    NodeSlots.IMAGE_DESCRIBE: image_describe,
    NodeSlots.VIDEO_GEN_TEXT: video_gen_text,
    NodeSlots.VIDEO_DESCRIBE: video_describe,
    NodeSlots.IMAGE_GEN: image_gen,
    NodeSlots.IMAGE_EDIT: image_edit,
    NodeSlots.IMAGE_FUSION: image_fusion,
    NodeSlots.TEXT_GEN_VIDEO: text_gen_video,
    NodeSlots.IMAGE_GEN_VIDEO: image_gen_video,
    NodeSlots.IMAGE_IMAGE_GEN_VIDEO: image_image_gen_video,
    NodeSlots.AUDIO_IMAGE_GEN_VIDEO: audio_image_gen_video,
    NodeSlots.IMAGES_GEN_VIDEO: images_gen_video,
    NodeSlots.REFS_GEN_VIDEO: refs_gen_video,
    NodeSlots.DROP_VIDEO: drop_video,
    NodeSlots.ARRANGE_GROUP: arrange_group,
}


def _write(out: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(out, ensure_ascii=False))
    sys.stdout.flush()


def main() -> int:
    global _REQUEST_MODEL
    try:
        raw = sys.stdin.read()
        req = json.loads(raw) if raw.strip() else {}
        prompt = req.get("prompt") if isinstance(req, dict) else {}
        if not isinstance(prompt, dict):
            prompt = {}
        slot = str(req.get("nodeSlot") or "") if isinstance(req, dict) else ""
        _REQUEST_MODEL = str(req.get("model") or "").strip() if isinstance(req, dict) else ""

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
