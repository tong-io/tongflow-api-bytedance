# tongflow-api-bytedance

Official [TongFlow](https://github.com/tong-io/tongflow) plugin. Video generation via
[Volcengine Ark](https://www.volcengine.com/product/ark) — **Doubao Seedance 2.0** (ByteDance).

## Capabilities

Implements these ABI slots (runs locally as a Python process, no GPU; calls the Ark API):

- **Video generation** (`text-gen-video`) — video from a text prompt.
- **Image-to-video** (`image-gen-video`) — animate a still image (first frame) into motion.
- **First/last-frame video** (`image-image-gen-video`) — two key images interpolated into a clip.
- **Image + audio → video** (`audio-image-gen-video`) — multimodal reference: image + audio → video.
- **Images → video** (`images-gen-video`) — free multi-image reference fusion: up to 9 reference images + text → a new video.

Output includes audio by default (Seedance generates audio + video jointly).

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `ARK_API_KEY` | ✅ | Volcengine Ark API key. Create one in the [Ark console](https://console.volcengine.com/ark). Requires an enabled Seedance 2.0 model (account balance ≥ ¥200 or a resource pack). |
| `SEEDANCE_MODEL` | optional | Model id. Default `doubao-seedance-2-0-mini-260615`. Other tiers: `doubao-seedance-2-0-fast-260128`, `doubao-seedance-2-0-260128` (only the full model supports 1080p/4K). |
| `SEEDANCE_RESOLUTION` | optional | `480p` / `720p` (Mini/Fast) or up to `4k` (full). Default `720p`. |
| `SEEDANCE_RATIO` | optional | Fallback aspect ratio when a node has no explicit width/height (e.g. `16:9`). Default `adaptive`. |
| `SEEDANCE_GENERATE_AUDIO` | optional | `true`/`false`. Default `true`. |
| `SEEDANCE_WATERMARK` | optional | `true`/`false`. Default `false`. |
| `SEEDANCE_BASE_URL` | optional | Override the Ark endpoint. Default `https://ark.cn-beijing.volces.com/api/v3`. |
| `SEEDANCE_POLL_TIMEOUT_S` | optional | Max seconds to wait for a generation. Default `600`. |

Values are stored locally and take effect without a restart.

## Limitations

- **No real human faces.** Seedance rejects input images/audio containing real people's faces
  (content-safety审核). Use model-generated or non-photoreal assets.
- **Mini / Fast cap at 720p.** Use `SEEDANCE_MODEL=doubao-seedance-2-0-260128` for 1080p/4K.
- **Audio ≤ 15s** for `audio-image-gen-video`.
- **Up to 9 reference images** for `images-gen-video`.
- **Video-input slots are not supported yet** (`video-edit`, `video-image-gen-video-*`,
  `speech-video-gen-video`). The Ark API only accepts public URLs / asset IDs for video inputs
  (not base64), so these require object-storage upload — planned for a later release.

## How it works

Each slot creates an async Ark task (`POST /contents/generations/tasks`), polls
(`GET /contents/generations/tasks/{id}`) until `succeeded`, then downloads the result mp4 and
returns it as an ABI `Asset`. Image/audio inputs are passed inline as base64 data URLs.
