# VGGT Browser

A self-hosted web interface for running [VGGT](https://github.com/facebookresearch/vggt)
and [VGGT-Omega](https://github.com/facebookresearch/vggt-omega) — feed-forward 3D
reconstruction models — and exploring their output directly in the browser.

Drop in a set of images, a video, or a single frame, and get back a dense colored
point cloud, recovered camera poses, and per-frame depth and confidence maps,
rendered live with WebGL. Models are swappable from a dropdown; everything runs
locally behind Docker.

![VGGT Browser reconstructing a scene: dense point cloud, recovered camera
frustums, and per-frame depth/confidence panels](docs/screenshot.png)

## Highlights

- **Point cloud, cameras, depth, confidence** — not just a merged cloud. Inspect
  any frame's depth and confidence map, view camera frustums in 3D, and recolor
  the cloud by RGB, source frame, or confidence on the fly.
- **Images, video, or a single frame.** Video is decoded server-side; you choose
  the sampling rate and frame budget.
- **Swap models without a restart.** `VGGT-1B` and `VGGT-Omega` ship configured;
  adding another is a few lines of YAML plus a small adapter.
- **GPU stays isolated.** Only one container carries CUDA and the resident model;
  the web and API layers are torch-free and lightweight.
- **Tuned for a real GPU budget.** Frame-count and resolution caps are enforced
  before work reaches the GPU, and inference auto-downgrades resolution on OOM
  rather than failing.

## How it works

Four Docker Compose services. Torch and CUDA live **only** in the worker:

| service  | role                                                            | GPU |
|----------|-----------------------------------------------------------------|:---:|
| `web`    | nginx serving the React/three.js SPA; proxies `/api` and `/ws`  |  —  |
| `api`    | FastAPI — uploads, ffmpeg frame extraction, result serialization|  —  |
| `worker` | torch + CUDA + VGGT/VGGT-Omega; resident model, single RQ worker |  ✓  |
| `redis`  | RQ broker, job state, and progress pub/sub                      |  —  |

```
  browser ──upload──▶ api ──enqueue──▶ redis ──▶ worker (resident model)
     ▲                 │                              │
     └── WebSocket ◀───┴──── progress pub/sub ◀───────┘
     └── binary points / camera JSON / depth PNGs ◀── results volume
```

A single RQ worker pulls one job at a time, which serializes access to the GPU
and lets the model stay loaded across jobs. Progress is published to Redis and
relayed to the browser over a WebSocket (with HTTP polling as a fallback).

Each model's raw output is normalized by an adapter (`worker/adapters/`) into one
`SceneResult` contract (`shared/scene.py`), so the API and frontend never need to
know that, say, VGGT exposes a point map while VGGT-Omega only exposes depth.
Point clouds reach the browser as a compact little-endian binary blob
(`shared/wire_format.py`) that maps straight onto a three.js buffer with no
parsing; depth and confidence maps are rendered to PNG on demand.

## Quick start

Requires an NVIDIA GPU with recent drivers and Docker configured with the
`nvidia` runtime.

```bash
cp .env.example .env       # optional: set HF_TOKEN for gated models
docker compose up --build
```

Open <http://localhost:8080>, pick a model, drop in some images or a video, and
hit **Reconstruct**. The worker loads the default model on startup; set
`WARM_DEFAULT_MODEL=0` to defer that to the first job instead.

### Model access

`facebook/VGGT-1B` is public — no token needed. **VGGT-Omega** checkpoints are
gated: request access on the [model page](https://huggingface.co/facebook/VGGT-Omega),
then put a Hugging Face token in `.env` as `HF_TOKEN`. A read-only token is
sufficient; a fine-grained token needs "read access to public gated repos."

### GPU smoke test

Run a single forward pass on whatever images are in `samples/` and print VRAM use,
without bringing up the full stack:

```bash
docker compose -f docker-compose.yml -f docker-compose.smoke.yml run --rm worker
```

## Inputs

| Input         | Formats                                  | Notes                                  |
|---------------|------------------------------------------|----------------------------------------|
| Images        | `.jpg` `.jpeg` `.png` `.bmp` `.webp`      | Multiple views of one scene work best  |
| Single image  | same                                     | Monocular depth / point estimate       |
| Video         | `.mp4` `.mov` `.avi` `.mkv` `.webm`       | Sampled at a chosen fps, capped        |

Decoding is handled by ffmpeg, so most codecs work; the extension list is just
the upload filter. Images and video can't be mixed in one job.

## Configuration

Models are declared in [`config/models.yaml`](config/models.yaml):

```yaml
default: vggt-1b
models:
  vggt-1b:
    adapter: vggt
    hf_repo: facebook/VGGT-1B
    dtype: bfloat16
    point_source: depth          # "depth" (unproject) or "pointmap" (direct)
    caps:
      max_frames: 64
      default_frames: 24
      recommended_resolution: 518
      max_resolution: 518
```

`caps` are the GPU budget knobs. The API clamps frame count and resolution to
these before a job is enqueued, and video extraction stops at `max_frames`.
Defaults here target a 24 GB card; adjust the numbers for your hardware — no code
changes required. On a CUDA out-of-memory error the worker halves resolution and
retries (down to a floor) instead of failing, keeping the model resident.

### Point source

VGGT can produce world points two ways, and the paper reports that unprojecting
the predicted depth with the recovered camera is more accurate than the model's
direct point-map head. Both are available per job (the **Point source** dropdown,
or `point_source` in the model config), defaulting to `depth`.

## Adding a model

1. Add an entry to `config/models.yaml` pointing at an `adapter`.
2. If it's a new architecture, add an adapter under `worker/adapters/`
   implementing `load` / `unload` / `run` and returning a `SceneResult`, then
   register it in `worker/adapters/registry.py`.

The existing `VGGTAdapter` and `VGGTOmegaAdapter` are short and make good
references — the second shows how to absorb a different output schema (decode the
pose encoding, unproject depth) behind the same contract.

## Development

The Python that doesn't touch CUDA — the wire format, the `SceneResult` schema,
config loading, result cleanup — has a small torch-free test suite:

```bash
python -m venv .venv && . .venv/bin/activate
pip install "numpy<2" pyyaml pytest pillow redis rq
pytest tests/ -q
```

It covers the binary point format round-trip (mirroring the TypeScript parser),
`SceneResult` validation and npz round-trip, config cap clamping, the adapter
contract, and the result-TTL sweep.

The frontend lives in `web/`. For a hot-reloading dev server pointed at a running
`api` container:

```bash
cd web && npm install && npm run dev
```

## Notes

- **Coordinate convention.** VGGT uses the OpenCV convention (+Y down, +Z
  forward); the viewer flips Y/Z so that up is up and orbit controls behave.
- **Confidence filtering.** VGGT's confidence is an unbounded score, so the
  per-job filter is expressed as a quantile ("drop the lowest *N*%") rather than
  an absolute threshold. The viewer's confidence slider then trims points live in
  the shader without re-fetching.
- **Result lifetime.** Job outputs are written to a volume and swept after
  `RESULT_TTL_SECONDS` (24h by default).

## License

See [LICENSE](LICENSE). VGGT and VGGT-Omega have their own licenses and gated
checkpoint terms — review those before redistributing any weights or output.
