"""GPU smoke test: load the default model, run one forward pass on the bundled
sample images, write a SceneResult npz, and print VRAM usage.

Run inside the worker container:
    python -m worker.smoke_test
or with explicit images:
    python -m worker.smoke_test /path/img1.jpg /path/img2.jpg
"""

from __future__ import annotations

import glob
import os
import sys
import time


def _vram(tag: str) -> None:
    import torch

    if not torch.cuda.is_available():
        print(f"[{tag}] CUDA not available")
        return
    alloc = torch.cuda.memory_allocated() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[{tag}] VRAM alloc={alloc:.2f}GB peak={peak:.2f}GB total={total:.2f}GB")


def main(argv: list[str]) -> int:
    import torch

    from worker.adapters.registry import ModelRegistry

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, torch={torch.__version__}")
    if device == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    sample_dir = os.environ.get("SAMPLE_DIR", "/app/samples")
    image_paths = argv[1:] or sorted(
        glob.glob(os.path.join(sample_dir, "*.jpg"))
        + glob.glob(os.path.join(sample_dir, "*.png"))
    )
    if not image_paths:
        print(f"no images found in {sample_dir}; pass paths as args", file=sys.stderr)
        return 2
    print(f"images: {len(image_paths)}")

    _vram("start")
    reg = ModelRegistry(device=device)
    t0 = time.time()
    adapter = reg.warm()
    print(f"loaded {adapter.name} in {time.time()-t0:.1f}s")
    _vram("loaded")

    caps = adapter.entry.caps
    image_paths = image_paths[: caps.default_frames]

    t1 = time.time()
    result = adapter.run(
        image_paths,
        resolution=caps.recommended_resolution,
        conf_threshold=0.1,
        progress_cb=lambda stage, frac: print(f"  {stage}: {frac:.0%}"),
    )
    print(f"inference {time.time()-t1:.1f}s -> {result.meta}")
    _vram("inferred")

    out = "/results/smoke/arrays.npz"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    result.save_npz(out)
    print(f"wrote {out}: points={result.points_xyz.shape[0]} frames={result.frame_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
