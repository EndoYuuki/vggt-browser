"""models.yaml loads and caps clamp correctly."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.model_config import load_models_config  # noqa: E402

CONFIG = os.path.join(os.path.dirname(__file__), "..", "config", "models.yaml")


def test_loads_models():
    cfg = load_models_config(CONFIG)
    assert cfg.default in cfg.models
    assert "vggt-1b" in cfg.models
    entry = cfg.get("vggt-1b")
    assert entry.adapter == "vggt"
    assert entry.options.get("hf_repo") == "facebook/VGGT-1B"


def test_caps_clamp():
    cfg = load_models_config(CONFIG)
    caps = cfg.get("vggt-1b").caps
    assert caps.clamp_frames(99999) == caps.max_frames
    assert caps.clamp_frames(0) == 1
    assert caps.clamp_resolution(99999) == caps.max_resolution


def test_public_dict_has_no_secrets():
    cfg = load_models_config(CONFIG)
    pub = cfg.get("vggt-1b").public_dict()
    assert "hf_repo" not in pub
    assert pub["caps"]["max_frames"] > 0
