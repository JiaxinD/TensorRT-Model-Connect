# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parakeet TDT-owned NeMo archive resolution."""

from __future__ import annotations

import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any


def _cfg_int(*values: object, default: int) -> int:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return default


def _read_model_config(nemo_path: Path) -> dict[str, Any]:
    import yaml

    try:
        with tarfile.open(str(nemo_path), "r") as tar:
            for member in tar.getmembers():
                if Path(member.name).name == "model_config.yaml":
                    handle = tar.extractfile(member)
                    if handle is not None:
                        loaded = yaml.safe_load(handle.read()) or {}
                        return loaded if isinstance(loaded, dict) else {}
    except (OSError, tarfile.TarError, yaml.YAMLError):
        # A downloaded HF snapshot can contain an unrelated or incomplete
        # archive.  This adapter participates in global family discovery, so
        # an unrecognized archive must fall through to the normal HF resolver.
        return {}
    return {}


def _matches_parakeet_tdt(cfg: dict[str, Any]) -> bool:
    target = " ".join(
        str(cfg.get(key, "") or "")
        for key in ("target", "_target_", "model_type")
    ).lower()
    decoding = cfg.get("decoding", {}) if isinstance(cfg.get("decoding"), dict) else {}
    durations = cfg.get(
        "durations", cfg.get("tdt_durations", decoding.get("durations", ()))
    )
    joint = cfg.get("joint", {}) if isinstance(cfg.get("joint"), dict) else {}
    jointnet = joint.get("jointnet", {}) if isinstance(joint.get("jointnet"), dict) else {}
    extra_outputs = _cfg_int(
        joint.get("num_extra_outputs"), jointnet.get("num_extra_outputs"), default=-1)
    try:
        is_tdt = tuple(int(value) for value in durations) == (0, 1, 2, 3, 4)
    except (TypeError, ValueError):
        is_tdt = False
    return "encdecrnntbpemodel" in target.replace(".", "") and is_tdt and extra_outputs == 5


def _symlink_archive(tmp_path: Path, nemo_path: Path) -> None:
    link = tmp_path / nemo_path.name
    if not link.exists():
        os.symlink(str(nemo_path.resolve()), str(link))


def resolve_nemo_archive(nemo_path: Path) -> str | None:
    """Create a Parakeet TDT model directory from a matching .nemo file."""
    cfg = _read_model_config(nemo_path)
    if not _matches_parakeet_tdt(cfg):
        return None

    enc_cfg = cfg.get("encoder", {})
    defaults = cfg.get("model_defaults", {})
    dec_cfg = cfg.get("decoder", {})
    prednet = dec_cfg.get("prednet", {})
    if not isinstance(prednet, dict):
        prednet = {}
    hidden = _cfg_int(enc_cfg.get("d_model"), defaults.get("enc_hidden"), default=1024)
    pred_layers = _cfg_int(prednet.get("pred_rnn_layers"), default=1)
    pred_hidden = _cfg_int(prednet.get("pred_hidden"), defaults.get("pred_hidden"), default=640)
    vocab_size = _cfg_int(cfg.get("vocab_size"), defaults.get("vocab_size"), default=1024)

    tmp_dir = tempfile.mkdtemp(prefix="trtmc_nemo_parakeet_tdt_")
    tmp_path = Path(tmp_dir)
    synthetic_config = {
        "model_type": "parakeet_tdt",
        "hidden_size": hidden,
        "num_hidden_layers": pred_layers,
        "num_attention_heads": 1,
        "intermediate_size": pred_hidden,
        "vocab_size": vocab_size,
        "rms_norm_eps": 1e-5,
        "_nemo_archive_path": str(nemo_path),
    }
    (tmp_path / "config.json").write_text(
        json.dumps(synthetic_config, indent=2),
        encoding="utf-8",
    )
    _symlink_archive(tmp_path, nemo_path)
    print(
        f"[trtmc build] Parakeet TDT NeMo resolved: tmp_dir={tmp_dir}",
        file=sys.stderr,
    )
    return tmp_dir


def resolve_model_dir(model_dir: Path) -> str | None:
    """Stage a matching NeMo snapshot without mutating the source cache.

    Some Hugging Face NeMo repositories include a top-level ``config.json``.
    The generic resolver would otherwise treat that snapshot as a writable HF
    model directory, while this family extracts tokenizer files beside the
    archive for bundle packaging.  Resolve the archive first so those generated
    files land in the family-owned temporary directory instead of the read-only
    shared cache.
    """
    if not model_dir.is_dir():
        return None
    # The official v3 snapshot publishes native Transformers safetensors and
    # a legacy NeMo archive side by side.  Claim the directory without staging
    # it so a broader NeMo adapter cannot reinterpret the legacy archive.
    config_path = model_dir / "config.json"
    if (model_dir / "model.safetensors").is_file() and config_path.is_file():
        try:
            hf_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            hf_config = {}
        if hf_config.get("model_type") == "parakeet_tdt":
            return model_dir
    for nemo_path in sorted(model_dir.glob("*.nemo")):
        resolved = resolve_nemo_archive(nemo_path)
        if resolved is not None:
            return resolved
    return None
