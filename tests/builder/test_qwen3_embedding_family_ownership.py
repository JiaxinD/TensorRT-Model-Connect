# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ownership regression tests for the standalone Qwen3-Embedding family."""

from pathlib import Path
import json

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_qwen3_embedding_owns_all_three_family_layers() -> None:
    python_root = (
        REPO_ROOT
        / "python"
        / "tensorrt_model_connect"
        / "families"
        / "qwen3_embedding"
    )
    runtime_root = REPO_ROOT / "src" / "runtime" / "models" / "qwen3_embedding"
    e2e_root = REPO_ROOT / "tests" / "e2e" / "models" / "qwen3_embedding"

    assert (python_root / "MODEL.toml").is_file()
    assert (python_root / "plugin.py").is_file()
    assert (runtime_root / "MODEL.toml").is_file()
    assert (runtime_root / "plugin.cpp").is_file()
    assert (e2e_root / "MODEL.toml").is_file()
    assert (e2e_root / "manifests" / "qwen3-embedding-0.6b.json").is_file()


def test_generation_qwen_family_does_not_own_embedding_strategy() -> None:
    python_qwen = (
        REPO_ROOT / "python" / "tensorrt_model_connect" / "families" / "qwen"
    )
    runtime_qwen = REPO_ROOT / "src" / "runtime" / "models" / "qwen"
    e2e_qwen = REPO_ROOT / "tests" / "e2e" / "models" / "qwen"

    assert "qwen_embedding" not in (python_qwen / "plugin.py").read_text(
        encoding="utf-8"
    )
    assert "qwen_embedding" not in (runtime_qwen / "MODEL.toml").read_text(
        encoding="utf-8"
    )
    assert "qwen3-embedding-0.6b.json" not in (e2e_qwen / "MODEL.toml").read_text(
        encoding="utf-8"
    )


def test_model_config_keeps_checkpoint_root_for_semantic_family_matching(
    tmp_path: Path,
) -> None:
    from tensorrt_model_connect.config import ModelConfig

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}),
        encoding="utf-8",
    )

    config = ModelConfig.from_dir(tmp_path)

    assert config.raw["_model_dir"] == str(tmp_path)


def test_semantic_checkpoint_routes_to_qwen3_embedding_family(tmp_path: Path) -> None:
    pytest.importorskip("tensorrt", reason="family plugins import TensorRT builders")
    from tensorrt_model_connect.config import ModelConfig
    from tensorrt_model_connect.families import find_plugin

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
                "hidden_size": 1024,
                "intermediate_size": 3072,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 151669,
                "max_position_embeddings": 32768,
                "eos_token_id": 151643,
                "hidden_act": "silu",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "modules.json").write_text(
        json.dumps(
            [
                {"path": "", "type": "sentence_transformers.models.Transformer"},
                {"path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
                {"path": "2_Normalize", "type": "sentence_transformers.models.Normalize"},
            ]
        ),
        encoding="utf-8",
    )
    pooling_dir = tmp_path / "1_Pooling"
    pooling_dir.mkdir()
    (pooling_dir / "config.json").write_text(
        json.dumps(
            {
                "word_embedding_dimension": 1024,
                "pooling_mode_cls_token": False,
                "pooling_mode_mean_tokens": False,
                "pooling_mode_max_tokens": False,
                "pooling_mode_lasttoken": True,
            }
        ),
        encoding="utf-8",
    )

    plugin = find_plugin(ModelConfig.from_dir(tmp_path))

    assert plugin is not None
    assert plugin.name == "qwen3_embedding"


def test_plain_qwen3_generation_stays_in_qwen_family(tmp_path: Path) -> None:
    pytest.importorskip("tensorrt", reason="family plugins import TensorRT builders")
    from tensorrt_model_connect.config import ModelConfig
    from tensorrt_model_connect.families import find_plugin

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
                "hidden_size": 1024,
                "intermediate_size": 3072,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 151669,
                "max_position_embeddings": 32768,
                "eos_token_id": 151643,
                "hidden_act": "silu",
            }
        ),
        encoding="utf-8",
    )

    plugin = find_plugin(ModelConfig.from_dir(tmp_path))

    assert plugin is not None
    assert plugin.name == "qwen"
