# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU coverage of serialized-plan lifetime and atomic bundle publication."""

import gc
import importlib.util
import json
from pathlib import Path
import struct
import sys
import types

import pytest

from tensorrt_model_connect import BuildRequest
from tensorrt_model_connect.bundle_writer import BundleWriter

@pytest.fixture
def request_and_writer(tmp_path, monkeypatch):
    # Jiaxin Deng: stub only GPU compilation; exercise the real build and writer on CPU.
    with monkeypatch.context() as imports:
        for module_name, function_name in (
            ("dual_profile_decoder_builder", "build_dual_profile_decoder_engine"),
            ("standard_decoder_builder", "build_standard_decoder_engine"),
        ):
            module = types.ModuleType(f"families.llama.{module_name}")
            setattr(module, function_name, None)
            imports.setitem(sys.modules, module.__name__, module)
        spec = importlib.util.spec_from_file_location(
            "families.llama._plan_lifetime_model", Path(__file__).parents[1] / "model.py"
        )
        model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({
        "model_type": "llama", "hidden_size": 8, "intermediate_size": 16,
        "num_hidden_layers": 1, "num_attention_heads": 2,
        "num_key_value_heads": 1, "vocab_size": 32,
        "max_position_embeddings": 32, "eos_token_id": 2,
    }))
    monkeypatch.setattr(model, "load_standard_weights", lambda *args, **kwargs: {})
    destination = tmp_path / "model.bundle"
    request = BuildRequest(
        model_dir=checkpoint, output_path=destination, family="llama",
        task="text_generation", precision="fp16", max_sequence_length=16,
    )
    writer = BundleWriter(destination)
    yield model, request, writer
    writer.abort()


def test_prefill_plan_is_released_before_decode_build(request_and_writer, monkeypatch):
    model, request, writer = request_and_writer
    released = []

    class Plan(bytes):
        def __del__(self):
            released.append(True)

    def build_engine(config, *args, **kwargs):
        if config.raw["_decoder_engine_role"] == "prefill":
            return Plan(b"prefill bytes")
        gc.collect()
        assert released, "prefill plan is still resident during decode build"
        return b"decode bytes"

    monkeypatch.setattr(model, "_build_engine", build_engine)
    model.build(request, writer)
    writer.finish()
    data = request.output_path.read_bytes()
    header_size = struct.unpack("<Q", data[8:16])[0]
    header = json.loads(data[16:16 + header_size])
    payload = data[16 + header_size:]
    sections = {
        name: payload[entry["offset"]:entry["offset"] + entry["length"]]
        for name, entry in header["sections"].items()
    }
    assert sections["engine.plan"] == b"decode bytes"
    assert sections["prefill.plan"] == b"prefill bytes"
    assert json.loads(sections["runtime.json"])["decoder_engine_layout"] == "split"


def test_decode_failure_preserves_published_bundle(request_and_writer, monkeypatch):
    model, request, writer = request_and_writer
    request.output_path.write_bytes(b"previous bundle")

    def build_engine(config, *args, **kwargs):
        if config.raw["_decoder_engine_role"] == "prefill":
            return b"prefill bytes"
        raise RuntimeError("decode build failed")

    monkeypatch.setattr(model, "_build_engine", build_engine)
    with pytest.raises(RuntimeError, match="decode build failed"):
        model.build(request, writer)
    writer.abort()
    assert request.output_path.read_bytes() == b"previous bundle"
    assert not list(request.output_path.parent.glob(".model.bundle.sections.*"))
