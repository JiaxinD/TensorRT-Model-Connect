# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]


def test_release_reference_uses_the_tdt_auto_model_contract(monkeypatch) -> None:
    pinned_revision = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
    runner = runpy.run_path(
        str(REPOSITORY / "benchmarks/performance/baselines/task_reference.py")
    )
    captured: dict[str, object] = {}

    class FakeTensor:
        def is_floating_point(self):
            return True

        def to(self, *args, **kwargs):
            captured.setdefault("input_to", []).append((args, kwargs))
            return self

    class FakeSequences:
        def __getitem__(self, _index):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return [4, 8, 15]

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            captured["processor_load"] = (model, kwargs)
            return cls()

        def __call__(self, audio, **kwargs):
            captured["processor_call"] = (audio.tolist(), kwargs)
            return {"input_features": FakeTensor()}

        def decode(self, sequences, **kwargs):
            captured["decode"] = (sequences, kwargs)
            return ["Parakeet transcript"]

    class FakeModel:
        config = SimpleNamespace(_commit_hash=pinned_revision)

        @classmethod
        def from_pretrained(cls, model, **kwargs):
            captured["model_load"] = (model, kwargs)
            return cls()

        def eval(self):
            return self

        def to(self, device):
            captured["model_device"] = device
            return self

        def parameters(self):
            return iter([SimpleNamespace(dtype="fp32")])

        def generate(self, **kwargs):
            captured["generate"] = kwargs
            return SimpleNamespace(sequences=FakeSequences())

    class WrongModel:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            raise AssertionError("Parakeet TDT must not use AutoModelForSpeechSeq2Seq")

    class InferenceMode:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return None

    fake_torch = ModuleType("torch")
    fake_torch.float16 = "fp16"
    fake_torch.float32 = "fp32"
    fake_torch.bfloat16 = "bf16"
    fake_torch.device = lambda value: value
    fake_torch.inference_mode = InferenceMode
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoModelForSpeechSeq2Seq = WrongModel
    fake_transformers.AutoModelForTDT = FakeModel
    fake_transformers.AutoProcessor = FakeProcessor
    fake_engine = ModuleType("tools.validation.engine")
    fake_engine._read_wav_float32 = lambda _path: (
        np.array([0.25], dtype=np.float32),
        16_000,
    )
    fake_engine._resample_audio = lambda audio, _source_rate, _target_rate: audio
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "tools.validation.engine", fake_engine)

    manifest = (
        REPOSITORY
        / "tests/e2e/models/parakeet_tdt/manifests/parakeet-tdt-0.6b-v3.json"
    )
    arguments = SimpleNamespace(
        family="parakeet_tdt",
        manifest=manifest,
        model="nvidia/parakeet-tdt-0.6b-v3",
        revision=pinned_revision,
        precision="fp32",
        trust_remote_code=False,
        local_files_only=True,
    )
    session = runner["_load_asr"](
        arguments,
        {"audio_path": "data/Recording.wav", "max_new_tokens": 50},
        {"auto_model_class": "AutoModelForTDT"},
    )

    assert session.invoke() == {
        "text": "Parakeet transcript",
        "token_ids": [4, 8, 15],
        "output_tokens": 3,
    }
    assert captured["model_load"] == (
        "nvidia/parakeet-tdt-0.6b-v3",
        {
            "trust_remote_code": False,
            "local_files_only": True,
            "torch_dtype": "fp32",
            "revision": pinned_revision,
        },
    )
    assert captured["generate"]["max_new_tokens"] == 50
    assert captured["generate"]["return_dict_in_generate"] is True
    assert isinstance(captured["decode"][0], FakeSequences)
    assert captured["decode"][1] == {"skip_special_tokens": True}
    assert session.resolved_revision == pinned_revision
