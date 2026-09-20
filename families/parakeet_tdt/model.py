# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the native Transformers Parakeet TDT checkpoint into owned sections."""

from pathlib import Path

from .config import ParakeetTDTConfig


def build(request, writer):
    required = {
        "family": "parakeet_tdt", "task": "speech_transcription", "backend": "trt",
        "tensor_parallel_size": 1, "context_parallel_size": 1, "max_batch_size": 1,
        "max_sequence_length": None, "image_height": None, "image_width": None,
        "video_num_frames": None, "fp32_layers": (), "dynamic_kv_cache": False,
        "graph_transform": None,
    }
    for name, expected in required.items():
        if getattr(request, name) != expected:
            raise ValueError(f"parakeet_tdt requires {name}={expected!r}")
    if request.precision not in {"fp16", "fp32"}:
        raise ValueError("parakeet_tdt supports precision=fp16 or fp32")
    if request.quantization not in {None, "none"}:
        raise ValueError("parakeet_tdt does not support quantization")

    model_dir = Path(request.model_dir)
    config = ParakeetTDTConfig.from_dir(model_dir)
    config.validate_supported_checkpoint()
    for filename in ("model.safetensors", "tokenizer.json"):
        if not (model_dir / filename).is_file():
            raise FileNotFoundError(f"parakeet_tdt requires {model_dir / filename}")

    from .engines import compile_engines

    plans, runtime = compile_engines(
        model_dir, precision=request.precision, verbose=request.verbose,
    )
    writer.set_header(family="parakeet_tdt", task=request.task, backend=request.backend)
    writer.add_json("runtime.json", runtime)
    for name, data in plans.items():
        writer.add_bytes(name, data)
    for filename in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        path = model_dir / filename
        if path.is_file():
            writer.add_bytes(filename, path.read_bytes())
