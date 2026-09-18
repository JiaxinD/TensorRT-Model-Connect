# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile and execute CPU-owned native contracts without CUDA or TensorRT."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize(("name", "sources"), [
    ("audio_input", ("resampler.cpp",)),
    ("audio_helpers", ("audio_helpers.cpp", "resampler.cpp")),
    ("decode_policy", ()),
])
def test_native_cpu_contract(tmp_path, name, sources):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("native CPU tests require a C++17 compiler (set CXX)")
    family = Path(__file__).resolve().parents[1]
    root = family.parents[1]
    output = tmp_path / name
    subprocess.run([
        compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
        "-I", str(root), "-I", str(root / "core/runtime/include"),
        str(family / "tests/cpp" / f"test_{name}.cpp"),
        *(str(family / "runtime" / source) for source in sources),
        "-o", str(output),
    ], check=True, capture_output=True, text=True, timeout=60)
    subprocess.run([str(output)], check=True, capture_output=True, text=True, timeout=60)
