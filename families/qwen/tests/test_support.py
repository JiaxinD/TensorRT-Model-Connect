# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep every Qwen checkpoint variant on the semantic Task contract."""

import pytest

from families.qwen.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


@pytest.mark.parametrize("model_type", ("qwen", "Qwen2", "qwen2", "qwen3", "qwq"))
def test_semantic_primary_task(model_type: str) -> None:
    support = describe(ModelMetadata(config={"model_type": model_type}, model_index={}))
    assert support is not None
    assert support.tasks == ("text_continuation",)
    assert support.default_task == "text_continuation"
