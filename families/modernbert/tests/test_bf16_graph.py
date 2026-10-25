# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import math
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

from families.modernbert.config import ModelConfig, resolve_attention_contract
from tensorrt_model_connect.build import BuildRequest


class _Tensor:
    def __init__(self, values, dtype):
        storage = {
            "fp32": np.float32,
            "fp16": np.float16,
            "bf16": ml_dtypes.bfloat16,
            "int32": np.int32,
        }
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            self.values = np.asarray(values).astype(storage[dtype]).astype(np.float32)
        self.dtype = dtype
        self.shape = self.values.shape


def _layer(tensor):
    return SimpleNamespace(get_output=lambda index: tensor)


class _Network:
    def __init__(self, sequence_length):
        self.sequence_length = sequence_length
        self.ropes = []
        self.constants = []
        self.gathers = []
        self.output = None

    def add_input(self, name, dtype, shape):
        values = (
            np.arange(self.sequence_length) % 4
            if name == "input_ids"
            else np.asarray([1] + [0] * (self.sequence_length - 1))
        )
        return _Tensor(values, dtype)

    def add_constant(self, shape, weights):
        values = weights.reshape(shape)
        self.constants.append(values.copy())
        return _layer(
            _Tensor(
                values,
                "fp16"
                if values.dtype == np.float16
                else "int32"
                if values.dtype == np.int32
                else "fp32",
            )
        )

    def add_cast(self, tensor, dtype):
        return _layer(_Tensor(tensor.values, dtype))

    def add_elementwise(self, lhs, rhs, operation):
        assert lhs.dtype == rhs.dtype, "mixed elementwise dtypes"
        function = {"prod": np.multiply, "sum": np.add, "sub": np.subtract}[operation]
        return _layer(_Tensor(function(lhs.values, rhs.values), lhs.dtype))

    def add_gather(self, table, indices, axis):
        self.gathers.append(table)
        return _layer(_Tensor(np.take(table.values, indices.values.astype(int), axis), table.dtype))

    def add_shuffle(self, tensor):
        layer = SimpleNamespace(
            reshape_dims=tensor.shape, first_transpose=None, second_transpose=None
        )

        def output(index):
            values = tensor.values
            if layer.first_transpose is not None:
                values = values.transpose(layer.first_transpose)
            values = values.reshape(layer.reshape_dims)
            if layer.second_transpose is not None:
                values = values.transpose(layer.second_transpose)
            return _Tensor(values, tensor.dtype)

        layer.get_output = output
        return layer

    def add_shape(self, tensor):
        return _layer(_Tensor(tensor.shape, "int32"))

    def add_slice(self, tensor, start, shape, stride):
        layer = SimpleNamespace(length=shape[0])
        layer.set_input = lambda index, value: setattr(layer, "length", int(value.values[0]))
        layer.get_output = lambda index: _Tensor(
            tensor.values[start[0] : start[0] + layer.length : stride[0]], tensor.dtype
        )
        return layer

    def add_rotary_embedding(self, tensor, cos, sin, interleaved, rotary_dim):
        assert tensor.dtype == cos.dtype == sin.dtype, "mixed RoPE dtypes"
        self.ropes.append((tensor, cos, sin))
        layer = _layer(tensor)
        layer.set_input = lambda index, value: None
        return layer

    def add_unary(self, tensor, operation):
        assert operation == "erf"
        return _layer(_Tensor(np.vectorize(math.erf)(tensor.values), tensor.dtype))

    def add_matrix_multiply(self, lhs, lhs_op, rhs, rhs_op):
        assert lhs.dtype == rhs.dtype, "mixed matmul dtypes"
        return _layer(_Tensor(lhs.values @ rhs.values, lhs.dtype))

    def add_normalization_v2(self, tensor, gamma, beta, axes):
        assert tensor.dtype == gamma.dtype == beta.dtype, "mixed normalization dtypes"
        layer = SimpleNamespace(epsilon=1e-5)

        def output(index):
            values = tensor.values
            normalized = (values - values.mean(-1, keepdims=True)) / np.sqrt(
                values.var(-1, keepdims=True) + layer.epsilon
            )
            return _Tensor(normalized * gamma.values + beta.values, tensor.dtype)

        layer.get_output = output
        return layer

    def mark_output(self, tensor):
        self.output = tensor


def _source_functions(path, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *[node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))],
    ]
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.fixture
def graph():
    def create(sequence_length):
        network = _Network(sequence_length)
        logger = type(
            "Logger", (), {"VERBOSE": 1, "WARNING": 2, "__init__": lambda self, level: None}
        )
        profile = SimpleNamespace(set_shape=lambda *args: None)
        builder = SimpleNamespace(
            create_network=lambda flags: network,
            create_builder_config=lambda: SimpleNamespace(
                clear_flag=lambda flag: None, add_optimization_profile=lambda profile: None
            ),
            create_optimization_profile=lambda: profile,
            build_serialized_network=lambda net, config: b"typed-graph",
        )
        trt = SimpleNamespace(
            float16="fp16",
            float32="fp32",
            bfloat16="bf16",
            int32="int32",
            Weights=lambda values: values,
            Logger=logger,
            Builder=lambda logger: builder,
            Permutation=tuple,
            NetworkDefinitionCreationFlag=SimpleNamespace(STRONGLY_TYPED=1),
            BuilderFlag=SimpleNamespace(TF32=1),
            ElementWiseOperation=SimpleNamespace(PROD="prod", SUM="sum", SUB="sub"),
            MatrixOperation=SimpleNamespace(NONE="none"),
            UnaryOperation=SimpleNamespace(ERF="erf"),
        )
        owner = Path(__file__).resolve().parents[1]
        ops = _source_functions(owner / "graph/model.py", {"np": np, "trt": trt})
        parallel = _source_functions(
            owner / "parallel.py",
            {
                "np": np,
                "dataclass": dataclass,
                "replace": replace,
                "__name__": __name__,
                "trt": trt,
            },
        )

        def attention(net, q, k, v, **kwargs):
            assert q.dtype == k.dtype == v.dtype == kwargs["mask"].dtype
            return q

        ops.add_attention_from_rows = attention
        encoder = _source_functions(
            owner / "model.py",
            {
                "np": np,
                "trt": trt,
                "sys": sys,
                "Path": Path,
                "graph_ops": ops,
                "ModelConfig": ModelConfig,
                "resolve_attention_contract": resolve_attention_contract,
                "normalize_parallel_config": parallel.normalize_parallel_config,
                "ParallelConfig": parallel.ParallelConfig,
                "__package__": "families.modernbert",
            },
        )
        return network, encoder

    return create


def _weights(values):
    weights = {
        "embedding": np.tile(values, (4, 1)).astype(np.float32),
        "embed_norm": np.ones(4, np.float32),
        "final_norm": np.ones(4, np.float32),
    }
    for index in range(2):
        prefix = f"layer.{index}."
        for name in ("w_q", "w_k", "w_v", "w_o", "w_mlp_input", "w_mlp_gate", "w_down"):
            weights[prefix + name] = np.eye(4, dtype=np.float32)
        weights[prefix + "mlp_norm"] = np.ones(4, np.float32)
        if index:
            weights[prefix + "attn_norm"] = np.ones(4, np.float32)
    return weights


@pytest.mark.parametrize("precision", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("sequence_length", [1, 3, 8])
def test_serial_graph_preserves_requested_precision(graph, precision, sequence_length):
    network, encoder = graph(sequence_length)
    config = ModelConfig.from_json(
        json.dumps(
            {
                "model_type": "modernbert",
                "hidden_size": 4,
                "vocab_size": 4,
                "num_hidden_layers": 2,
                "num_attention_heads": 1,
                "intermediate_size": 4,
            }
        )
    )
    values = [2**20, -(2**20), 2**-30, -(2**-30)] if precision != "fp16" else [1, -1, 0.5, -0.5]
    assert (
        encoder._ModernBertModel().build_engine(config, _weights(values), 8, precision=precision)
        == b"typed-graph"
    )
    table = network.gathers[0]
    assert table.dtype == precision
    expected = np.asarray(
        values,
        dtype={"fp32": np.float32, "fp16": np.float16, "bf16": ml_dtypes.bfloat16}[precision],
    ).astype(np.float32)
    np.testing.assert_array_equal(table.values, np.tile(expected, (4, 1)))
    assert len(network.ropes) == 4
    assert all(tensor.dtype == precision for rope in network.ropes for tensor in rope)
    assert network.output.dtype == "fp32"
    assert network.output.shape == (sequence_length, 4)
    assert np.isfinite(network.output.values).all()
    if precision == "bf16":
        assert not any(values.dtype == np.float16 for values in network.constants)


@pytest.mark.parametrize("precision", ["fp32", "fp16", "bf16"])
def test_tp_only_dispatches_its_implemented_fp32_precision(graph, monkeypatch, precision):
    _, encoder = graph(1)
    parallel = SimpleNamespace(enabled=True, validate=lambda: None)
    encoder._ModernBertModel.build_engine.__globals__["normalize_parallel_config"] = lambda config: (
        parallel
    )
    monkeypatch.setitem(
        sys.modules,
        "families.modernbert.parallel",
        SimpleNamespace(build_tp_modernbert_engine=lambda *args, **kwargs: b"fp32-only"),
    )
    if precision == "fp32":
        assert (
            encoder._ModernBertModel().build_engine(
                None, {}, 8, precision=precision, parallel_config=parallel
            )
            == b"fp32-only"
        )
    else:
        with pytest.raises(NotImplementedError, match="require fp32"):
            encoder._ModernBertModel().build_engine(
                None, {}, 8, precision=precision, parallel_config=parallel
            )


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_public_tp_request_fails_before_loading_weights(graph, tmp_path, precision):
    _, encoder = graph(1)
    namespace = encoder.build.__globals__
    namespace["_ModernBertModel"] = lambda: pytest.fail(
        "weights loaded for unsupported TP precision"
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "modernbert"}), encoding="utf-8"
    )
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "model.bundle",
        family="modernbert",
        task="encoding",
        precision=precision,
        tensor_parallel_size=2,
    )
    with pytest.raises(NotImplementedError, match="require fp32"):
        encoder.build(request, None)


def test_public_serial_bf16_request_reaches_typed_graph(graph, tmp_path, monkeypatch):
    network, encoder = graph(3)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "modernbert",
                "hidden_size": 4,
                "vocab_size": 4,
                "num_hidden_layers": 2,
                "num_attention_heads": 1,
                "intermediate_size": 4,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        encoder._ModernBertModel, "load_weights", lambda *args: _weights([1, -1, 0.5, -0.5])
    )
    namespace = encoder.build.__globals__
    namespace["_tokenizer_runtime_contract"] = lambda path: {}
    namespace["_BUNDLE_FILES"] = ()
    engines = {}
    writer = SimpleNamespace(
        set_header=lambda **kwargs: None,
        add_json=lambda *args: None,
        add_bytes=lambda name, data: engines.update({name: data}),
    )
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "model.bundle",
        family="modernbert",
        task="encoding",
        precision="bf16",
        max_sequence_length=8,
    )
    encoder.build(request, writer)
    assert engines["engine.plan"] == b"typed-graph"
    assert network.gathers[0].dtype == "bf16"
