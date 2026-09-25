# MiniCPM5-1B checkpoint coverage

This adds a pinned E2E selection for `openbmb/MiniCPM5-1B` through the existing
Llama family. It does not add a new architecture or claim GPU qualification.

The configuration fixture is copied from the Apache-2.0 publisher checkpoint:
https://huggingface.co/openbmb/MiniCPM5-1B/blob/87179e5c1f455ef22e6223592d2d61351b525bfc/config.json

Unlike the existing 2B checkpoint, the 1B model has hidden width 1536 but
16 attention heads of width 128 (attention width 2048), with two KV heads and
24 layers. CPU tests preserve that explicit width, full-context KV byte
geometry and the two stop IDs. The E2E uses the existing family build/runtime
and reference comparison with thinking disabled, FP16 candidate, FP32 reference,
a 256-token build limit and ten generated tokens. These limits are a small
parity experiment, not a long-context or performance claim.

Validation remaining: build the pinned checkpoint on authorized GPU hardware
and run `families/llama/tests/test_e2e.py` with `--e2e-model minicpm5-1b`, using
the normal family E2E runtime environment. No large weights were downloaded locally.

The existing Llama tokenizer's Sequence/Split classification issue is tracked
in upstream PR #1423. This checkpoint coverage must not be treated as complete
runtime qualification until tokenizer compatibility and target-GPU parity are
verified. This change does not duplicate that tokenizer implementation.
