# CPU migration tests

`config.json` is NVIDIA's configuration from
[`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/blob/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/config.json),
revision `541d1f99c6b0c3cd0b11a95167540bb8edefd82b`, retrieved September 18, 2026.
The publisher lists the checkpoint under CC BY 4.0. No model weights are included.

The bundle-composition tests replace TensorRT compilation with byte payloads.
They exercise the real BuildRequest, family dispatch, configuration validation,
and BundleWriter. They do not establish engine validity, numerical equivalence,
transcript quality, GPU execution, or performance.
