# Native Safetensors loading

The runtime loads files itself with POSIX `open`, `fstat`, and read-only private `mmap`. File descriptors close immediately after mapping; RAII unmaps storage after all model views are finished. Neither Python nor an inference engine reads runtime weights.

File layout:

```text
[8-byte little-endian unsigned header length N]
[N bytes UTF-8 JSON, optionally padded with spaces]
[contiguous little-endian tensor payload]
```

Each tensor entry has `dtype`, `shape`, and `data_offsets: [begin,end]`. Offsets are relative to the payload start, not the file start. The byte range is half-open. `__metadata__` is a string-to-string map. Tensor byte count must equal the checked product of shape dimensions times dtype size. Scalar shape `[]` has one element; zero dimensions produce empty tensors.

The reader rejects truncated prefixes/headers/payloads, headers over 100 MB, non-object headers, duplicate JSON keys, invalid/negative dimensions, overflow, unsupported dtypes, reversed/out-of-bounds ranges, overlaps, holes, trailing payload bytes, and non-string metadata. JSON nesting is capped at 128. Dtypes: F32 (4 bytes), F16 (2), BF16 (2). Half subnormals, signed zero, infinity, and NaN decode according to their bit representations. Nonfinite generated logits fail sampling.

For sharded models, `model.safetensors.index.json` supplies `weight_map`. Every tensor must belong to the specified shard; duplicate tensors, missing entries, or unsafe shard filenames fail. Only simple filenames in the model directory are accepted. A symlinked model directory is not a security isolation boundary; model files must remain immutable while mapped.

Configuration and weight shapes are validated before inference. This supports the selected unscaled Llama architecture, not every model that happens to use Safetensors. No quantization metadata, alternate architectures, biases, sliding window, scaled RoPE, custom head dimension, or tensor-parallel layout is silently accepted.

The only vendored dependency is the pinned nlohmann JSON parser, with its MIT license and checksum in `third_party/README.md`. Source format reference: [Hugging Face Safetensors specification](https://github.com/huggingface/safetensors#format).
