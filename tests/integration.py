"""Real executable integration tests, including malformed checkpoints and CLI errors."""

import argparse
import json
import math
import pathlib
import shutil
import struct
import subprocess
import tempfile

p = argparse.ArgumentParser()
p.add_argument("--binary", required=True)
p.add_argument("--bench", required=True)
p.add_argument("--model", required=True)
a = p.parse_args()


def run(*args, ok=True, match=None):
    r = subprocess.run(args, text=True, capture_output=True, timeout=30)
    assert (r.returncode == 0) == ok, (args, r.stdout, r.stderr)
    if match:
        assert match in r.stderr, (match, r.stderr)
    return r


with tempfile.TemporaryDirectory(prefix="kestrel-integration-") as d:
    root = pathlib.Path(d)
    model = root / "model"
    shutil.copytree(a.model, model)
    base = [a.binary, "--model", str(model), "--tokens", "1,5,9", "--max-tokens", "5"]
    first = run(*base, "--json", str(root / "out.json"))
    second = run(*base)
    assert first.stdout == second.stdout
    result = json.loads((root / "out.json").read_text())
    assert len(result["generated_ids"]) == 5
    run(*base, "--top-p", "0", ok=False, match="top-p")
    run(*base, "--top-k", "-1", ok=False, match="unsigned integer")
    run(*base, "--temperature", "nan", ok=False, match="temperature")
    run(*base, "--context", "2", ok=False, match="context")
    run(a.binary, "--model", str(model), "--tokens", "32", ok=False, match="token ID")
    run(
        a.binary,
        "--model",
        str(model),
        "--tokens",
        "1,",
        ok=False,
        match="trailing comma",
    )
    run(a.binary, "--unknown", ok=False, match="unknown option")
    run(a.binary, "--help")
    run(
        a.bench,
        "--model",
        str(model),
        "--prompt-tokens",
        "3",
        "--max-tokens",
        "4",
        "--warmup",
        "1",
        "--iterations",
        "2",
        "--json",
        str(root / "bench.json"),
        "--profile",
    )
    bench = json.loads((root / "bench.json").read_text())
    assert len(bench["runs"]) == 2 and bench["decode_steps_per_run"] == 3
    # Decode profiles the LM head per step (3 per run); batched prefill runs it once at the
    # final position and is timed as the "__prefill" scope rather than per op.
    assert bench["profile"]["LM head"]["calls"] == 2 * 3
    assert all(
        bench[key] > 0
        for key in [
            "prefill_tokens_per_second",
            "decode_tokens_per_second",
            "kv_cache_bytes",
        ]
    )
    assert bench["schema_version"] == 2 and bench["benchmark"]["batch_size"] == 1
    assert bench["model"]["weight_dtype"] == "F32"
    assert bench["model"]["hidden_size"] == 16
    samples = sorted(x for r in bench["runs"] for x in r["inter_token_ms"])
    assert len(samples) == 6 and bench["inter_token_sample_count"] == 6
    for label, q in [("p50", 0.5), ("p95", 0.95), ("p99", 0.99)]:
        assert (
            bench[f"{label}_inter_token_ms"] == samples[math.ceil(q * len(samples)) - 1]
        )
    assert math.isclose(bench["decode_elapsed_ms_total"], sum(samples))
    assert math.isclose(
        bench["decode_tokens_per_second"], 1000 * len(samples) / sum(samples)
    )
    assert math.isclose(
        bench["prefill_elapsed_ms_total"], sum(r["prefill_ms"] for r in bench["runs"])
    )
    assert bench["profile"]["Sampling"]["calls"] == 8
    for v in bench["profile"].values():
        assert math.isclose(v["mean_ms"] * v["calls"], v["total_ms"])
        assert math.isclose(
            v["percent_measured_runtime"],
            100 * v["total_ms"] / bench["measured_elapsed_ms_total"],
        )
    assert bench["backend_diagnostics"] == {}
    config = model / "config.json"
    original_config = config.read_text()
    for field, value in [
        ("model_type", "mistral"),
        ("attention_bias", True),
        ("head_dim", 3),
        ("hidden_size", -1),
        ("rms_norm_eps", 0),
        ("rope_scaling", {"factor": 2}),
    ]:
        cfg = json.loads(original_config)
        cfg[field] = value
        config.write_text(json.dumps(cfg))
        run(*base, ok=False)
    config.write_text(original_config)
    weights = model / "model.safetensors"
    original = weights.read_bytes()
    length = struct.unpack("<Q", original[:8])[0]
    header = json.loads(original[8 : 8 + length])
    payload = original[8 + length :]

    def write(h, data=payload):
        raw = json.dumps(h).encode()
        weights.write_bytes(struct.pack("<Q", len(raw)) + raw + data)

    first_key = next(iter(header))
    other = list(header)[1]
    import copy

    malformed = []
    h = copy.deepcopy(header)
    h[first_key]["shape"] = [2**63, 4]
    malformed.append(h)
    h = copy.deepcopy(header)
    h[first_key]["data_offsets"] = [0, len(payload) + 1]
    malformed.append(h)
    h = copy.deepcopy(header)
    h[first_key]["dtype"] = "I8"
    malformed.append(h)
    h = copy.deepcopy(header)
    h[other]["data_offsets"] = h[first_key]["data_offsets"]
    h[other]["shape"] = h[first_key]["shape"]
    malformed.append(h)
    h = copy.deepcopy(header)
    h["unexpected"] = h.pop("model.norm.weight")
    malformed.append(h)
    h = copy.deepcopy(header)
    h["model.norm.weight"]["shape"] = [4, 4]
    malformed.append(h)
    for h in malformed:
        write(h)
        run(*base, ok=False)
    for data in [
        b"",
        b"bad",
        struct.pack("<Q", 2**64 - 1),
        original[:-1],
        original + b"!",
    ]:
        weights.write_bytes(data)
        run(*base, ok=False)
    weights.write_bytes(original)
    index = model / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {first_key: "../escape.safetensors"}}))
    run(*base, ok=False, match="unsafe shard")
    index.unlink()
print(
    "CLI, benchmark schema, model rejection, and malformed Safetensors integration passed"
)
