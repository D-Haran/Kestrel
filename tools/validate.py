"""Testing only: compare the native executable with Hugging Face's eager Llama."""

import argparse
import json
import pathlib
import subprocess
import tempfile
import datetime
import hashlib
import numpy as np
import torch
from transformers import LlamaForCausalLM, LlamaConfig
from safetensors.torch import load_file


def metrics(actual, expected):
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape:
        raise AssertionError(f"shape mismatch: {a.shape} vs {b.shape}")
    delta = np.abs(a - b)
    denom = np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel())
    cosine = (
        float(np.dot(a.ravel(), b.ravel()) / denom)
        if denom
        else float(np.array_equal(a, b))
    )
    return dict(
        max_absolute_error=float(delta.max()),
        mean_absolute_error=float(delta.mean()),
        cosine_similarity=cosine,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--executable", default="build/kestrel_trace")
    p.add_argument("--backend", default="cpu")
    p.add_argument("--tokens", default="1,5,9,4,7")
    p.add_argument("--output", default="artifacts/validation.json")
    p.add_argument("--atol", type=float, default=2e-4)
    p.add_argument("--mean-tol", type=float, default=2e-5)
    p.add_argument("--cosine", type=float, default=0.99999)
    a = p.parse_args()
    torch.set_num_threads(1)
    ids = [int(x) for x in a.tokens.split(",")]
    config = LlamaConfig.from_pretrained(a.model, local_files_only=True)
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).float()
    state = {}
    for shard in sorted(pathlib.Path(a.model).glob("*.safetensors")):
        state.update(load_file(shard))
    if config.tie_word_embeddings:
        if "model.embed_tokens.weight" not in state:
            state["model.embed_tokens.weight"] = state["lm_head.weight"]
        if "lm_head.weight" not in state:
            state["lm_head.weight"] = state["model.embed_tokens.weight"]
    model.load_state_dict(state, strict=True)
    model.eval()
    reference = {}
    handles = []

    def hook(name):
        def save(module, inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            reference[name] = output.detach().float().cpu().numpy()[0]

        return save

    handles.append(model.model.embed_tokens.register_forward_hook(hook("embedding")))
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(hook(f"layer_{i}")))
    handles.append(model.model.norm.register_forward_hook(hook("final_hidden")))
    with torch.no_grad():
        reference["logits"] = (
            model(torch.tensor([ids]), use_cache=False).logits.float().numpy()[0]
        )
    with tempfile.TemporaryDirectory(prefix="kestrel-validate-") as temp:
        trace = pathlib.Path(temp) / "trace.json"
        subprocess.run(
            [a.executable, a.model, a.backend, a.tokens, str(trace)], check=True
        )
        native = json.loads(trace.read_text())
    report = dict(
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        pytorch_version=torch.__version__,
        reference_dtype="float32",
        stored_weight_dtypes=sorted({str(v.dtype) for v in state.values()}),
        checkpoint_sha256={
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in pathlib.Path(a.model).glob("*.safetensors")
        },
        checkpoint=json.loads(
            (pathlib.Path(a.model) / "download-manifest.json").read_text()
        )
        if (pathlib.Path(a.model) / "download-manifest.json").exists()
        else None,
        backend=a.backend,
        model=str(pathlib.Path(a.model)),
        tokens=ids,
        reference="transformers " + __import__("transformers").__version__,
        tolerances=dict(
            max_absolute=a.atol, mean_absolute=a.mean_tol, cosine_min=a.cosine
        ),
        stages={},
    )
    failures = []
    assert set(native) == set(reference), "trace stages differ"
    for name, expected in reference.items():
        m = metrics(native[name], expected)
        report["stages"][name] = m
        if not (
            m["max_absolute_error"] <= a.atol
            and m["mean_absolute_error"] <= a.mean_tol
            and m["cosine_similarity"] >= a.cosine
        ):
            failures.append(name)
        print(name, json.dumps(m))
    report["passed"] = not failures
    output = pathlib.Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        raise SystemExit("Numerical tolerance exceeded: " + ", ".join(failures))


if __name__ == "__main__":
    main()
