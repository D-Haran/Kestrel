"""Exercise F16, BF16, indexed shards, and both tied-weight storage layouts."""

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile
import torch
from safetensors.torch import load_file, save_file

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--executable", required=True)
p.add_argument("--backend", default="cpu")
a = p.parse_args()
original = load_file(str(pathlib.Path(a.model) / "model.safetensors"))
config = json.loads((pathlib.Path(a.model) / "config.json").read_text())
with tempfile.TemporaryDirectory(prefix="kestrel-variants-") as temp:
    for variant in ["f16", "bf16", "sharded", "tied_embedding", "tied_head"]:
        root = pathlib.Path(temp) / variant
        root.mkdir()
        cfg = dict(config)
        state = dict(original)
        if variant in ["f16", "bf16"]:
            state = {
                k: v.to(torch.float16 if variant == "f16" else torch.bfloat16)
                for k, v in state.items()
            }
        if variant.startswith("tied"):
            cfg["tie_word_embeddings"] = True
            state.pop("lm_head.weight")
            if variant == "tied_head":
                state["lm_head.weight"] = state.pop("model.embed_tokens.weight")
        (root / "config.json").write_text(json.dumps(cfg))
        if variant == "sharded":
            keys = list(state)
            shards = [
                {k: state[k] for k in keys[::2]},
                {k: state[k] for k in keys[1::2]},
            ]
            mapping = {}
            for i, shard in enumerate(shards):
                name = f"model-{i + 1:05d}-of-00002.safetensors"
                save_file(shard, str(root / name))
                mapping.update({key: name for key in shard})
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": mapping})
            )
        else:
            save_file(state, str(root / "model.safetensors"))
        subprocess.run(
            [
                sys.executable,
                str(pathlib.Path(__file__).with_name("validate.py")),
                "--model",
                str(root),
                "--executable",
                a.executable,
                "--backend",
                a.backend,
                "--output",
                str(root / "result.json"),
            ],
            check=True,
        )
        print(variant, "passed", flush=True)
