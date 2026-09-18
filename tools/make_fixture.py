"""Create a deterministic random miniature Llama checkpoint, never a pretrained model."""

import json
import random
import struct
import pathlib
import argparse


def create(path):
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)
    config = dict(
        model_type="llama",
        architectures=["LlamaForCausalLM"],
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=32,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
    )
    (path / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    rng = random.Random(1729)
    tensors = {}

    def add(name, shape, norm=False):
        n = 1
        for d in shape:
            n *= d
        tensors[name] = (
            shape,
            [
                1 + rng.uniform(-0.1, 0.1) if norm else rng.uniform(-0.2, 0.2)
                for _ in range(n)
            ],
        )

    add("model.embed_tokens.weight", [32, 16])
    add("lm_head.weight", [32, 16])
    add("model.norm.weight", [16], True)
    for layer in range(2):
        p = f"model.layers.{layer}."
        for name, shape in [
            ("q", [16, 16]),
            ("k", [8, 16]),
            ("v", [8, 16]),
            ("o", [16, 16]),
        ]:
            add(p + f"self_attn.{name}_proj.weight", shape)
        for name, shape in [("gate", [32, 16]), ("up", [32, 16]), ("down", [16, 32])]:
            add(p + f"mlp.{name}_proj.weight", shape)
        for name in ["input_layernorm", "post_attention_layernorm"]:
            add(p + name + ".weight", [16], True)
    header = {}
    data = bytearray()
    for name, (shape, values) in sorted(tensors.items()):
        start = len(data)
        data.extend(struct.pack("<" + "f" * len(values), *values))
        header[name] = dict(dtype="F32", shape=shape, data_offsets=[start, len(data)])
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((-len(h)) % 8)
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(h)) + h + data)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory")
    create(p.parse_args().directory)
