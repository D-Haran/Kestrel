"""Verify Kestrel's llama3 RoPE scaling against HuggingFace on a small model that uses
the EXACT rope_scaling block of Llama 3.2 1B (factor 32, low 1, high 4, orig 8192)."""
import json, subprocess, pathlib, tempfile, os
import numpy as np, torch
from transformers import LlamaForCausalLM, LlamaConfig
from safetensors.torch import save_file

ROOT = pathlib.Path("/Users/derrickratnaharan/Documents/kestral")
TRACE = ROOT / "build-run/kestrel_trace"
KESTREL = ROOT / "build-run/kestrel"
import tempfile
MDIR = pathlib.Path(tempfile.mkdtemp(prefix="llama3rope-"))
torch.manual_seed(0)

cfg = LlamaConfig(
    hidden_size=256, intermediate_size=688, num_hidden_layers=3,
    num_attention_heads=8, num_key_value_heads=2, head_dim=32,
    vocab_size=512, max_position_embeddings=1024, rope_theta=500000.0,
    rms_norm_eps=1e-5, tie_word_embeddings=True,
    rope_scaling={"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                  "original_max_position_embeddings": 8192, "rope_type": "llama3"},
)
cfg._attn_implementation = "eager"
model = LlamaForCausalLM(cfg).float().eval()
# randomize a bit so logits are not degenerate
with torch.no_grad():
    for p in model.parameters():
        p.mul_(1.0).add_(torch.randn_like(p) * 0.02)

# Save config.json + model.safetensors (Kestrel-loadable)
cfg.to_json_file(MDIR / "config.json")
state = {k: v.contiguous() for k, v in model.state_dict().items() if "lm_head" not in k}
save_file(state, str(MDIR / "model.safetensors"), metadata={"format": "pt"})

ids = [1, 7, 42, 100, 256, 3, 9, 300, 55, 128, 400, 11, 77, 2, 480, 63]

# HF logits at the last position (position 0..len-1 exercises llama3 scaling at each)
with torch.no_grad():
    hf_logits = model(torch.tensor([ids])).logits[0, -1].double().numpy()

# Kestrel trace -> last-position logits, CPU and Metal
def kestrel_logits(backend):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tp = f.name
    r = subprocess.run([str(TRACE), str(MDIR), backend, ",".join(map(str, ids)), tp],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{backend} trace failed: {r.stderr.strip()}")
    d = json.load(open(tp)); os.unlink(tp)
    lg = d["logits"]
    row = lg[str(max(int(k) for k in lg))] if isinstance(lg, dict) else next(r for r in reversed(lg) if r)
    return np.asarray(row, dtype=np.float64)

def greedy(backend, n=16):
    r = subprocess.run([str(KESTREL), "--model", str(MDIR), "--backend", backend,
                        "--tokens", ",".join(map(str, ids)), "--max-tokens", str(n),
                        "--temperature", "0", "--context", "512"],
                       capture_output=True, text=True)
    return r.stdout.strip()

# HF greedy for comparison
with torch.no_grad():
    hf_gen = model.generate(torch.tensor([ids]), max_new_tokens=16, do_sample=False,
                            pad_token_id=2)[0, len(ids):].tolist()
hf_gen_str = " ".join(map(str, hf_gen))

print("config loaded by Kestrel:", (MDIR / "config.json").exists())
for be in ["cpu", "metal"]:
    kl = kestrel_logits(be)
    d = np.abs(kl - hf_logits)
    g = greedy(be)
    print(f"[{be}] last-token logit  max {d.max():.3e}  mean {d.mean():.3e} | "
          f"greedy matches HF: {g == hf_gen_str}")
print("HF greedy:     ", hf_gen_str)
