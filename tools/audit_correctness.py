"""Greedy-decode correctness: Kestrel (CPU & Metal) vs HuggingFace eager fp32.

Same token IDs fed to both engines (tokenizer removed as a variable).
For each prompt: 32 greedy tokens from HF and from Kestrel; compare sequences.
Also compares first-generated-token logits (Kestrel trace vs HF)."""
import json, subprocess, sys, pathlib, tempfile, os
import numpy as np, torch, sentencepiece as spm
from transformers import LlamaForCausalLM, LlamaConfig
from safetensors.torch import load_file

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL = ROOT / "models/tinystories-15m"
KESTREL = ROOT / "build-run/kestrel"
TRACE = ROOT / "build-run/kestrel_trace"
OUT = ROOT / "benchmarks/results/audit"
NEW = 32
torch.manual_seed(0); torch.set_num_threads(4)

PROMPTS = [
    # short
    "Once upon a time", "The cat sat on", "Hello", "She said", "A big red",
    "Tom and Jane", "It was a sunny day", "The end", "Why did the", "Run fast",
    # longer narrative
    "Once upon a time there was a little girl who lived in a small village near the forest and she loved to",
    "The old man walked slowly down the long road, thinking about all the years that had passed since he was young and",
    "In the morning the children went outside to play in the garden where the flowers were blooming and the birds were",
    "Every night before bed, the little boy asked his mother to read him a story about a brave knight who",
    "The dog ran across the field as fast as it could, chasing the ball that the boy had thrown far into the",
    # code-like
    "def add(a, b):\n    return", "for i in range(10):", "import numpy as np\n",
    "class Animal:\n    def __init__(self", "SELECT * FROM users WHERE",
    "#include <stdio.h>\nint main() {", "print('hello world')", "x = [1, 2, 3]\n",
    "if x > 0:\n    print", "while True:",
    # math
    "2 + 2 =", "The sum of 5 and 7 is", "10 times 10 equals", "One plus one",
    "Half of 100 is", "3, 6, 9, 12,", "The square root of 16", "Pi is approximately",
    "A triangle has three", "Count: 1 2 3",
    # multilingual
    "Bonjour le monde", "Hola, ¿cómo estás?", "Guten Tag", "こんにちは世界",
    "Привет мир", "Ciao mondo", "El gato duerme", "La petite fille",
    "Das Kind spielt", "Il fait beau aujourd'hui",
    # misc/varied
    "The quick brown fox", "To be or not to be", "Dear friend,", "Breaking news:",
    "The weather today is", "My favorite color is", "Yesterday I went", "Thank you very",
]
PROMPTS = PROMPTS[:50]
assert len(PROMPTS) == 50, len(PROMPTS)

sp = spm.SentencePieceProcessor(model_file=str(MODEL / "tokenizer.model"))
BOS = 1  # Llama <s>; add_bos_token=True
def encode(text):
    return [BOS] + sp.encode(text)
cfg = LlamaConfig.from_pretrained(str(MODEL), local_files_only=True)
cfg._attn_implementation = "eager"
hf = LlamaForCausalLM(cfg).float().eval()
state = load_file(str(MODEL / "model.safetensors"))
if "lm_head.weight" not in state and cfg.tie_word_embeddings:
    state["lm_head.weight"] = state["model.embed_tokens.weight"]
hf.load_state_dict(state, strict=False)

def kestrel_gen(ids, backend):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        jp = f.name
    subprocess.run([str(KESTREL), "--model", str(MODEL), "--backend", backend,
                    "--tokens", ",".join(map(str, ids)), "--max-tokens", str(NEW),
                    "--temperature", "0", "--json", jp],
                   check=True, capture_output=True)
    d = json.load(open(jp)); os.unlink(jp)
    return d["generated_ids"]

def kestrel_logits(ids, backend):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tp = f.name
    subprocess.run([str(TRACE), str(MODEL), backend, ",".join(map(str, ids)), tp],
                   check=True, capture_output=True)
    d = json.load(open(tp)); os.unlink(tp)
    lg = d["logits"]
    if isinstance(lg, dict):
        row = lg[str(max(int(k) for k in lg))]
    else:
        row = next(r for r in reversed(lg) if r is not None)
    return np.asarray(row, dtype=np.float64)

def run_backend(backend, hf_cache):
    rows, identical = [], 0
    logit_max, logit_mean = [], []
    for i, p in enumerate(PROMPTS):
        ids = hf_cache[i]["ids"]
        hf_gen = hf_cache[i]["gen"]
        hf_lg = hf_cache[i]["logits"]
        k_gen = kestrel_gen(ids, backend)
        k_lg = kestrel_logits(ids, backend)
        same = (k_gen == hf_gen)
        if same: identical += 1
        div = next((j for j in range(min(len(k_gen), len(hf_gen)))
                    if k_gen[j] != hf_gen[j]), None)
        d = np.abs(k_lg - hf_lg)
        logit_max.append(float(d.max())); logit_mean.append(float(d.mean()))
        rows.append(dict(idx=i, prompt=p[:40], n_prompt=len(ids), identical=same,
                         first_divergence=div, logit_max=float(d.max()),
                         logit_mean=float(d.mean())))
    return dict(backend=backend, identical=identical, total=len(PROMPTS),
                logit_max_overall=max(logit_max), logit_mean_overall=float(np.mean(logit_mean)),
                rows=rows)

# HF reference (compute once)
hf_cache = []
for p in PROMPTS:
    id_list = encode(p)
    ids = torch.tensor([id_list])
    with torch.no_grad():
        out = hf.generate(ids, max_new_tokens=NEW, do_sample=False,
                          num_beams=1, pad_token_id=2)
        gen = out[0, ids.shape[1]:].tolist()
        logits = hf(ids).logits[0, -1].double().numpy()
    hf_cache.append(dict(ids=id_list, gen=gen, logits=logits))

result = {}
for backend in (sys.argv[1:] or ["cpu", "metal"]):
    r = run_backend(backend, hf_cache)
    result[backend] = r
    print(f"[{backend}] identical {r['identical']}/{r['total']}  "
          f"logit max {r['logit_max_overall']:.3e}  mean {r['logit_mean_overall']:.3e}")

OUT.mkdir(parents=True, exist_ok=True)
json.dump(result, open(OUT / "correctness.json", "w"), indent=2)
print("saved", OUT / "correctness.json")
