"""Audit executed evidence and generate documentation tables without fallback results."""

import argparse
import hashlib
import json
import math
import pathlib
import re

parser = argparse.ArgumentParser()
parser.add_argument(
    "--check", action="store_true", help="fail if generated documentation differs"
)
args = parser.parse_args()
root = pathlib.Path(__file__).resolve().parents[1]
results = root / "benchmarks/results"
manifest = json.loads((results / "latest.json").read_text())
# A missing Git checkout must not make native-source drift invisible.
for filename, digest in manifest["source_sha256"].items():
    if (
        filename.startswith(("src/", "include/", "benchmarks/"))
        or filename == "CMakeLists.txt"
    ):
        assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == digest, (
            filename
        )

artifacts = {
    k: json.loads((results / f).read_text()) for k, f in manifest["artifacts"].items()
}


def link(key, in_docs=True):
    return (
        ("../" if in_docs else "") + "benchmarks/results/" + manifest["artifacts"][key]
    )


def close(a, b):
    assert (
        math.isfinite(a)
        and math.isfinite(b)
        and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
    ), (a, b)


for key, j in artifacts.items():
    if key.startswith("validation"):
        assert j["passed"]
        for m in j["stages"].values():
            assert m["max_absolute_error"] <= j["tolerances"]["max_absolute"]
            assert m["mean_absolute_error"] <= j["tolerances"]["mean_absolute"]
            assert m["cosine_similarity"] >= j["tolerances"]["cosine_min"]
        continue
    b = j["benchmark"]
    assert j["schema_version"] == 2 and j["build_type"] == "Release"
    assert len(j["runs"]) == b["measured_runs"] and b["batch_size"] == 1
    assert len(b["prompt_ids"]) == b["prompt_tokens"]
    samples = sorted(x for r in j["runs"] for x in r["inter_token_ms"])
    assert (
        len(samples)
        == j["inter_token_sample_count"]
        == b["measured_runs"] * (b["generated_tokens"] - 1)
    )
    for r in j["runs"]:
        assert len(r["generated_ids"]) == r["generated_tokens"] == b["generated_tokens"]
        assert (
            len(r["inter_token_ms"]) == r["decode_tokens"] == b["generated_tokens"] - 1
        )
        close(r["decode_elapsed_ms"], sum(r["inter_token_ms"]))
    close(j["prefill_elapsed_ms_total"], sum(r["prefill_ms"] for r in j["runs"]))
    close(j["decode_elapsed_ms_total"], sum(samples))
    close(j["decode_tokens_per_second"], len(samples) * 1000 / sum(samples))
    close(
        j["prefill_tokens_per_second"],
        b["prompt_tokens"] * len(j["runs"]) * 1000 / j["prefill_elapsed_ms_total"],
    )
    close(j["mean_inter_token_ms"], sum(samples) / len(samples))
    for label, q in [("p50", 0.5), ("p95", 0.95), ("p99", 0.99)]:
        close(j[f"{label}_inter_token_ms"], samples[math.ceil(q * len(samples)) - 1])
    assert j["profiling_enabled"] == key.startswith("profile")
    for category in ["profile", "backend_diagnostics"]:
        for v in j[category].values():
            close(v["mean_ms"] * v["calls"], v["total_ms"])
            close(
                v["percent_measured_runtime"],
                100 * v["total_ms"] / j["measured_elapsed_ms_total"],
            )

cpu, metal = (artifacts[f"benchmark-{b}"] for b in ["cpu", "metal"])
for key in ["benchmark", "model", "compiler", "build_type", "checkpoint"]:
    assert cpu[key] == metal[key], key
for key in ["benchmark-cpu", "benchmark-metal", "profile-cpu", "profile-metal"]:
    assert artifacts[key]["benchmark"] == cpu["benchmark"]
    assert [r["generated_ids"] for r in artifacts[key]["runs"]] == [
        r["generated_ids"] for r in cpu["runs"]
    ], "greedy trajectories differ"


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join(["---"] * len(headers)) + "|",
        ]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def update(file, marker, body):
    path = root / file
    text = path.read_text()
    replacement = f"<!-- BEGIN {marker} -->\n\n{body}\n\n<!-- END {marker} -->"
    revised, n = re.subn(
        f"<!-- BEGIN {marker} -->.*?<!-- END {marker} -->",
        lambda _: replacement,
        text,
        flags=re.S,
    )
    assert n == 1, (file, marker)
    if args.check:
        assert revised == text, f"Stale generated table: {file}, {marker}"
    else:
        path.write_text(revised)


rows = []
for backend in ["cpu", "metal"]:
    key = f"validation-{backend}"
    m = artifacts[key]["stages"]["logits"]
    rows.append(
        [
            f"[{backend}]({link(key, False)})",
            f"{m['max_absolute_error']:.6e}",
            f"{m['mean_absolute_error']:.6e}",
            f"{m['cosine_similarity']:.15f}",
        ]
    )
update(
    "README.md",
    "NUMERICAL RESULTS",
    table(
        ["Backend", "Max logit error", "Mean logit error", "Cosine similarity"], rows
    ),
)
rows = []
for backend in ["cpu", "metal"]:
    b = artifacts[f"benchmark-{backend}"]
    rows.append(
        [f"[{backend}]({link(f'benchmark-{backend}', False)})"]
        + [
            f"{b[k]:.2f}"
            for k in [
                "prefill_tokens_per_second",
                "decode_tokens_per_second",
                "time_to_first_token_ms",
            ]
        ]
    )
update(
    "README.md",
    "BENCHMARK RESULTS",
    table(["Backend", "Prefill tok/s", "Decode tok/s", "TTFT ms"], rows),
)

system, model = cpu["system"], cpu["model"]
rows = [
    [k, system[k]]
    for k in [
        "os",
        "os_build",
        "machine",
        "chip",
        "memory_bytes",
        "hw.physicalcpu",
        "hw.logicalcpu",
        "hw.perflevel0.physicalcpu",
        "hw.perflevel1.physicalcpu",
    ]
]
rows += [
    [k, cpu[k]]
    for k in ["timestamp", "compiler", "cmake_version", "build_type", "git_commit"]
]
rows += [[f"model.{k}", v] for k, v in model.items() if k != "weight_dtypes"]
update("docs/performance.md", "ENVIRONMENT", table(["Field", "Executed value"], rows))
rows = []
for backend in ["cpu", "metal"]:
    b = artifacts[f"benchmark-{backend}"]
    for label, field in [
        ("Prefill elapsed mean (ms)", "prefill_elapsed_ms_mean"),
        ("Prefill throughput (tok/s)", "prefill_tokens_per_second"),
        ("Decode elapsed mean (ms)", "decode_elapsed_ms_mean"),
        ("Decode throughput (tok/s)", "decode_tokens_per_second"),
        ("TTFT mean (ms)", "time_to_first_token_ms"),
        ("Inter-token mean (ms)", "mean_inter_token_ms"),
        ("Inter-token p50 (ms)", "p50_inter_token_ms"),
        ("Inter-token p95 (ms)", "p95_inter_token_ms"),
        ("Inter-token p99 (ms)", "p99_inter_token_ms"),
        ("Inter-token min (ms)", "min_inter_token_ms"),
        ("Inter-token max (ms)", "max_inter_token_ms"),
    ]:
        if backend == "cpu":
            rows.append([label, f"{b[field]:.2f}"])
        else:
            rows[next(i for i, r in enumerate(rows) if r[0] == label)].append(
                f"{b[field]:.2f}"
            )
update(
    "docs/performance.md",
    "PERFORMANCE",
    table(
        [
            "Metric",
            f"[CPU]({link('benchmark-cpu')})",
            f"[Metal]({link('benchmark-metal')})",
        ],
        rows,
    ),
)
update(
    "docs/performance.md",
    "MEMORY",
    table(
        ["Bytes", "CPU", "Metal"],
        [
            [k, cpu[k], metal[k]]
            for k in [
                "model_weight_bytes",
                "backend_resident_weight_bytes",
                "kv_cache_bytes",
                "peak_process_memory_bytes",
            ]
        ],
    ),
)
profiles = []
for backend in ["cpu", "metal"]:
    b = artifacts[f"profile-{backend}"]
    rows = [
        [
            name,
            v["calls"],
            f"{v['total_ms']:.2f}",
            f"{v['mean_ms']:.4f}",
            f"{v['percent_measured_runtime']:.2f}",
        ]
        for name, v in b["profile"].items()
    ]
    remainder = b["measured_elapsed_ms_total"] - sum(
        v["total_ms"] for v in b["profile"].values()
    )
    profiles.append(
        f"**[{backend}]({link(f'profile-{backend}')})**\n\n"
        + table(["Operation", "Calls", "Total ms", "Mean ms", "Runtime %"], rows)
        + f"\n\nUnattributed elapsed time: {remainder:.2f} ms ({100 * remainder / b['measured_elapsed_ms_total']:.2f}%)."
    )
update("docs/performance.md", "PROFILE", "\n\n".join(profiles))
b = artifacts["profile-metal"]
update(
    "docs/performance.md",
    "DIAGNOSTICS",
    table(
        ["Nested host category", "Calls", "Total ms", "Mean ms", "Runtime %"],
        [
            [
                k,
                v["calls"],
                f"{v['total_ms']:.2f}",
                f"{v['mean_ms']:.4f}",
                f"{v['percent_measured_runtime']:.2f}",
            ]
            for k, v in b["backend_diagnostics"].items()
        ],
    ),
)
wait = b["backend_diagnostics"]["Waiting"]["percent_measured_runtime"]
comparison = f"**MEASURED FACT:** Metal decode was {metal['decode_tokens_per_second']:.2f} tok/s and CPU decode was {cpu['decode_tokens_per_second']:.2f} tok/s; CPU throughput was {cpu['decode_tokens_per_second'] / metal['decode_tokens_per_second']:.2f}× Metal in these unprofiled runs. The profiled Metal run spent {wait:.2f}% of measured generation wall time in existing command waits.\n\n"
comparison += (
    "Profiling changes the amount of host work. The separately executed profiled runs measured "
    + ", ".join(
        f"{backend} {artifacts[f'profile-{backend}']['decode_tokens_per_second']:.2f} tok/s"
        for backend in ["cpu", "metal"]
    )
    + ". These differences include run-to-run variation and cannot isolate profiler overhead."
)
update("docs/performance.md", "COMPARISON", comparison)

v = artifacts["validation-cpu"]
update(
    "docs/correctness.md",
    "REFERENCE",
    f"Executed reference: **{v['reference']}, PyTorch {v['pytorch_version']}**, dtype `{v['reference_dtype']}`. Checkpoint SHA256: `{next(iter(v['checkpoint_sha256'].values()))}`.",
)
rows = []
for backend in ["cpu", "metal"]:
    for stage, m in artifacts[f"validation-{backend}"]["stages"].items():
        rows.append(
            [
                f"[{backend}]({link(f'validation-{backend}')})",
                stage,
                f"{m['max_absolute_error']:.6e}",
                f"{m['mean_absolute_error']:.6e}",
                f"{m['cosine_similarity']:.15f}",
            ]
        )
update(
    "docs/correctness.md",
    "STAGES",
    table(
        ["Backend", "Stage", "Max abs error", "Mean abs error", "Cosine similarity"],
        rows,
    ),
)
rows = []
for label, filename, count in [
    ("CPU Release", "cpu-host-tests.log", 7),
    ("CPU + Metal Release", "release-device-tests.log", 11),
    ("CPU Debug UBSan", "ubsan-strict-tests.log", 7),
]:
    assert (
        f"100% tests passed, 0 tests failed out of {count}"
        in (results / filename).read_text()
    )
    rows.append(
        [label, f"{count}/{count} passed", f"[log](../benchmarks/results/{filename})"]
    )
rows.append(
    [
        "ASan Debug",
        "Compiled; execution timed out before main",
        "[probe](../benchmarks/results/asan-probe.json)",
    ]
)
update(
    "docs/correctness.md", "TESTS", table(["Configuration", "Result", "Evidence"], rows)
)
print(
    "Evidence arithmetic, comparison inputs, greedy trajectories, and documentation tables verified."
)
