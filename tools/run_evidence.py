"""Execute comparable local measurements, retaining commands and outputs. Run after CTest."""

import argparse
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("--build", default="build-audit")
p.add_argument("--output", default="benchmarks/results")
a = p.parse_args()
out = pathlib.Path(a.output)
out.mkdir(parents=True, exist_ok=True)
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
record = {"timestamp": stamp, "commands": [], "artifacts": {}}
model = "models/tinystories-15m"


def run(key, command):
    log = out / f"{stamp}-{key}.log"
    record["commands"].append(command)
    print("Running:", " ".join(command), flush=True)
    with log.open("w") as f:
        subprocess.run(
            command, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=180
        )


run("checkpoint", [sys.executable, "tools/download_model.py"])
for backend in ["cpu", "metal"]:
    key = f"validation-{backend}"
    path = out / f"{stamp}-{key}.json"
    run(
        key,
        [
            sys.executable,
            "tools/validate.py",
            "--model",
            model,
            "--executable",
            f"{a.build}/kestrel_trace",
            "--backend",
            backend,
            "--tokens",
            "1,9038,2501,263,931,29889",
            "--output",
            str(path),
        ],
    )
    record["artifacts"][key] = path.name
    run(
        f"tokenizer-{backend}",
        [
            sys.executable,
            "tests/tokenizer_integration.py",
            "--binary",
            f"{a.build}/kestrel",
            "--model",
            model,
            "--backend",
            backend,
        ],
    )
# Execute sequentially: no competing tests/builds during timed inference.
for profile in [False, True]:
    for backend in ["cpu", "metal"]:
        key = f"{'profile' if profile else 'benchmark'}-{backend}"
        path = out / f"{stamp}-{key}.json"
        command = [
            f"{a.build}/kestrel_bench",
            "--model",
            model,
            "--backend",
            backend,
            "--prompt-tokens",
            "16",
            "--max-tokens",
            "32",
            "--warmup",
            "2",
            "--iterations",
            "5",
            "--json",
            str(path),
        ]
        if profile:
            command.append("--profile")
        run(key, command)
        record["artifacts"][key] = path.name
sources = sorted(
    f
    for directory in ["src", "include", "benchmarks", "tools", "tests"]
    for f in pathlib.Path(directory).rglob("*")
    if f.is_file()
    and f.suffix in {".cpp", ".hpp", ".mm", ".metal", ".cu", ".cuh", ".py"}
    and "results" not in f.parts
)
sources.append(pathlib.Path("CMakeLists.txt"))
record["source_sha256"] = {
    str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources
}
(out / "latest.json").write_text(json.dumps(record, indent=2) + "\n")
print("Evidence manifest:", out / "latest.json")
