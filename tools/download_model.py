"""Download and verify a pinned pretrained Llama checkpoint. No model code runs."""

import argparse
import hashlib
import json
import pathlib
import time
import urllib.error
import urllib.request


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models/tinystories-15m")
    args = parser.parse_args()
    lock = json.loads(pathlib.Path(__file__).with_name("model-lock.json").read_text())
    output = pathlib.Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, expected in lock["files"].items():
        path = output / name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".partial")
            url = f"https://huggingface.co/{lock['repository']}/resolve/{lock['revision']}/{name}?download=true"
            print("Downloading", name, flush=True)
            for attempt in range(3):
                try:
                    with (
                        urllib.request.urlopen(url, timeout=120) as response,
                        temporary.open("wb") as target,
                    ):
                        while chunk := response.read(1024 * 1024):
                            target.write(chunk)
                    break
                except urllib.error.HTTPError as error:
                    if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                        raise
                    time.sleep(3 * (attempt + 1))
            if (
                temporary.stat().st_size != expected["bytes"]
                or digest(temporary) != expected["sha256"]
            ):
                raise RuntimeError(f"Checksum mismatch: {name}")
            temporary.replace(path)
        if (
            path.stat().st_size != expected["bytes"]
            or digest(path) != expected["sha256"]
        ):
            raise RuntimeError(f"Existing file differs from checkpoint lock: {path}")
    (output / "download-manifest.json").write_text(json.dumps(lock, indent=2) + "\n")
    print("All checkpoint files verified.")


if __name__ == "__main__":
    main()
