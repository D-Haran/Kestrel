"""Optional pretrained text/interactive integration; uses only tokenizer dependencies."""

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

p = argparse.ArgumentParser()
p.add_argument("--binary", required=True)
p.add_argument("--model", required=True)
p.add_argument("--backend", default="cpu")
a = p.parse_args()
helper = pathlib.Path(__file__).resolve().parents[1] / "tools/tokenizer_bridge.py"
text = "Once upon a time, a child said “héllo”."


def bridge(request):
    return json.loads(
        subprocess.check_output(
            [sys.executable, str(helper), a.model], input=json.dumps(request), text=True
        )
    )


ids = bridge({"operation": "encode", "text": text})
assert ids and bridge({"operation": "decode", "ids": ids}) == text
with tempfile.TemporaryDirectory(prefix="kestrel-text-") as temp:
    output = pathlib.Path(temp) / "result.json"
    from tokenizers import Tokenizer, models, pre_tokenizers

    json_root = pathlib.Path(temp) / "json-tokenizer"
    json_root.mkdir()
    tok = Tokenizer(
        models.WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]")
    )
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(json_root / "tokenizer.json"))
    for request, expected in [
        ({"operation": "encode", "text": "hello world"}, [1, 2]),
        ({"operation": "decode", "ids": [1, 2]}, "hello world"),
    ]:
        response = subprocess.check_output(
            [sys.executable, str(helper), str(json_root)],
            input=json.dumps(request),
            text=True,
        )
        assert json.loads(response) == expected
    common = [
        a.binary,
        "--model",
        a.model,
        "--backend",
        a.backend,
        "--max-tokens",
        "4",
        "--tokenizer-python",
        sys.executable,
    ]
    subprocess.run(
        common + ["--prompt", text, "--json", str(output)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = json.loads(output.read_text())
    assert result["prompt_ids"] == ids
    r = subprocess.run(
        common + ["--interactive"],
        input="Once upon a time\n/quit\n",
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert (
        r.returncode == 0 and "error:" not in r.stderr and ", there was" in r.stdout
    ), (r.stdout, r.stderr)
    r = subprocess.run(
        common + ["--prompt", text, "--tokenizer-helper", "/does/not/exist"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode != 0 and "tokenizer helper failed" in r.stderr
print(
    "Unicode tokenization, text CLI, interactive mode, and helper error handling passed"
)
