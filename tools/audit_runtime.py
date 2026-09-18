"""Fail if native sources or the tokenizer helper acquire forbidden inference calls."""

import ast
import pathlib
import re

root = pathlib.Path(__file__).resolve().parents[1]
forbidden = re.compile(
    r"(torch/|torch::|transformers[./]|llama\.h|llama_|onnxruntime|Ort::|mlx[/:]|TensorRT|nvinfer|vllm)",
    re.I,
)
for directory in ["src", "include", "benchmarks"]:
    for path in (root / directory).rglob("*"):
        if path.is_file():
            for n, line in enumerate(path.read_text().splitlines(), 1):
                if forbidden.search(line):
                    raise SystemExit(
                        f"Forbidden runtime dependency: {path}:{n}: {line}"
                    )
helper = root / "tools/tokenizer_bridge.py"
for node in ast.walk(ast.parse(helper.read_text())):
    if isinstance(node, ast.Import):
        names = [x.name.split(".")[0] for x in node.names]
    elif isinstance(node, ast.ImportFrom):
        names = [(node.module or "").split(".")[0]]
    else:
        continue
    if not set(names) <= {"json", "pathlib", "sys", "tokenizers", "sentencepiece"}:
        raise SystemExit(f"Unexpected tokenizer import: {names}")
print(
    "Native inference dependency audit passed; external libraries are confined to validation/tokenization."
)
