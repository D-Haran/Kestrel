"""Tokenization only. Never imports or invokes a transformer/inference framework.

One JSON request on stdin, one JSON response on stdout. Hugging Face tokenizer.json
uses the tokenizers library; tokenizer.model uses the SentencePiece library.
"""

import json
import pathlib
import sys


def main():
    root = pathlib.Path(sys.argv[1])
    req = json.load(sys.stdin)
    if (root / "tokenizer.json").exists():
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(root / "tokenizer.json"))
        result = (
            tok.encode(req["text"], add_special_tokens=True).ids
            if req["operation"] == "encode"
            else tok.decode(req["ids"], skip_special_tokens=True)
        )
    else:
        import sentencepiece as spm

        tok = spm.SentencePieceProcessor(model_file=str(root / "tokenizer.model"))
        config = (
            json.loads((root / "tokenizer_config.json").read_text())
            if (root / "tokenizer_config.json").exists()
            else {}
        )
        if req["operation"] == "encode":
            result = tok.encode(
                req["text"],
                out_type=int,
                add_bos=config.get("add_bos_token", True),
                add_eos=config.get("add_eos_token", False),
            )
        else:
            result = tok.decode(req["ids"])
    json.dump(result, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
