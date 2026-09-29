#!/usr/bin/env python3
"""Fail unless the installed vLLM is 0.17.1 and carries the overlay's exact files.

    python check_overlay.py --manifest <overlay>/patch-manifest.json

Run it with the same interpreter that will run ``vllm serve``.
"""

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    import vllm

    manifest = json.loads(args.manifest.read_text())
    if vllm.__version__ != manifest["base_version"]:
        raise SystemExit(f"vLLM {vllm.__version__} != {manifest['base_version']}")
    root = Path(vllm.__file__).resolve().parent.parent
    for name, hashes in manifest["files"].items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != hashes["patched_sha256"]:
            raise SystemExit(f"{name}: installed file is not the overlay file")
    print(f"ok: vLLM {vllm.__version__}, patch {manifest.get('patch')}")


if __name__ == "__main__":
    main()
