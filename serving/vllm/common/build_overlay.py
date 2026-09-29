#!/usr/bin/env python3
"""Build the native-readout overlay: five patched vLLM 0.17.1 files.

Copies the five original files from an installed (or checked-out) vLLM 0.17.1,
verifies their SHA-256 against patch/patch-manifest.json, applies
patch/native-readout.patch with ``patch -p1``, verifies the patched hashes and
writes ``<output>/patch-manifest.json`` (the format that
instinct-dual-4b/prepare_padded_overlay.py consumes).

The overlay is a directory tree ``<output>/vllm/...``. Use it by copying the
files over the installed vLLM package, or by bind-mounting each file
read-only over the same path inside a container. Nothing is modified in place.

Derived from vLLM (Apache-2.0); see ../NOTICE.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
PATCH_DIR = HERE.parent / "patch"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vllm-root",
        type=Path,
        required=True,
        help="directory that CONTAINS the 'vllm' package (e.g. site-packages) "
        "or a vLLM v0.17.1 checkout root",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    manifest = json.loads((PATCH_DIR / "patch-manifest.json").read_text())
    if manifest["base_version"] != "0.17.1":
        raise SystemExit("unexpected manifest base version")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name, hashes in manifest["files"].items():
            source = args.vllm_root / name
            if not source.is_file():
                raise SystemExit(f"missing vLLM source file: {source}")
            if sha256(source) != hashes["original_sha256"]:
                raise SystemExit(
                    f"{name} is not the vLLM 0.17.1 original (hash mismatch)"
                )
            (work / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, work / name)
        subprocess.run(
            ["patch", "-p1", "-i", str(PATCH_DIR / "native-readout.patch")],
            cwd=work,
            check=True,
        )
        for name, hashes in manifest["files"].items():
            if sha256(work / name) != hashes["patched_sha256"]:
                raise SystemExit(f"patched hash mismatch: {name}")
        shutil.copytree(work, args.output)
    shutil.copy2(PATCH_DIR / "patch-manifest.json", args.output / "patch-manifest.json")
    print(f"overlay written to {args.output}")


if __name__ == "__main__":
    main()
