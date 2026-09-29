#!/usr/bin/env python3
"""Derive a hash-pinned native-padded overlay from the verified native-readout overlay.

The output files are derived from vLLM 0.17.1 (Apache-2.0); see ../NOTICE.
Usage: prepare_padded_overlay.py --source <overlay from common/build_overlay.py> --output <new dir>
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


REPLACEMENTS = {
    "vllm/entrypoints/openai/completion/serving.py": (
        (
            'if jqv_args.get("jqv_readout") == "native-v1":',
            'if jqv_args.get("jqv_readout") in ("native-v1", "native-padded-v1"):',
        ),
        (
            "jqv_readout_position=prompt_len - 1))",
            'jqv_readout_position=prompt_len - (2 if jqv_args.get("jqv_readout") == "native-padded-v1" else 1)))',
        ),
    ),
    "vllm/v1/worker/gpu_model_runner.py": (
        (
            'if jqv_args.get("jqv_readout") == "native-v1":',
            'if jqv_args.get("jqv_readout") in ("native-v1", "native-padded-v1"):',
        ),
        (
            "last_hidden = hidden_states[offset + num_tokens - 1:offset + num_tokens]",
            'readout_offset = 2 if jqv_args.get("jqv_readout") == "native-padded-v1" else 1\n'
            "                last_hidden = hidden_states[\n"
            "                    offset + num_tokens - readout_offset:\n"
            "                    offset + num_tokens - readout_offset + 1\n"
            "                ]",
        ),
    ),
    "vllm/v1/core/sched/scheduler.py": (
        (
            '"readout_position": request.num_prompt_tokens - 1}',
            '"readout_position": request.num_prompt_tokens - (2 if (request.sampling_params.extra_args or {}).get("jqv_readout") == "native-padded-v1" else 1)}',
        ),
        (
            '"computed_tokens": request.num_computed_tokens,',
            '"computed_tokens": request.num_computed_tokens,\n'
            '                         "cached_tokens": request.num_cached_tokens,',
        ),
    ),
    "vllm/sampling_params.py": (
        (
            "self.skip_reading_prefix_cache = self.prompt_logprobs is not None",
            'jqv_readout = (self.extra_args or {}).get("jqv_readout")\n'
            "            self.skip_reading_prefix_cache = (\n"
            "                self.prompt_logprobs is not None\n"
            '                and jqv_readout not in ("native-v1", "native-padded-v1")\n'
            "            )",
        ),
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text: str, old: str, new: str, path: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match in {path}, found {count}: {old!r}")
    return text.replace(old, new)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")

    manifest = json.loads((args.source / "patch-manifest.json").read_text())
    for name, hashes in manifest["files"].items():
        actual = sha256(args.source / name)
        if actual != hashes["patched_sha256"]:
            raise SystemExit(f"source overlay hash mismatch: {name}")

    shutil.copytree(args.source, args.output)
    for name, replacements in REPLACEMENTS.items():
        path = args.output / name
        text = path.read_text(encoding="utf-8")
        for old, new in replacements:
            text = replace_once(text, old, new, name)
        path.write_text(text, encoding="utf-8")

    derived = {
        "base_version": manifest["base_version"],
        "patch": "reflex-native-padded-cache-v1",
        "purpose": (
            "Native candidate readout from a penultimate position with one ignored "
            "token, preserving the legacy sequence shape without candidate expansion; "
            "native readouts may consume automatic-prefix-cache blocks."
        ),
        "source_patch": manifest["patch"],
        "files": {},
    }
    for name, hashes in manifest["files"].items():
        derived["files"][name] = {
            "original_sha256": hashes["patched_sha256"],
            "patched_sha256": sha256(args.output / name),
        }
    (args.output / "patch-manifest.json").write_text(
        json.dumps(derived, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(derived, sort_keys=True))


if __name__ == "__main__":
    main()
