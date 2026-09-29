#!/usr/bin/env python3
"""Supervise the frozen instinct-dual-4b service: two vLLM servers plus the adapter.

Starts one native-readout-patched vLLM 0.17.1 server per GPU (the first two
entries of CUDA_VISIBLE_DEVICES), waits for both, starts the SystemOne adapter
(common/serve_adapter.py) in dual-order mode, runs a semantic smoke request and
writes the ready file. Adapted from the original deployment supervisor.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request



def dual_gpu_devices(environ: dict[str, str]) -> tuple[str, str]:
    """Return the two visible device selectors or fail closed."""
    raw = environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = tuple(item.strip() for item in raw.split(",") if item.strip())
    if (
        len(devices) < 2
        or any(item in ("-1", "NoDevFiles") for item in devices)
        or devices[0] == devices[1]
    ):
        raise RuntimeError(
            "the dual-backend arm requires at least two distinct entries in "
            "CUDA_VISIBLE_DEVICES"
        )
    return devices[0], devices[1]


def gpu_identity(selector: str, environ: dict[str, str]) -> dict[str, str]:
    """Resolve a CUDA selector to auditable physical GPU identity."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--id",
            selector,
            "--query-gpu=index,uuid,name",
            "--format=csv,noheader",
        ],
        env=environ,
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"could not resolve CUDA device {selector!r}")
    fields = [field.strip() for field in rows[0].split(",", 2)]
    if len(fields) != 3:
        raise RuntimeError(f"invalid nvidia-smi identity for CUDA device {selector!r}")
    return {"selector": selector, "index": fields[0], "uuid": fields[1], "name": fields[2]}


EXPECTED_CONFIG_KEYS = {
    "schema_version",
    "public_model_id",
    "served_name",
    "model_path",
    "model_revision",
    "reflex_revision",
    "execution_mode",
    "readout",
    "backend_batch_size",
    "max_model_len",
    "max_num_batched_tokens",
    "max_num_seqs",
    "gpu_memory_utilization",
    "prefix_caching",
}


def load_config(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != EXPECTED_CONFIG_KEYS:
        raise ValueError("deployment config keys differ from schema v1")
    if value["schema_version"] != 2:
        raise ValueError("unsupported deployment config schema")
    if value["execution_mode"] != "eager":
        raise ValueError("production deployment is pinned to eager execution")
    if value["readout"] != "native-padded" or value["backend_batch_size"] != 1:
        raise ValueError("deployment requires native-padded batch-one readout")
    if value["prefix_caching"] is not False:
        raise ValueError("deployment requires prefix caching disabled")
    for key in ("max_model_len", "max_num_batched_tokens", "max_num_seqs"):
        if not isinstance(value[key], int) or value[key] < 1:
            raise ValueError(f"invalid positive integer: {key}")
    utilization = value["gpu_memory_utilization"]
    if not isinstance(utilization, (int, float)) or not 0 < utilization < 1:
        raise ValueError("invalid gpu_memory_utilization")
    for key in (
        "public_model_id",
        "served_name",
        "model_path",
        "model_revision",
        "reflex_revision",
    ):
        if not isinstance(value[key], str) or not value[key]:
            raise ValueError(f"invalid non-empty string: {key}")
    if value["public_model_id"] == value["served_name"]:
        raise ValueError("public_model_id must not expose the backend served_name")
    return value


def free_port(excluded: set[int] | None = None) -> int:
    excluded = excluded or set()
    for _ in range(32):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        if port not in excluded:
            return port
    raise RuntimeError("could not allocate a distinct backend port")


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def request_ok(url: str, timeout: float = 2) -> None:
    with urllib.request.urlopen(
        urllib.request.Request(url), timeout=timeout
    ) as response:
        response.read()


def request_json(url: str, body: dict, timeout: float = 120) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def startup_smoke(port: int, model: str) -> dict:
    response = request_json(
        f"http://127.0.0.1:{port}/v1/systemone",
        {
            "model": model,
            "state": "The payment service is down and customers cannot pay.",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Which team owns this incident?",
                    "criteria": {
                        "engineering": "Technical service failures",
                        "billing": "Invoice questions",
                        "other": "Everything else",
                    },
                }
            },
        },
    )
    if response.get("answers", {}).get("route", {}).get("choice") != "engineering":
        raise RuntimeError("startup semantic smoke failed")
    return response


def wait_ready(
    url: str,
    process: subprocess.Popen,
    stopping: threading.Event | None = None,
) -> None:
    for _ in range(480):
        if stopping is not None and stopping.is_set():
            raise RuntimeError("startup interrupted by shutdown request")
        if process.poll() is not None:
            raise RuntimeError(f"service exited before readiness: {process.returncode}")
        try:
            request_ok(url)
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"service did not become ready: {url}")


def start(command: list[str], log_path: Path, environ: dict[str, str]) -> subprocess.Popen:
    handle = log_path.open("w", encoding="utf-8")
    try:
        return subprocess.Popen(
            command,
            env=environ,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        handle.close()


def stop(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def backend_command(config: dict, port: int) -> list[str]:
    return [
        "vllm",
        "serve",
        config["model_path"],
        "--served-model-name",
        config["served_name"],
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        str(config["max_model_len"]),
        "--max-num-batched-tokens",
        str(config["max_num_batched_tokens"]),
        "--max-num-seqs",
        str(config["max_num_seqs"]),
        "--max-logprobs",
        "32",
        "--gpu-memory-utilization",
        str(config["gpu_memory_utilization"]),
        "--no-async-scheduling",
        "--disable-uvicorn-access-log",
        "--trust-remote-code",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--enforce-eager",
        "--no-enable-prefix-caching",
    ]


def production_environment(environ: dict[str, str]) -> dict[str, str]:
    result = dict(environ)
    result.update(
        VLLM_NO_USAGE_STATS="1",
        VLLM_BATCH_INVARIANT="0",
        DO_NOT_TRACK="1",
        TOKENIZERS_PARALLELISM="false",
        # The patched scheduler requires this sink. Per-request audits are
        # deliberately disabled in the long-running deployment.
        JQV_PREFILL_AUDIT=os.devnull,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument(
        "--model-path", help="local Qwen3.5-4B directory; overrides the config value"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model_path:
        config["model_path"] = args.model_path
    args.runtime_dir.mkdir(parents=True, exist_ok=False)
    args.ready_file.parent.mkdir(parents=True, exist_ok=True)
    args.ready_file.unlink(missing_ok=True)
    model_path = Path(config["model_path"])
    if not model_path.is_dir():
        raise SystemExit(f"model path does not exist: {model_path}")

    environ = dict(os.environ)
    environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    selectors = dual_gpu_devices(environ)
    identities = tuple(gpu_identity(selector, environ) for selector in selectors)
    if identities[0]["uuid"] == identities[1]["uuid"]:
        raise SystemExit("both CUDA selectors resolved to the same GPU")

    primary_port = free_port({args.port})
    secondary_port = free_port({args.port, primary_port})
    ports = (primary_port, secondary_port)
    primary = secondary = adapter = None
    stopping = threading.Event()

    def request_stop(_signum, _frame) -> None:
        stopping.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    base_env = production_environment(environ)
    try:
        primary_env = dict(base_env)
        primary_env["CUDA_VISIBLE_DEVICES"] = selectors[0]
        primary = start(
            backend_command(config, ports[0]),
            args.runtime_dir / "vllm-primary.log",
            primary_env,
        )
        wait_ready(
            f"http://127.0.0.1:{ports[0]}/health",
            primary,
            stopping,
        )

        secondary_env = dict(base_env)
        secondary_env["CUDA_VISIBLE_DEVICES"] = selectors[1]
        secondary = start(
            backend_command(config, ports[1]),
            args.runtime_dir / "vllm-secondary.log",
            secondary_env,
        )
        wait_ready(
            f"http://127.0.0.1:{ports[1]}/health",
            secondary,
            stopping,
        )

        adapter_env = dict(base_env)
        adapter = start(
            [
                sys.executable,
                str(Path(__file__).resolve().parent.parent / "common" / "serve_adapter.py"),
                "--model",
                "instinct-dual-4b",
                "--weights",
                config["model_path"],
                "--model-revision",
                config["model_revision"],
                "--reflex-revision",
                config["reflex_revision"],
                "--backend",
                f"http://127.0.0.1:{ports[0]}",
                "--backend-secondary",
                f"http://127.0.0.1:{ports[1]}",
                "--backend-model",
                config["served_name"],
                "--served-name",
                config["public_model_id"],
                "--readout",
                config["readout"],
                "--backend-batch-size",
                str(config["backend_batch_size"]),
                "--host",
                args.host,
                "--port",
                str(args.port),
            ],
            args.runtime_dir / "adapter.log",
            adapter_env,
        )
        wait_ready(
            f"http://127.0.0.1:{args.port}/health", adapter, stopping=stopping
        )
        write_json(
            args.runtime_dir / "startup-smoke.json",
            startup_smoke(args.port, config["public_model_id"]),
        )

        config_sha = hashlib.sha256(args.config.read_bytes()).hexdigest()
        write_json(
            args.ready_file,
            {
                "status": "ready",
                "host": args.host,
                "node": socket.gethostname(),
                "port": args.port,
                "served_name": config["public_model_id"],
                "public_model_id": config["public_model_id"],
                "backend_model_id": config["served_name"],
                "execution_mode": config["execution_mode"],
                "config_sha256": config_sha,
                "gpus": list(identities),
                "processes": {
                    "adapter": adapter.pid,
                    "primary": primary.pid,
                    "secondary": secondary.pid,
                },
            },
        )
        while not stopping.wait(1):
            for name, process in (
                ("primary", primary),
                ("secondary", secondary),
                ("adapter", adapter),
            ):
                if process.poll() is not None:
                    raise RuntimeError(
                        f"{name} exited unexpectedly: {process.returncode}"
                    )
    finally:
        args.ready_file.unlink(missing_ok=True)
        stop(adapter)
        stop(secondary)
        stop(primary)


if __name__ == "__main__":
    main()
