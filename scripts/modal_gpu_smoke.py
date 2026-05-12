"""Opt-in Modal entrypoint for the tiny remote GPU validation sweep."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

app = modal.App("cola-dlm-gpu-smoke")
image = (
    modal.Image.debian_slim()
    .pip_install("torch")
    .add_local_python_source("cola_dlm")
)


@app.function(
    image=image,
    gpu="T4",
    max_containers=1,
    min_containers=0,
    scaledown_window=30,
    timeout=300,
)
def run_gpu_smoke() -> dict[str, Any]:
    import torch

    from cola_dlm.modal_gpu_smoke import run_tiny_modal_gpu_validation

    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count()
    cuda_current_device_index = (
        torch.cuda.current_device()
        if cuda_available and cuda_device_count > 0
        else None
    )
    cuda_device_name = (
        torch.cuda.get_device_name(0) if cuda_device_count > 0 else None
    )
    if not cuda_available:
        raise RuntimeError("Modal GPU smoke test requires CUDA, but CUDA is unavailable")

    result = run_tiny_modal_gpu_validation(device="cuda", require_cuda=True)
    result["cuda_metadata"] = {
        "available": bool(cuda_available),
        "device_count": int(cuda_device_count),
        "current_device_index": cuda_current_device_index,
        "device_name": cuda_device_name,
    }
    return result


@app.local_entrypoint()
def main() -> None:
    result = run_gpu_smoke.remote()
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    if not result.get("success", False):
        raise SystemExit(1)
