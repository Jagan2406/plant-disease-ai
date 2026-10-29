"""Report Python / PyTorch / CUDA environment. Safe when CUDA is unavailable."""

from __future__ import annotations

import sys

try:
    import torch
except ImportError as exc:
    print("Missing dependency: torch")
    print("Install a CPU or CUDA PyTorch build for this machine, then re-run this command.")
    raise SystemExit(1) from exc


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def collect_device_info() -> dict[str, str]:
    device = get_device()
    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda if cuda_available else "N/A"
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    try:
        import torchvision

        torchvision_version = torchvision.__version__
    except ImportError:
        torchvision_version = "not installed"

    return {
        "Python version": sys.version.split()[0],
        "PyTorch version": torch.__version__,
        "torchvision version": torchvision_version,
        "CUDA available": str(cuda_available),
        "CUDA version": str(cuda_version),
        "GPU name": gpu_name,
        "selected device": str(device),
    }


def print_device_info() -> None:
    info = collect_device_info()
    for key, value in info.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    print_device_info()
