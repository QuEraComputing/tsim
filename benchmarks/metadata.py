"""Collect machine and environment metadata for benchmark results.

All collection is best-effort: if a particular piece of information is
unavailable (e.g. ``nvidia-smi`` on a CPU-only machine), it is silently
omitted rather than raising an error.
"""

from __future__ import annotations

import os
import platform
import subprocess
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_git_info() -> dict[str, str | bool]:
    """Return current git commit hash and whether the working tree is dirty."""
    info: dict[str, str | bool] = {}
    try:
        info["git_commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        info["git_dirty"] = bool(status)
    except Exception:
        pass
    return info


def _get_cpu_info() -> dict[str, object]:
    """Return CPU model name, core counts, and total memory (cross-platform)."""
    info: dict[str, object] = {}
    info["cpu_count_logical"] = os.cpu_count()

    system = platform.system()
    try:
        if system == "Darwin":
            info["cpu_model"] = (
                subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
            info["cpu_count_physical"] = int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.physicalcpu"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
            info["memory_gb"] = round(
                int(
                    subprocess.check_output(
                        ["sysctl", "-n", "hw.memsize"],
                        stderr=subprocess.DEVNULL,
                    )
                    .decode()
                    .strip()
                )
                / (1024**3),
                1,
            )
        elif system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        info["cpu_model"] = line.split(":", 1)[1].strip()
                        break
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        info["memory_gb"] = round(kb / (1024**2), 1)
                        break
    except Exception:
        pass
    return info


def _get_device_info() -> list[dict[str, str]]:
    """Return JAX device information (works for CPU, GPU, and TPU backends).

    When NVIDIA GPUs are present, ``nvidia-smi`` is queried for additional
    details (GPU name, VRAM, driver version, CUDA version).
    """
    import jax

    devices: list[dict[str, str]] = []
    for dev in jax.devices():
        entry: dict[str, str] = {
            "id": str(dev.id),
            "platform": str(dev.platform),
            "device_kind": str(dev.device_kind),
        }
        for attr in ("process_index", "core_on_chip"):
            val = getattr(dev, attr, None)
            if val is not None:
                entry[attr] = str(val)
        devices.append(entry)

    # Enrich with nvidia-smi when available
    try:
        smi = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version,cuda_version",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        for i, line in enumerate(smi.splitlines()):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4 and i < len(devices):
                devices[i]["nvidia_name"] = parts[0]
                devices[i]["nvidia_memory_mb"] = parts[1]
                devices[i]["nvidia_driver"] = parts[2]
                devices[i]["cuda_version"] = parts[3]
    except Exception:
        pass

    return devices


def _get_package_versions() -> dict[str, str]:
    """Return versions of key packages."""
    versions: dict[str, str] = {}
    for pkg in ("tsim", "jax", "stim", "numpy"):
        try:
            mod = __import__(pkg)
            versions[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except Exception:
            pass
    return versions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def collect_metadata(machine_label: str) -> dict:
    """Collect comprehensive machine and environment metadata.

    Returns a JSON-serialisable dict with sections for identity, platform,
    CPU, devices (GPU/TPU), package versions, and git state.
    """
    import jax

    meta: dict = {
        # Identity
        "machine_label": machine_label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        # Platform
        "platform": platform.system(),
        "platform_version": platform.version(),
        "platform_release": platform.release(),
        "hostname": platform.node(),
        "architecture": platform.machine(),
        # Python
        "python_version": platform.python_version(),
        # CPU
        **_get_cpu_info(),
        # JAX
        "jax_backend": str(jax.default_backend()),
        "jax_devices": _get_device_info(),
        # Package versions
        **_get_package_versions(),
        # Git
        **_get_git_info(),
    }
    return meta
