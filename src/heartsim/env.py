"""Environment report: lets a reviewer (or you) see what hardware/software a result came from,
and lets run sizes be matched to the machine.  Best-effort and cross-platform; any field that
cannot be determined is None rather than a guess."""
from __future__ import annotations

import ctypes
import importlib
import os
import platform
import shutil
import subprocess
import sys


def _ram_gb():
    try:
        import psutil
        return round(psutil.virtual_memory().total / 2 ** 30, 1)
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) / 2 ** 20, 1)
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            class MS(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("m", ctypes.c_ulong), ("tp", ctypes.c_ulonglong),
                            ("ap", ctypes.c_ulonglong), ("tpf", ctypes.c_ulonglong),
                            ("apf", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong),
                            ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
            s = MS()
            s.l = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
            return round(s.tp / 2 ** 30, 1)
        except Exception:
            pass
    return None


def _gpu():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines() if out.returncode == 0 else None
    except Exception:
        return None


def _git_commit(cwd=None):
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10, cwd=cwd)
        if out.returncode != 0:
            return None
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                               timeout=10, cwd=cwd).stdout.strip() != ""
        return out.stdout.strip() + ("+dirty" if dirty else "")
    except Exception:
        return None


def env_info(repo_dir=None) -> dict:
    pkgs = {}
    for name in ("numpy", "scipy", "sklearn", "pandas", "torch", "psutil"):
        try:
            pkgs[name] = getattr(importlib.import_module(name), "__version__", "?")
        except Exception:
            pkgs[name] = None
    return dict(python=sys.version.split()[0], platform=platform.platform(),
                machine=platform.machine(), cpu_count=os.cpu_count(), ram_gb=_ram_gb(),
                gpu=_gpu(), packages=pkgs, git_commit=_git_commit(repo_dir))


def sizing_advice(info: dict) -> list[str]:
    """Rough, labelled-as-estimate guidance based on measurements from a 2-core sandbox."""
    notes = []
    cpus = info.get("cpu_count") or 1
    ram = info.get("ram_gb")
    notes.append(f"{cpus} CPU threads detected" + (f", {ram} GB RAM" if ram else ", RAM unknown"))
    notes.append("Simulator cost measured in a 2-core sandbox: ~0.5 s compute per second of audio "
                 "at grid 140, 1 ms frames (cost grows ~ grid_n^3). Expect the same order on a laptop.")
    notes.append("Never store frames: ~70 MB per audio second at grid 140. Use iter_frames / the "
                 "streaming feature path (the smoke command does).")
    if ram and ram < 8:
        notes.append("Under 8 GB RAM: keep --workers 1 and grid_n <= 140.")
    if info.get("gpu"):
        notes.append("NVIDIA GPU found: useful for the later CNN stage; the current simulator is CPU/numpy.")
    else:
        notes.append("No NVIDIA GPU found: classical models are fine; plan CNN training on a small grid or in the cloud.")
    return notes
