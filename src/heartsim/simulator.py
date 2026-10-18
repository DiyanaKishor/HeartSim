"""Time-domain 2D wave-equation membrane simulator (HeartSim core).

    d2z/dt2 = c^2 * laplacian(z) - damping(x, y) * dz/dt + F(t) * kernel(source)

Differences from the prototype wave_sim_v2.py (all found in the QC / audit):
  * frames are emitted on an exact `frame_interval_ms` grid (the prototype gave
    0.85 ms spacing for a requested 1 ms, and the spacing changed with grid size);
  * the sponge defaults are the ones that were measured to work (width 30 cells,
    max damping 400 /s: 0.6% / 0.03% error at 25 / 60 Hz against a 4x larger
    open-domain reference; the old 18 cells / 45 /s left 38% / 28%);
  * frames are streamed (generator) so nothing has to be stored;
  * the update is done on the interior only, with a small truncated source kernel.

The model is linear, non-dispersive in the continuum limit, and has no threshold:
it is NOT a Faraday-wave fluid model.  Treat it as a designed feature transform.
`reliable_band_hz` tells you up to which frequency the grid itself is trustworthy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Iterator, Tuple

import numpy as np


@dataclass(frozen=True)
class SimParams:
    grid_n: int = 140
    c: float = 3.0                    # wave speed, domain-widths per second
    base_damping: float = 0.3         # interior damping, 1/s
    sponge_width: int = 30            # cells over which the absorbing layer ramps up
    sponge_max_damping: float = 400.0  # 1/s at the outer edge (quadratic ramp)
    source_sigma_cells: float = 1.6
    forcing_gain: float = 350.0       # arbitrary until calibrated against hardware
    frame_interval_ms: float = 1.0
    cfl_safety: float = 0.5           # fraction of the 2D stability limit used

    def to_dict(self) -> dict:
        return asdict(self)


def derived(p: SimParams) -> dict:
    """dt, steps per output frame and cell-level quantities for a parameter set."""
    dx = 1.0 / (p.grid_n - 1)
    dt_cfl = p.cfl_safety * dx / (p.c * math.sqrt(2.0))
    frame_s = p.frame_interval_ms / 1000.0
    steps_per_frame = max(1, math.ceil(frame_s / dt_cfl - 1e-12))
    dt = frame_s / steps_per_frame             # <= dt_cfl, so still stable
    return dict(dx=dx, dt=dt, steps_per_frame=steps_per_frame,
                fs_sim=1.0 / dt, nyquist_hz=0.5 / dt,
                courant=p.c * dt / dx, cells_per_s=p.c / dx)


def numerical_dispersion_error(f_hz: float, p: SimParams, diagonal: bool = False) -> float:
    """Relative phase-speed error of the discrete scheme at frequency f (negative = too slow)."""
    d = derived(p)
    k = 2.0 * math.pi * f_hz / p.c
    if diagonal:
        k = k / math.sqrt(2.0)
        arg = d["courant"] * math.sqrt(2.0) * math.sin(k * d["dx"] / 2.0)
    else:
        arg = d["courant"] * math.sin(k * d["dx"] / 2.0)
    arg = min(1.0, max(-1.0, arg))
    kk = 2.0 * math.pi * f_hz / p.c
    v = (2.0 / (kk * p.c * d["dt"])) * math.asin(arg)
    return v - 1.0


def reliable_band_hz(p: SimParams, tol: float = 0.05) -> float:
    """Highest frequency at which the axis-direction phase-speed error is within +-tol."""
    lo, hi = 1.0, derived(p)["nyquist_hz"]
    if abs(numerical_dispersion_error(hi, p)) <= tol:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if abs(numerical_dispersion_error(mid, p)) <= tol:
            lo = mid
        else:
            hi = mid
    return lo


def damping_field(p: SimParams) -> np.ndarray:
    n = p.grid_n
    idx = np.arange(n)
    edge = np.minimum(idx, n - 1 - idx)
    ii, jj = np.meshgrid(edge, edge, indexing="ij")
    dist = np.minimum(ii, jj).astype(float)
    ramp = np.clip((p.sponge_width - dist) / max(p.sponge_width, 1), 0.0, 1.0)
    return p.base_damping + p.sponge_max_damping * ramp ** 2


def _source_window(p: SimParams, i0: int, j0: int):
    """Truncated (5 sigma) normalised Gaussian source kernel as (slice_i, slice_j, weights)."""
    rad = int(math.ceil(5.0 * p.source_sigma_cells))
    i_lo, i_hi = max(1, i0 - rad), min(p.grid_n - 1, i0 + rad + 1)
    j_lo, j_hi = max(1, j0 - rad), min(p.grid_n - 1, j0 + rad + 1)
    ii, jj = np.meshgrid(np.arange(i_lo, i_hi), np.arange(j_lo, j_hi), indexing="ij")
    w = np.exp(-((ii - i0) ** 2 + (jj - j0) ** 2) / (2.0 * p.source_sigma_cells ** 2))
    w /= w.sum()
    return slice(i_lo, i_hi), slice(j_lo, j_hi), w


def iter_frames(audio: np.ndarray, audio_fs: float, params: SimParams | None = None,
                source_xy: Tuple[float, float] = (0.5, 0.5),
                duration_s: float | None = None) -> Iterator[Tuple[float, np.ndarray]]:
    """Yield (t_ms, field) every `frame_interval_ms` of simulated time.

    `audio` must already be preprocessed (see heartsim.preprocess); this function does
    NOT filter, normalise or remove DC.  The yielded array is never mutated afterwards,
    so it is safe to keep a reference, but copy it if you intend to modify it.
    """
    p = params or SimParams()
    d = derived(p)
    dt, spf = d["dt"], d["steps_per_frame"]
    audio = np.asarray(audio, dtype=np.float64)
    if duration_s is None:
        duration_s = len(audio) / audio_fs
    n_steps = int(duration_s / dt)
    t_sim = np.arange(n_steps) * dt
    a_sim = np.interp(t_sim, np.arange(len(audio)) / audio_fs, audio, left=0.0, right=0.0)

    n = p.grid_n
    i0 = int(round(source_xy[1] * (n - 1)))
    j0 = int(round(source_xy[0] * (n - 1)))
    si, sj, w = _source_window(p, i0, j0)
    force_scale = (dt ** 2) * p.forcing_gain
    r = (p.c * dt / d["dx"]) ** 2
    damp_dt = damping_field(p) * dt

    z = np.zeros((n, n))
    z_prev = np.zeros((n, n))
    lap = np.zeros((n, n))
    for step in range(n_steps):
        lap[1:-1, 1:-1] = (z[2:, 1:-1] + z[:-2, 1:-1] + z[1:-1, 2:] + z[1:-1, :-2]
                           - 4.0 * z[1:-1, 1:-1])
        z_new = 2.0 * z - z_prev + r * lap - damp_dt * (z - z_prev)
        z_new[si, sj] += force_scale * a_sim[step] * w
        z_new[0, :] = 0.0
        z_new[-1, :] = 0.0
        z_new[:, 0] = 0.0
        z_new[:, -1] = 0.0
        z_prev, z = z, z_new
        if (step + 1) % spf == 0:
            yield (step + 1) * dt * 1000.0, z


def simulate_frames(audio: np.ndarray, audio_fs: float, params: SimParams | None = None,
                    source_xy: Tuple[float, float] = (0.5, 0.5),
                    duration_s: float | None = None):
    """Convenience wrapper that stores every frame (float32).  Mind the memory:
    ~0.07 MB per frame at grid 140, i.e. ~70 MB per second of audio at 1 ms."""
    times, frames = [], []
    for t_ms, z in iter_frames(audio, audio_fs, params, source_xy, duration_s):
        times.append(t_ms)
        frames.append(z.astype(np.float32))
    p = params or SimParams()
    shape = (0, p.grid_n, p.grid_n)
    return (np.array(times), np.stack(frames) if frames else np.zeros(shape, np.float32))
