"""Regression tests for the simulator.  Each one encodes a defect found in the audit of the
prototype (frame timing, boundary reflection, dispersion) or a property the pipeline relies on."""
import math

import numpy as np
import pytest

from heartsim.simulator import (SimParams, damping_field, derived, iter_frames,
                                numerical_dispersion_error, reliable_band_hz, simulate_frames)

FS = 4000


def burst(fc, T, t0=0.12):
    t = np.arange(0, T, 1 / FS)
    return np.sin(2 * np.pi * fc * (t - t0)) * np.exp(-((t - t0) / (1.5 / fc)) ** 2)


@pytest.mark.parametrize("grid_n", [48, 80, 100, 140])
def test_frames_are_on_an_exact_1ms_grid(grid_n):
    # prototype bug: 0.85 ms at grid 140, 1.49 ms at grid 80
    p = SimParams(grid_n=grid_n)
    times, frames = simulate_frames(burst(60, 0.3), FS, p)
    assert frames.shape[1:] == (grid_n, grid_n)
    assert np.allclose(np.diff(times), 1.0, atol=1e-9)
    assert math.isclose(times[0], 1.0, abs_tol=1e-9)


@pytest.mark.parametrize("grid_n,c", [(48, 3.0), (140, 3.0), (140, 6.0), (220, 3.0)])
def test_time_step_is_stable(grid_n, c):
    d = derived(SimParams(grid_n=grid_n, c=c))
    assert d["courant"] <= 1 / math.sqrt(2) + 1e-12      # 2D explicit stability limit


def test_simulation_is_linear():
    p = SimParams(grid_n=60)
    a, b = burst(60, 0.4), 0.6 * burst(140, 0.4)
    _, ra = simulate_frames(a, FS, p)
    _, rb = simulate_frames(b, FS, p)
    _, rab = simulate_frames(a + b, FS, p)
    _, r2a = simulate_frames(2 * a, FS, p)
    ra, rb, rab, r2a = (x.astype(np.float64) for x in (ra, rb, rab, r2a))
    assert np.linalg.norm(rab - ra - rb) / np.linalg.norm(rab) < 1e-5     # float32 storage
    assert np.linalg.norm(r2a - 2 * ra) / np.linalg.norm(r2a) < 1e-5


def _reference_frames(audio, p, n_frames_ms):
    """Deliberately naive implementation of the same equation (padded Laplacian, full-grid
    normalised Gaussian source) used to guard the optimised interior-only update."""
    d = derived(p)
    n, dt = p.grid_n, d["dt"]
    steps = int(n_frames_ms / 1000 / dt)
    t_sim = np.arange(steps) * dt
    a = np.interp(t_sim, np.arange(len(audio)) / FS, audio, left=0, right=0)
    c0 = int(round(0.5 * (n - 1)))
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    ker = np.exp(-((ii - c0) ** 2 + (jj - c0) ** 2) / (2 * p.source_sigma_cells ** 2))
    ker /= ker.sum()
    damp = damping_field(p)
    z = np.zeros((n, n)); zp = np.zeros((n, n))
    r = (p.c * dt / d["dx"]) ** 2
    for s in range(steps):
        pad = np.pad(z, 1)
        lap = pad[2:, 1:-1] + pad[:-2, 1:-1] + pad[1:-1, 2:] + pad[1:-1, :-2] - 4 * z
        zn = 2 * z - zp + r * lap - damp * dt * (z - zp) + dt ** 2 * a[s] * p.forcing_gain * ker
        zn[0, :] = zn[-1, :] = 0; zn[:, 0] = zn[:, -1] = 0
        zp, z = z, zn
    return z


def test_matches_naive_reference_implementation():
    p = SimParams(grid_n=64, sponge_width=12)
    audio = burst(80, 0.3)
    ms = 200
    last = None
    for t_ms, z in iter_frames(audio, FS, p, duration_s=ms / 1000):
        last = z
    ref = _reference_frames(audio, p, ms)
    # source kernel is truncated at 5 sigma (tail mass ~1e-5)
    assert np.linalg.norm(last - ref) / np.linalg.norm(ref) < 1e-4


def test_field_is_zero_before_any_audio_and_finite_after():
    p = SimParams(grid_n=48)
    _, fr = simulate_frames(np.zeros(2000), FS, p)
    assert np.all(fr == 0)
    _, fr = simulate_frames(burst(100, 0.5), FS, p)
    assert np.all(np.isfinite(fr))


def test_numerical_dispersion_matches_values_recorded_in_the_audit():
    # old prototype time step (0.8478 ms) -> -0.33 / -2.98 / -8.29 % at 20 / 60 / 100 Hz
    dt_cfl_ms = 0.5 / (139 * 3.0 * math.sqrt(2)) * 1000
    p = SimParams(frame_interval_ms=dt_cfl_ms * 0.999999)
    assert derived(p)["steps_per_frame"] == 1
    for f, expected in [(20, -0.33), (60, -2.98), (100, -8.29)]:
        assert abs(100 * numerical_dispersion_error(f, p) - expected) < 0.06


def test_reliable_band_is_consistent_and_low():
    p = SimParams()
    for tol in (0.02, 0.05, 0.10):
        f = reliable_band_hz(p, tol)
        assert abs(abs(numerical_dispersion_error(f, p)) - tol) < 5e-3
    # the defining limitation of the default grid: far below the 300-700 Hz murmur range
    assert reliable_band_hz(p, 0.05) < 150


@pytest.mark.slow
@pytest.mark.parametrize("fc", [25, 60])
def test_absorbing_boundary_matches_open_domain(fc):
    """Prototype defaults left 38% / 28% error at 25 / 60 Hz.  The shipped sponge must stay
    within 3% of a domain 4x larger (same cells/s, so the same dt and wave physics)."""
    N0, c0, Nb, T, half = 140, 3.0, 560, 0.55, 35
    a = burst(fc, T)

    def win(fr, n):
        c = (n - 1) // 2
        return fr[:, c - half:c + half, c - half:c + half]
    _, small = simulate_frames(a, FS, SimParams(grid_n=N0, c=c0), duration_s=T)
    _, big = simulate_frames(a, FS, SimParams(grid_n=Nb, c=c0 * (N0 - 1) / (Nb - 1)), duration_s=T)
    m = min(len(small), len(big))
    s, b = win(small[:m], N0), win(big[:m], Nb)
    err = np.linalg.norm(s - b, axis=(1, 2)).max() / np.linalg.norm(b, axis=(1, 2)).max()
    assert err < 0.03, f"boundary reflection error {err:.3%} at {fc} Hz"
