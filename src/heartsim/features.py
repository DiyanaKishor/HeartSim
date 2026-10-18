"""Feature extractors used for the baselines and for the simulated-pattern features.

* trivial_features   - NON-cardiac recording properties (loudness, duration, silence, DC...).
                       They exist to measure how much of a dataset's label is a shortcut.
                       On the Yaseen set a random forest on these alone reached 97% (5-fold),
                       so every reported result must be shown next to this baseline.
* spectral_features  - classical log-band-energy summary of the preprocessed signal.
* RingAccumulator    - streaming radial-energy summary of the simulated field (no frames stored).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert, welch, spectrogram

TRIVIAL_NAMES = ["dur", "rms", "peak", "crest", "lead", "trail", "dc", "clipfrac",
                 "f50", "f95", "centroid", "frac_gt300", "frac_gt500"]


def trivial_features(x_raw: np.ndarray, fs: float) -> np.ndarray:
    x = np.asarray(x_raw, dtype=np.float64)
    n = len(x)
    dc = x.mean()
    xc = x - dc
    rms = np.sqrt(np.mean(xc ** 2))
    peak = np.max(np.abs(xc))
    f, p = welch(xc, fs=fs, nperseg=min(1024, n))
    tot = p.sum() + 1e-30
    cum = np.cumsum(p) / tot
    f95 = f[min(np.searchsorted(cum, 0.95), len(f) - 1)]
    f50 = f[min(np.searchsorted(cum, 0.50), len(f) - 1)]
    centroid = float((f * p).sum() / tot)
    env = np.abs(hilbert(xc))
    active = np.where(env > 0.05 * env.max())[0]
    lead = active[0] / n if len(active) else 1.0
    trail = 1.0 - (active[-1] + 1) / n if len(active) else 1.0
    clip = float(np.mean(np.abs(x) > 0.99 * np.max(np.abs(x))))
    return np.array([n / fs, rms, peak, peak / (rms + 1e-12), lead, trail, dc, clip, f50, f95,
                     centroid, p[f > 300].sum() / tot, p[f > 500].sum() / tot])


def spectral_features(x_pre: np.ndarray, fs: float, n_bands: int = 20,
                      f_lo: float = 20.0, f_hi: float = 500.0) -> np.ndarray:
    """Mean and std over time of log band energies (log-spaced bands). Input = preprocessed signal."""
    nper = int(min(256, max(32, len(x_pre) // 8)))
    f, _, s = spectrogram(x_pre, fs=fs, nperseg=nper, noverlap=nper * 3 // 4)
    edges = np.geomspace(f_lo, min(f_hi, fs / 2 - 1), n_bands + 1)
    bands = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (f >= a) & (f < b)
        bands.append(s[m].sum(axis=0) if m.any() else np.zeros(s.shape[1]))
    lb = np.log(np.asarray(bands) + 1e-12)
    return np.concatenate([lb.mean(axis=1), lb.std(axis=1)])


class RingAccumulator:
    """Streaming summary of a simulated field: energy in concentric rings around the source.

    Output (3 * n_rings values): per ring, mean / std / max over time of log ring-RMS.
    Intentionally simple: it is a *baseline* representation of the simulated pattern, not
    the final model input.  It only sees radial structure, which (measured on one real
    recording) carried a median 88% of the field energy.
    """

    def __init__(self, grid_n: int, n_rings: int = 10, r_max_cells: float | None = None,
                 source_xy=(0.5, 0.5)):
        cj = source_xy[0] * (grid_n - 1)
        ci = source_xy[1] * (grid_n - 1)
        yy, xx = np.mgrid[0:grid_n, 0:grid_n]
        rr = np.hypot(yy - ci, xx - cj)
        r_max = r_max_cells or 0.35 * grid_n
        self.n_rings = n_rings
        self.bin = np.minimum((rr / r_max * n_rings).astype(int), n_rings)   # n_rings = "outside"
        self.counts = np.maximum(np.bincount(self.bin.ravel(), minlength=n_rings + 1), 1)
        self._log = []

    def update(self, field: np.ndarray) -> None:
        e = np.bincount(self.bin.ravel(), weights=(field ** 2).ravel(),
                        minlength=self.n_rings + 1) / self.counts
        self._log.append(0.5 * np.log(e[: self.n_rings] + 1e-30))

    def finalize(self) -> np.ndarray:
        a = np.asarray(self._log)
        return np.concatenate([a.mean(axis=0), a.std(axis=0), a.max(axis=0)])
