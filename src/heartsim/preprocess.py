"""Audio preprocessing in front of the simulator.

Order matters and is fixed here: read -> mono float -> DC removal -> zero-phase
band-pass (anti-alias + hum/rumble removal) -> amplitude normalisation.  Resampling to
the simulator's time grid happens inside the simulator (np.interp), which is only safe
because the band-pass has already removed everything above the simulator's Nyquist.

The default band (25-400 Hz) is a *design choice*, not a sourced standard: the
simulator grid itself is only trustworthy up to `simulator.reliable_band_hz`, and
murmur energy above that is either distorted or discarded.  Change it deliberately.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt


@dataclass(frozen=True)
class PreprocessParams:
    band_lo_hz: float = 25.0
    band_hi_hz: float = 400.0
    filter_order: int = 4
    # "rms" | "p99" | "none".  Measured (synthetic beats + white noise, band-pass first):
    # neither mode is clearly better than the other, so the prototype's RMS is kept.
    # What did help was the ORDER: band-pass before normalising stops broadband noise from
    # shrinking the heart-sound amplitude (relative gain at -5 dB SNR: 0.49 -> 0.80).
    normalize: str = "rms"
    target_level: float = 0.2   # RMS target (rms mode) or p99 |x| target (p99 mode)

    def to_dict(self) -> dict:
        return asdict(self)


def read_wav(path: str):
    """Return (float64 mono signal scaled to about [-1, 1], sample rate, original dtype name)."""
    fs, x = wavfile.read(path)
    dtype = str(x.dtype)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    if np.issubdtype(np.dtype(dtype), np.integer):
        x /= float(np.iinfo(np.dtype(dtype)).max)
    return x, int(fs), dtype


def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    if not (0 < lo < hi < fs / 2):
        raise ValueError(f"band ({lo}, {hi}) Hz is not inside (0, Nyquist={fs/2}) Hz")
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def normalize(x: np.ndarray, mode: str, level: float, eps: float = 1e-12) -> np.ndarray:
    if mode == "none":
        return x
    if mode == "rms":
        return x * (level / (np.sqrt(np.mean(x ** 2)) + eps))
    if mode == "p99":
        return x * (level / (np.percentile(np.abs(x), 99.0) + eps))
    raise ValueError(f"unknown normalisation mode {mode!r}")


def preprocess(x: np.ndarray, fs: float, params: PreprocessParams | None = None) -> np.ndarray:
    p = params or PreprocessParams()
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    x = bandpass(x, fs, p.band_lo_hz, p.band_hi_hz, p.filter_order)
    return normalize(x, p.normalize, p.target_level)


def quality_flags(x_raw: np.ndarray, fs: float) -> dict:
    """Cheap, explicit quality indicators (computed on the RAW signal).

    These are *reported*, never silently used to drop data: filtering a dataset on them
    is a modelling decision that has to be made and logged by the person running it.
    """
    x = np.asarray(x_raw, dtype=np.float64)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    xc = x - x.mean()
    rms = float(np.sqrt(np.mean(xc ** 2))) if len(x) else 0.0
    clip = float(np.mean(np.abs(x) >= 0.999 * peak)) if peak > 0 else 1.0
    flat = float(np.mean(np.abs(np.diff(x)) == 0)) if len(x) > 1 else 1.0
    return dict(duration_s=len(x) / fs, rms=rms, peak=peak, clip_fraction=clip,
                flatline_fraction=flat, dc_over_rms=abs(float(x.mean())) / (rms + 1e-12))
