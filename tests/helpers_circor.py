"""Synthetic dataset in the real CirCor file formats (as observed on a real download):

  training_data.csv   header: Patient ID,Recording locations:,Age,Sex,Height,Weight,Pregnancy status,Murmur,
                      Murmur locations,...,Outcome,Campaign,Additional ID   (missing values are the text 'nan')
  <pid>_<LOC>.wav     int16, 4000 Hz
  <pid>_<LOC>.tsv     tab separated: start end state   (first row is an unannotated 0 segment)
  <pid>_<LOC>.hea     "<name> 1 4000 <n>" / "<name>.wav 16+44 1 16 0 0 0 0 <LOC>"

It checks plumbing only; the audio is toy beats, not heart sounds.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from helpers import synth_beats

HEADER = ("Patient ID,Recording locations:,Age,Sex,Height,Weight,Pregnancy status,Murmur,Murmur locations,"
          "Most audible location,Systolic murmur timing,Systolic murmur shape,Systolic murmur grading,"
          "Systolic murmur pitch,Systolic murmur quality,Diastolic murmur timing,Diastolic murmur shape,"
          "Diastolic murmur grading,Diastolic murmur pitch,Diastolic murmur quality,Outcome,Campaign,"
          "Additional ID")


def beat_tsv(seconds, period, gap, first=0.15):
    """tsv rows (start, end, state) for the toy beats: 1=S1, 2=systole, 3=S2, 4=diastole."""
    rows, t = [(0.0, first, 0)], first
    while t + period <= seconds:
        rows += [(t, t + 0.10, 1), (t + 0.10, t + gap, 2), (t + gap, t + gap + 0.10, 3),
                 (t + gap + 0.10, t + period, 4)]
        t += period
    return rows


def write_recording(td: Path, pid: str, loc: str, kind: int, seconds: float, seed: int, tsv=True):
    x, fs = synth_beats(fs=4000, seconds=seconds, kind=kind, seed=seed)
    x = x / (np.abs(x).max() + 1e-9) * 0.5
    name = f"{pid}_{loc}"
    wavfile.write(str(td / f"{name}.wav"), fs, (x * 32767).astype(np.int16))
    (td / f"{name}.hea").write_text(f"{name} 1 {fs} {len(x)}\n{name}.wav 16+44 1 16 0 0 0 0 {loc}\n")
    if tsv:
        period, gap = [(0.8, 0.30), (0.8, 0.38), (0.7, 0.30)][kind % 3]
        rows = beat_tsv(seconds, period, gap)
        (td / f"{name}.tsv").write_text("\n".join(f"{a}\t{b}\t{c}" for a, b, c in rows) + "\n")


def write_synth_circor(root: Path, n_patients=24, seed=0):
    """Returns a dict describing the quirks that were planted, so tests can assert on them."""
    root = Path(root)
    td = root / "training_data"
    td.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows, facts = [], dict(linked=[], unknown_murmur=None, no_tsv=None, short=None)
    locs = ["AV", "MV", "PV", "TV"]
    for i in range(n_patients):
        pid = str(1000 + i)
        abnormal = i % 2 == 0
        kind = 2 if abnormal else 0
        murmur = "Present" if abnormal else "Absent"
        add = "nan"
        if i == 5:                      # patient 1005 is also known as 1004 (chain: 1006 -> 1005 -> 1004)
            add = "1004"
            facts["linked"] += ["1004", "1005"]
        if i == 6:
            add = "1005"
            facts["linked"].append("1006")
        if i == 9:
            murmur = "Unknown"
            facts["unknown_murmur"] = pid
        rows.append(f"{pid},AV+MV+PV+TV,Child,{'Female' if i % 3 else 'Male'},{90 + i}.0,{14 + i % 5}.5,False,"
                    f"{murmur},nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,"
                    f"{'Abnormal' if abnormal else 'Normal'},{'CC2014' if i % 4 else 'CC2015'},{add}")
        for j, loc in enumerate(locs[: 2 + (i % 3)]):
            secs = 8.0
            tsv = True
            if i == 3 and j == 0:
                secs = 2.0                                  # too short for a 4 s excerpt
                facts["short"] = f"{pid}_{loc}"
            if i == 7 and j == 1:
                tsv = False
                facts["no_tsv"] = f"{pid}_{loc}"
            write_recording(td, pid, loc, kind, secs, seed=seed * 7919 + i * 31 + j, tsv=tsv)
    # a second recording at the same location, named with a numeric suffix as in the real data
    write_recording(td, "1000", "AV_1", 2, 8.0, seed=99)
    (root / "training_data.csv").write_text(HEADER + "\n" + "\n".join(rows) + "\n")
    facts["n_patients"] = n_patients
    return facts
