"""Loader for the CirCor DigiScope phonocardiogram training set (PhysioNet, v1.0.3).

Expected layout under <data_root> (this is exactly what the PhysioNet zip contains):

    training_data.csv              one row per patient (Patient ID, Murmur, Outcome, Age, ..., Additional ID)
    training_data/<pid>_<LOC>.wav  one file per auscultation location (AV, MV, PV, TV, ...), 4 kHz, 16 bit
    training_data/<pid>_<LOC>.tsv  S1/S2 segmentation: start_s  end_s  state
                                   (0 = unannotated, 1 = S1, 2 = systole, 3 = S2, 4 = diastole)
    training_data/<pid>_<LOC>.hea  WFDB header, first line "<name> 1 <fs> <n_samples>"

Facts this module relies on, all observed in the real files:
  * Murmur and Outcome are PATIENT-level labels; every recording of a patient shares them.
  * Several recordings per patient, so folds MUST be grouped by patient.
  * Some patients appear under two ids; the CSV links them with "Additional ID".  Both ids are
    merged into one group so the same child can never be in train and test.
  * Recordings last 5-80 s.  The simulator is the expensive step, so each recording is reduced to
    one fixed-length excerpt that starts at the first annotated S1.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REC_NAME = re.compile(r"^(?P<pid>\d+)_(?P<loc>[A-Za-z]+)(?:_(?P<n>\d+))?$")
TARGETS = {"outcome": ("Outcome", ("Abnormal", "Normal")),
           "murmur": ("Murmur", ("Absent", "Present"))}       # "Unknown" murmur rows are dropped
S1_STATE = 1


# --------------------------------------------------------------------------- small file parsers
def parse_hea(path) -> tuple[int, int]:
    """(sampling rate, number of samples) from the first line of a WFDB header."""
    with open(path, encoding="utf-8") as f:
        parts = f.readline().split()
    return int(float(parts[2])), int(parts[3])


def read_tsv(path) -> np.ndarray:
    """(n, 3) float array: start_s, end_s, state.  Empty file -> shape (0, 3)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            if len(p) != 3:
                raise ValueError(f"{path}: expected 3 columns, got {len(p)}: {line!r}")
            rows.append([float(v) for v in p])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 3)


def excerpt_range(tsv: np.ndarray, fs: int, n_samples: int, seconds: float, min_frac: float = 0.75):
    """(start, stop) sample indices of one excerpt, or None if the recording is too short.

    The excerpt starts at the first annotated S1 (sample 0 if there is none).  If that would run
    past the end it is moved back so the excerpt ends with the recording.  Recordings shorter than
    min_frac * seconds are rejected rather than padded.
    """
    length = int(round(seconds * fs))
    if n_samples < min_frac * length:
        return None
    s1 = tsv[tsv[:, 2] == S1_STATE][:, 0] if len(tsv) else np.zeros(0)
    start = int(round(float(s1[0]) * fs)) if len(s1) else 0
    start = min(max(start, 0), n_samples)
    if start + length > n_samples:
        start = max(0, n_samples - length)
    return start, min(n_samples, start + length)


# --------------------------------------------------------------------------- metadata
def _norm_id(v) -> str:
    s = str(v).strip()
    return s[:-2] if s.endswith(".0") else s


def load_metadata(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df.columns = [c.strip().rstrip(":") for c in df.columns]
    df = df.apply(lambda col: col.str.strip()).replace({"nan": ""})
    need = ["Patient ID", "Murmur", "Outcome"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"{csv_path}: missing columns {missing}; found {list(df.columns)}")
    df["Patient ID"] = df["Patient ID"].map(_norm_id)
    if df["Patient ID"].duplicated().any():
        raise SystemExit(f"{csv_path}: duplicated Patient ID values")
    return df


def patient_groups(meta: pd.DataFrame) -> tuple[dict[str, str], int]:
    """patient id -> group id.  Ids linked through 'Additional ID' share one group (transitively).
    Returns (mapping, number of patients that were merged into a group with someone else)."""
    parent: dict[str, str] = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    extra = meta["Additional ID"] if "Additional ID" in meta.columns else [""] * len(meta)
    for pid, add in zip(meta["Patient ID"], extra):
        ra = find(pid)
        if add:
            rb = find(_norm_id(add))
            if ra != rb:
                parent[rb] = ra
    members: dict[str, list[str]] = {}
    for pid in list(parent):
        members.setdefault(find(pid), []).append(pid)
    label = {r: "G" + min(m, key=lambda s: (len(s), s)) for r, m in members.items()}
    mapping = {pid: label[find(pid)] for pid in meta["Patient ID"]}
    sizes = Counter(mapping.values())
    return mapping, sum(1 for g in mapping.values() if sizes[g] > 1)


# --------------------------------------------------------------------------- discovery
def discover_circor(data_root, target: str = "outcome", max_per_class: int | None = None,
                    locations: set[str] | None = None):
    """Return (items, info).  Each item: relpath, aux_relpath (the .tsv), label, patient, group,
    location, meta (the patient's CSV row as a dict).  info counts everything that was dropped."""
    if target not in TARGETS:
        raise SystemExit(f"unknown target {target!r}; choose from {sorted(TARGETS)}")
    root = Path(data_root)
    csv_path, td = root / "training_data.csv", root / "training_data"
    if not csv_path.exists() or not td.is_dir():
        raise SystemExit(f"expected {csv_path} and the folder {td}; see docs/WORKFLOW.md")
    col, valid = TARGETS[target]
    meta = load_metadata(csv_path)
    groups, n_merged = patient_groups(meta)
    rows = {r["Patient ID"]: r for r in meta.to_dict("records")}

    dropped = Counter()
    items = []
    for wav in sorted(td.glob("*.wav")):
        m = REC_NAME.match(wav.stem)
        if not m:
            dropped["unparseable_name"] += 1
            continue
        pid = m["pid"]
        row = rows.get(pid)
        if row is None:
            dropped["no_metadata_row"] += 1
            continue
        label = row[col]
        if label not in valid:
            dropped[f"label_{label or 'missing'}"] += 1
            continue
        loc = m["loc"].upper()
        if locations and loc not in locations:
            dropped["location_filtered"] += 1
            continue
        tsv = wav.with_suffix(".tsv")
        if not tsv.exists():
            dropped["no_tsv"] += 1
            continue
        items.append(dict(relpath=f"training_data/{wav.name}", aux_relpath=f"training_data/{tsv.name}",
                          label=label, patient=pid, group=groups[pid], location=loc, meta=row))
    if max_per_class:
        kept, seen = [], Counter()
        for it in items:
            if seen[it["label"]] < max_per_class:
                kept.append(it)
                seen[it["label"]] += 1
        items = kept
    if not items:
        raise SystemExit(f"no usable recordings under {root} (dropped: {dict(dropped)})")
    info = dict(target=target, classes=list(valid), n_recordings=len(items),
                n_patients=len({i["patient"] for i in items}),
                n_groups=len({i["group"] for i in items}),
                patients_merged_by_additional_id=n_merged, dropped=dict(dropped))
    return items, info


# --------------------------------------------------------------------------- metadata-only baseline
def build_meta_matrix(items) -> tuple[np.ndarray, list[str]]:
    """Features that need NO audio: age category, sex, height, weight, pregnancy, campaign and
    recording location.  A model on these is the 'meta' baseline: an audio model has to beat it
    before its result says anything about heart sounds.  Missing numbers become -1."""
    cat_cols = ["Age", "Sex", "Campaign"]
    vocab = {c: sorted({it["meta"].get(c, "") for it in items}) for c in cat_cols}
    locs = sorted({it["location"] for it in items})
    names = [f"{c}={v}" for c in cat_cols for v in vocab[c]] + [f"loc={v}" for v in locs] \
        + ["height", "weight", "pregnant"]
    X = np.zeros((len(items), len(names)))
    for i, it in enumerate(items):
        j = 0
        for c in cat_cols:
            for v in vocab[c]:
                X[i, j] = float(it["meta"].get(c, "") == v)
                j += 1
        for v in locs:
            X[i, j] = float(it["location"] == v)
            j += 1
        for c in ("Height", "Weight"):
            try:
                X[i, j] = float(it["meta"].get(c, ""))
            except ValueError:
                X[i, j] = -1.0
            j += 1
        X[i, j] = float(it["meta"].get("Pregnancy status", "").lower() == "true")
    return X, names
