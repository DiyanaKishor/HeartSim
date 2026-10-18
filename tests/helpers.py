"""Shared test helpers: synthetic heart-like audio, synthetic datasets, synthetic run folders."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from heartsim.metrics import summarize
from heartsim.runlog import MANIFEST_FIELDS, PRED_BASE_FIELDS, RunLog, sha256_file


def damped(t, t0, f, w, a):
    return a * np.exp(-((t - t0) / w) ** 2) * np.sin(2 * np.pi * f * (t - t0))


def synth_beats(fs=4000, seconds=2.0, kind=0, seed=0):
    """Toy heart-like signal. `kind` changes S1/S2 spacing and adds a murmur-like band for kind 2."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / fs)
    x = np.zeros_like(t)
    period = [0.8, 0.8, 0.7][kind % 3]
    gap = [0.30, 0.38, 0.30][kind % 3]
    for t0 in np.arange(0.15, seconds - 0.3, period):
        x += damped(t, t0, 60, 0.02, 1.0) + damped(t, t0 + gap, 90, 0.02, 0.7)
        if kind % 3 == 2:
            x += damped(t, t0 + 0.15, 140, 0.05, 0.3)
    return x + 0.02 * rng.standard_normal(len(t)), fs


def write_synth_dataset(root: Path, n_per_class=12, n_classes=3, seed=0):
    names = ["N", "AS", "MR"][:n_classes]
    for ci, name in enumerate(names):
        d = Path(root) / name
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            x, fs = synth_beats(kind=ci, seed=seed * 1000 + ci * 100 + i)
            x = x / (np.abs(x).max() + 1e-9) * (0.5 + 0.1 * (i % 3))
            wavfile.write(str(d / f"{name}_{i:03d}.wav"), fs, (x * 32767).astype(np.int16))
    return Path(root)


def make_run(run_dir: Path, n=240, k=4, group_size=2, seed=0, with_controls=True,
             patient_ids=True, data_dir: Path | None = None):
    """Build a verifiable run folder from random data with an honest, modest classifier signal."""
    rng = np.random.default_rng(seed)
    classes = [f"C{i}" for i in range(k)]
    y = np.array([classes[i % k] for i in range(n)])
    gid = np.arange(n) // group_size
    groups = np.array([f"P{g}" for g in gid])
    folds = (gid % 4)
    man = []
    for i in range(n):
        rel = f"{y[i]}/f{i:04d}.bin"
        sha = "0" * 63 + str(i % 10)
        if data_dir is not None:
            p = Path(data_dir) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(f"file-{i}".encode())
            sha = sha256_file(p)
        man.append(dict(id=f"id{i:04d}", relpath=rel, sha256=sha, bytes=10, label=y[i],
                        group=groups[i], n_samples=100, fs=1000))
    ids = [m["id"] for m in man]

    def noisy(acc):
        out = y.copy()
        flip = rng.random(n) > acc
        out[flip] = rng.choice(classes, size=int(flip.sum()))
        return out
    model_preds = {"real": noisy(0.8), "trivial": noisy(0.4)}
    if with_controls:
        model_preds["real_shuffled"] = rng.choice(classes, size=n)
    rows, metrics = [], {"classes": classes, "models": {}}
    for m, yp in model_preds.items():
        s = summarize(y.tolist(), yp.tolist(), classes)
        metrics["models"][m] = {kk: s[kk] for kk in ("accuracy", "balanced_accuracy", "macro_f1")}
        for i in range(n):
            rows.append(dict(model=m, id=ids[i], fold=int(folds[i]), y_true=y[i], y_pred=yp[i]))
    rl = RunLog(run_dir)
    cfg = dict(seed=seed, has_patient_ids=patient_ids, trivial_baseline="trivial",
               controls={"real": "real_shuffled"} if with_controls else {})
    rl.write_json("config.json", cfg)
    rl.write_json("env.json", dict(git_commit="a" * 40, python="3"))
    rl.write_csv("manifest.csv", man, MANIFEST_FIELDS)
    rl.write_csv("predictions.csv", rows, PRED_BASE_FIELDS)
    rl.write_json("metrics.json", metrics)
    rl.close()
    return Path(run_dir)


def status(report, name_part):
    return [c["status"] for c in report.checks if name_part in c["name"]]
