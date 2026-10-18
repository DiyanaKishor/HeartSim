"""End-to-end smoke pipeline on a folder-per-class WAV dataset (the Yaseen layout):

    <data_root>/<CLASS>/<file>.wav

For every file it computes three feature sets (non-cardiac 'trivial', classical 'spectral',
and 'sim' = radial summary of the simulated field), then cross-validates a random forest on
each with group-aware folds, plus shuffled-label controls, and writes a verifiable run folder.

This is a plumbing + baseline harness, not the final model: it deliberately uses classical
classifiers so it runs on a laptop CPU.  The CNN stage plugs in by adding another model name
to predictions.csv / metrics.json with the same schema.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from .features import RingAccumulator, spectral_features, trivial_features
from .preprocess import PreprocessParams, preprocess, quality_flags, read_wav
from .runlog import (MANIFEST_FIELDS, PRED_BASE_FIELDS, RunLog, new_run_dir, sha256_file)
from .simulator import SimParams, derived, iter_frames, reliable_band_hz
from .splits import grouped_folds


def discover(data_root: str, max_per_class: int | None = None):
    root = Path(data_root)
    items = []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        wavs = sorted(cls_dir.glob("*.wav"))
        if max_per_class:
            wavs = wavs[:max_per_class]
        for w in wavs:
            items.append((str(w.relative_to(root)).replace("\\", "/"), cls_dir.name))
    if not items:
        raise SystemExit(f"no <class>/<file>.wav found under {data_root}")
    return items


def _cfg_key(pre: dict, sim: dict) -> str:
    return hashlib.sha256(json.dumps([pre, sim], sort_keys=True).encode()).hexdigest()[:12]


def _features_for_file(args):
    data_root, relpath, sha, pre_d, sim_d, cache_dir = args
    key = _cfg_key(pre_d, sim_d)
    cpath = Path(cache_dir) / f"{sha[:20]}_{key}.npz"
    if cpath.exists():
        z = np.load(cpath)
        return relpath, z["trivial"], z["spectral"], z["sim"], json.loads(str(z["quality"])), True
    x, fs, _ = read_wav(str(Path(data_root) / relpath))
    q = quality_flags(x, fs)
    triv = trivial_features(x, fs)
    pre, sim = PreprocessParams(**pre_d), SimParams(**sim_d)
    xp = preprocess(x, fs, pre)
    spec = spectral_features(xp, fs)
    acc = RingAccumulator(sim.grid_n)
    for _, field in iter_frames(xp, fs, sim):
        acc.update(field)
    ring = acc.finalize()
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez(cpath, trivial=triv, spectral=spec, sim=ring, quality=json.dumps(q))
    return relpath, triv, spec, ring, q, False


def _fit_predict(X, y, tr, te, seed, shuffle_train_labels=False):
    ytr = y[tr].copy()
    if shuffle_train_labels:
        ytr = np.random.default_rng(seed).permutation(ytr)
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=1)
    clf.fit(X[tr], ytr)
    return clf.predict(X[te])


def run_smoke(data_root: str, out_root: str = "runs", name: str = "smoke", seed: int = 0,
              n_splits: int = 5, max_per_class: int | None = None, workers: int = 1,
              groups_csv: str | None = None, sim_overrides: dict | None = None,
              pre_overrides: dict | None = None, repo_dir: str | None = None) -> Path:
    t0 = time.time()
    run_dir = new_run_dir(out_root, name)
    rl = RunLog(run_dir, repo_dir=repo_dir)
    sim = SimParams(**(sim_overrides or {}))
    pre = PreprocessParams(**(pre_overrides or {}))
    d = derived(sim)
    reliable = reliable_band_hz(sim, 0.05)
    if pre.band_hi_hz > 0.9 * d["nyquist_hz"]:
        raise ValueError(f"band_hi_hz={pre.band_hi_hz} is too close to the simulator Nyquist "
                         f"({d['nyquist_hz']:.0f} Hz); resampling onto the sim grid would alias")
    warnings = []
    if pre.band_hi_hz > reliable:
        warnings.append(f"band upper edge {pre.band_hi_hz:.0f} Hz is above the grid's reliable limit "
                        f"{reliable:.0f} Hz (5% phase-speed error): content above it is dispersed by the scheme")
    rl.log(f"run folder: {run_dir}")
    rl.log(f"sim: {sim.to_dict()}  dt={d['dt']*1e3:.4f} ms  steps/frame={d['steps_per_frame']}  "
           f"nyquist={d['nyquist_hz']:.0f} Hz  reliable<={reliable:.0f} Hz")
    for w in warnings:
        rl.log("WARNING: " + w)

    items = discover(data_root, max_per_class)
    classes = sorted({c for _, c in items})
    rl.log(f"{len(items)} files, classes={classes}")

    group_map = {}
    has_ids = False
    if groups_csv:
        import csv
        with open(groups_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                group_map[r["file"].replace("\\", "/")] = r["group"]
        has_ids = True

    manifest, fs_of = [], {}
    for rel, cls in items:
        p = Path(data_root) / rel
        import scipy.io.wavfile as wf
        fs, x = wf.read(str(p))
        sid = rel.replace("\\", "/")
        manifest.append(dict(id=sid, relpath=rel, sha256=sha256_file(p), bytes=p.stat().st_size,
                             label=cls, group=group_map.get(sid, sid), n_samples=len(x), fs=int(fs)))
    if groups_csv:
        missing = [m["id"] for m in manifest if m["id"] not in group_map]
        if missing:
            raise SystemExit(f"{len(missing)} files have no group in {groups_csv}, e.g. {missing[:3]}")

    cache_dir = Path(out_root) / "_cache"
    jobs = [(data_root, m["relpath"], m["sha256"], pre.to_dict(), sim.to_dict(), str(cache_dir))
            for m in manifest]
    feats = {}
    done = 0
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = ex.map(_features_for_file, jobs, chunksize=2)
            for r in results:
                feats[r[0]] = r
                done += 1
                if done % 25 == 0:
                    rl.log(f"features {done}/{len(jobs)}  elapsed {time.time()-t0:.0f}s")
    else:
        for j in jobs:
            r = _features_for_file(j)
            feats[r[0]] = r
            done += 1
            if done % 25 == 0 or done == len(jobs):
                el = time.time() - t0
                rl.log(f"features {done}/{len(jobs)}  elapsed {el:.0f}s  eta {el/done*(len(jobs)-done):.0f}s")

    ids = [m["id"] for m in manifest]
    y = np.array([m["label"] for m in manifest])
    groups = np.array([m["group"] for m in manifest])
    Xs = {
        "trivial": np.stack([feats[m["relpath"]][1] for m in manifest]),
        "spectral": np.stack([feats[m["relpath"]][2] for m in manifest]),
        "sim": np.stack([feats[m["relpath"]][3] for m in manifest]),
    }
    quality = {m["id"]: feats[m["relpath"]][4] for m in manifest}

    model_defs = {"trivial": ("trivial", False), "spectral": ("spectral", False),
                  "sim": ("sim", False),
                  "spectral_shuffled": ("spectral", True), "sim_shuffled": ("sim", True)}
    preds = {m: np.empty(len(y), dtype=object) for m in model_defs}
    fold_of = np.full(len(y), -1)
    for k, tr, te in grouped_folds(y, groups, n_splits=n_splits, seed=seed):
        fold_of[te] = k
        for mname, (fset, shuf) in model_defs.items():
            preds[mname][te] = _fit_predict(Xs[fset], y, tr, te, seed + k, shuf)
        rl.log(f"fold {k+1}/{n_splits} done")

    rows = []
    metrics = {"classes": classes, "models": {}}
    for mname in model_defs:
        yp = preds[mname]
        metrics["models"][mname] = dict(
            accuracy=float(accuracy_score(y, yp)),
            balanced_accuracy=float(balanced_accuracy_score(y, yp)),
            macro_f1=float(f1_score(y, yp, labels=classes, average="macro")))
        for i, sid in enumerate(ids):
            rows.append(dict(model=mname, id=sid, fold=int(fold_of[i]), y_true=y[i], y_pred=yp[i]))
        rl.log(f"{mname:18s} acc={metrics['models'][mname]['accuracy']:.4f} "
               f"bal_acc={metrics['models'][mname]['balanced_accuracy']:.4f}")

    cfg = dict(seed=seed, n_splits=n_splits, data_root_name=Path(data_root).name,
               has_patient_ids=has_ids, trivial_baseline="trivial",
               controls={"spectral": "spectral_shuffled", "sim": "sim_shuffled"},
               preprocess=pre.to_dict(), sim=sim.to_dict(),
               sim_derived=dict(d, reliable_band_hz_5pct=reliable),
               classifier="RandomForest(300, min_samples_leaf=3)",
               warnings=warnings, workers=workers, max_per_class=max_per_class,
               cache_key=_cfg_key(pre.to_dict(), sim.to_dict()),
               seconds=round(time.time() - t0, 1))
    rl.write_json("config.json", cfg)
    rl.write_env()
    rl.write_csv("manifest.csv", manifest, MANIFEST_FIELDS)
    rl.write_csv("predictions.csv", rows, PRED_BASE_FIELDS)
    rl.write_json("metrics.json", metrics)
    rl.write_json("quality.json", quality)
    rl.log(f"done in {time.time()-t0:.0f}s. Next: heartsim verify {run_dir} --data-root {data_root}")
    rl.close()
    return run_dir
