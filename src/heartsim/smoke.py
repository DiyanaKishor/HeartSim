"""End-to-end smoke pipeline.  Two dataset layouts are supported:

  folders  (default)  <data_root>/<CLASS>/<file>.wav            e.g. Yaseen 2018 (no patient ids)
  circor              <data_root>/training_data.csv + training_data/<pid>_<LOC>.{wav,tsv,hea}
                      CirCor DigiScope: patient ids, one excerpt per recording starting at the first
                      S1, patient-grouped folds, a metadata-only baseline and patient-level scores.

For every recording it computes feature sets (non-cardiac 'trivial', classical 'spectral', and
'sim' = radial summary of the simulated field; CirCor adds 'meta'), then cross-validates a random
forest on each with group-aware folds, plus shuffled-label controls, and writes a verifiable run
folder.

This is a plumbing + baseline harness, not the final model: it deliberately uses classical
classifiers so it runs on a laptop CPU.  The CNN stage plugs in by adding another model name
to predictions.csv / metrics.json with the same schema.
"""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from . import circor
from .features import RingAccumulator, spectral_features, trivial_features
from .metrics import aggregate_patients, patient_mean_proba
from .preprocess import PreprocessParams, preprocess, quality_flags, read_wav
from .runlog import (MANIFEST_FIELDS, PRED_BASE_FIELDS, RunLog, new_run_dir, sha256_file)
from .simulator import SimParams, derived, iter_frames, reliable_band_hz
from .splits import grouped_folds

CIRCOR_MANIFEST_EXTRA = ["patient", "aux_relpath", "aux_sha256", "ex_start", "ex_end"]


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
    data_root, relpath, sha, pre_d, sim_d, cache_dir, excerpt = args
    key = _cfg_key(pre_d, sim_d)
    suffix = "" if excerpt is None else "_" + hashlib.sha256(json.dumps(list(excerpt)).encode()).hexdigest()[:8]
    cpath = Path(cache_dir) / f"{sha[:20]}_{key}{suffix}.npz"
    if cpath.exists():
        z = np.load(cpath)
        return relpath, z["trivial"], z["spectral"], z["sim"], json.loads(str(z["quality"])), True
    x, fs, _ = read_wav(str(Path(data_root) / relpath))
    q = quality_flags(x, fs)
    triv = trivial_features(x, fs)                    # whole recording: duration etc. are part of the shortcut check
    xe = x if excerpt is None else x[excerpt[0]:excerpt[1]]
    pre, sim = PreprocessParams(**pre_d), SimParams(**sim_d)
    xp = preprocess(xe, fs, pre)
    spec = spectral_features(xp, fs)
    acc = RingAccumulator(sim.grid_n)
    for _, field in iter_frames(xp, fs, sim):
        acc.update(field)
    ring = acc.finalize()
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez(cpath, trivial=triv, spectral=spec, sim=ring, quality=json.dumps(q))
    return relpath, triv, spec, ring, q, False


def _fit_predict(X, y, tr, te, seed, classes, shuffle_train_labels=False, balance=False):
    """Return (predicted labels, probability matrix with one column per entry of `classes`).

    balance=True is for imbalanced targets (CirCor murmur: about 20% Present).  The forest is grown with
    balanced_subsample class weights AND the returned probabilities are divided by the training-set class
    frequencies and renormalised, so the argmax decision treats every class equally instead of always
    predicting the majority.  Predictions are always the argmax of the returned probabilities.
    """
    ytr = y[tr].copy()
    if shuffle_train_labels:
        ytr = np.random.default_rng(seed).permutation(ytr)
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=1,
                                 class_weight="balanced_subsample" if balance else None)
    clf.fit(X[tr], ytr)
    p = clf.predict_proba(X[te])
    proba = np.zeros((len(te), len(classes)))
    for j, c in enumerate(clf.classes_):
        proba[:, classes.index(c)] = p[:, j]
    if balance:
        prior = np.array([(ytr == c).mean() for c in classes])
        proba = proba / np.maximum(prior, 1e-9)
        proba = proba / np.maximum(proba.sum(axis=1, keepdims=True), 1e-300)
    return np.array(classes, dtype=object)[np.argmax(proba, axis=1)], proba


def run_smoke(data_root: str, out_root: str = "runs", name: str = "smoke", seed: int = 0,
              n_splits: int = 5, max_per_class: int | None = None, workers: int = 1,
              groups_csv: str | None = None, sim_overrides: dict | None = None,
              pre_overrides: dict | None = None, repo_dir: str | None = None,
              dataset: str = "folders", target: str = "outcome", excerpt_s: float = 4.0,
              balance: str = "auto") -> Path:
    if dataset not in ("folders", "circor"):
        raise ValueError(f"unknown dataset layout {dataset!r}")
    if balance not in ("auto", "on", "off"):
        raise ValueError(f"balance must be auto, on or off, got {balance!r}")
    # auto: on for CirCor (imbalanced murmur task), off for the balanced folder datasets so earlier
    # Yaseen runs stay reproducible
    bal = balance == "on" or (balance == "auto" and dataset == "circor")
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

    is_circor = dataset == "circor"
    manifest, meta_X, circor_info = [], None, None
    if is_circor:
        citems, circor_info = circor.discover_circor(data_root, target, max_per_class)
        classes = list(circor_info["classes"])
        rl.log(f"circor target={target}: {circor_info['n_recordings']} recordings, "
               f"{circor_info['n_patients']} patients in {circor_info['n_groups']} groups "
               f"({circor_info['patients_merged_by_additional_id']} merged via Additional ID); "
               f"dropped {circor_info['dropped']}")
        too_short = 0
        kept = []
        for it in citems:
            wp, tp = Path(data_root) / it["relpath"], Path(data_root) / it["aux_relpath"]
            fs, x = wf.read(str(wp), mmap=True)
            rng = circor.excerpt_range(circor.read_tsv(tp), int(fs), len(x), excerpt_s)
            if rng is None:
                too_short += 1
                continue
            kept.append(it)
            manifest.append(dict(id=it["relpath"], relpath=it["relpath"], sha256=sha256_file(wp),
                                 bytes=wp.stat().st_size, label=it["label"], group=it["group"],
                                 n_samples=len(x), fs=int(fs), patient=it["patient"],
                                 aux_relpath=it["aux_relpath"], aux_sha256=sha256_file(tp),
                                 ex_start=rng[0], ex_end=rng[1]))
        if too_short:
            circor_info["dropped"]["too_short"] = too_short
            rl.log(f"{too_short} recordings shorter than 0.75 x {excerpt_s}s were dropped")
        if not manifest:
            raise SystemExit("no recording is long enough for the requested excerpt")
        meta_X, _ = circor.build_meta_matrix(kept)
        has_ids = True
        rl.log(f"{len(manifest)} recordings kept, classes={classes}, excerpt {excerpt_s}s from first S1")
    else:
        items = discover(data_root, max_per_class)
        classes = sorted({c for _, c in items})
        rl.log(f"{len(items)} files, classes={classes}")
        group_map, has_ids = {}, False
        if groups_csv:
            import csv
            with open(groups_csv, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    group_map[r["file"].replace("\\", "/")] = r["group"]
            has_ids = True
        for rel, cls in items:
            p = Path(data_root) / rel
            fs, x = wf.read(str(p))
            sid = rel.replace("\\", "/")
            manifest.append(dict(id=sid, relpath=rel, sha256=sha256_file(p), bytes=p.stat().st_size,
                                 label=cls, group=group_map.get(sid, sid), n_samples=len(x), fs=int(fs)))
        if groups_csv:
            missing = [m["id"] for m in manifest if m["id"] not in group_map]
            if missing:
                raise SystemExit(f"{len(missing)} files have no group in {groups_csv}, e.g. {missing[:3]}")

    cache_dir = Path(out_root) / "_cache"
    jobs = [(data_root, m["relpath"], m["sha256"], pre.to_dict(), sim.to_dict(), str(cache_dir),
             (m["ex_start"], m["ex_end"]) if is_circor else None) for m in manifest]
    feats = {}
    done = 0
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = ex.map(_features_for_file, jobs, chunksize=2)
            for r in results:
                feats[r[0]] = r
                done += 1
                if done % 25 == 0:
                    el = time.time() - t0
                    rl.log(f"features {done}/{len(jobs)}  elapsed {el:.0f}s  eta {el/done*(len(jobs)-done):.0f}s")
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
    if is_circor:
        Xs["meta"] = meta_X
    quality = {m["id"]: feats[m["relpath"]][4] for m in manifest}

    model_defs = {"trivial": ("trivial", False)}
    if is_circor:
        model_defs["meta"] = ("meta", False)
    model_defs.update({"spectral": ("spectral", False), "sim": ("sim", False),
                       "spectral_shuffled": ("spectral", True), "sim_shuffled": ("sim", True)})
    baselines = ["trivial"] + (["meta"] if is_circor else [])
    preds = {m: np.empty(len(y), dtype=object) for m in model_defs}
    probas = {m: np.zeros((len(y), len(classes))) for m in model_defs}
    fold_of = np.full(len(y), -1)
    for k, tr, te in grouped_folds(y, groups, n_splits=n_splits, seed=seed):
        fold_of[te] = k
        for mname, (fset, shuf) in model_defs.items():
            preds[mname][te], probas[mname][te] = _fit_predict(Xs[fset], y, tr, te, seed + k, classes, shuf, bal)
        rl.log(f"fold {k+1}/{n_splits} done")

    def auroc(yt, proba):
        """One-vs-rest AUROC per class (scikit-learn), averaged over classes present on both sides."""
        yt = np.asarray(yt)
        vals = [roc_auc_score(yt == c, proba[:, j]) for j, c in enumerate(classes)
                if 0 < (yt == c).sum() < len(yt)]
        return float(np.mean(vals)) if vals else float("nan")

    def headline(yt, yp, proba):
        return dict(accuracy=float(accuracy_score(yt, yp)),
                    balanced_accuracy=float(balanced_accuracy_score(yt, yp)),
                    macro_f1=float(f1_score(yt, yp, labels=classes, average="macro")),
                    auroc=auroc(yt, proba))

    patients = [m["patient"] for m in manifest] if is_circor else None
    rows = []
    metrics = {"classes": classes, "models": {}}
    for mname in model_defs:
        yp = preds[mname]
        metrics["models"][mname] = headline(y, yp, probas[mname])
        for i, sid in enumerate(ids):
            row = dict(model=mname, id=sid, fold=int(fold_of[i]), y_true=y[i], y_pred=yp[i])
            row.update({f"proba_{c}": float(probas[mname][i, j]) for j, c in enumerate(classes)})
            rows.append(row)
        msg = (f"{mname:18s} acc={metrics['models'][mname]['accuracy']:.4f} "
               f"bal_acc={metrics['models'][mname]['balanced_accuracy']:.4f} "
               f"auroc={metrics['models'][mname]['auroc']:.4f}")
        if is_circor:
            _, pt, py = aggregate_patients(patients, ids, list(y), probas[mname], classes)
            _, _, pmean = patient_mean_proba(patients, ids, list(y), probas[mname])
            pl = headline(pt, py, pmean)
            pl["n"] = len(pt)
            metrics["models"][mname]["patient_level"] = pl
            msg += (f"   | per patient (n={len(pt)}) bal_acc={pl['balanced_accuracy']:.4f} "
                    f"auroc={pl['auroc']:.4f}")
        rl.log(msg)

    cfg = dict(seed=seed, n_splits=n_splits, data_root_name=Path(data_root).name,
               has_patient_ids=has_ids, trivial_baseline="trivial", baselines=baselines,
               controls={"spectral": "spectral_shuffled", "sim": "sim_shuffled"},
               preprocess=pre.to_dict(), sim=sim.to_dict(),
               sim_derived=dict(d, reliable_band_hz_5pct=reliable),
               classifier=("RandomForest(300, min_samples_leaf=3, balanced_subsample + prior-corrected decision)"
                           if bal else "RandomForest(300, min_samples_leaf=3)"), balance=bal,
               warnings=warnings, workers=workers, max_per_class=max_per_class,
               cache_key=_cfg_key(pre.to_dict(), sim.to_dict()),
               dataset=dataset, seconds=round(time.time() - t0, 1))
    if is_circor:
        cfg.update(target=target, excerpt_s=excerpt_s, circor=circor_info)
    rl.write_json("config.json", cfg)
    rl.write_env()
    rl.write_csv("manifest.csv", manifest, MANIFEST_FIELDS + (CIRCOR_MANIFEST_EXTRA if is_circor else []))
    rl.write_csv("predictions.csv", rows, PRED_BASE_FIELDS + [f"proba_{c}" for c in classes])
    rl.write_json("metrics.json", metrics)
    rl.write_json("quality.json", quality)
    rl.log(f"done in {time.time()-t0:.0f}s. Next: heartsim verify {run_dir} --data-root {data_root}")
    rl.close()
    return run_dir
