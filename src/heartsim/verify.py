"""Independent verification of a run folder.

Everything here is recomputed from the files in the folder (predictions.csv, manifest.csv)
with code that does not share an implementation with the training script's metric calls.
A PASS means "the logged numbers are internally consistent and the guards held", NOT
"the result is scientifically valid" and NOT "the training really happened as logged".
Re-running one fold yourself occasionally is the only way to close that last gap.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import HEADLINE_KEYS, summarize
from .runlog import sha256_file

REQUIRED_FILES = ["config.json", "env.json", "manifest.csv", "predictions.csv", "metrics.json"]
REQUIRED_PRED_COLS = ["model", "id", "fold", "y_true", "y_pred"]
REQUIRED_MANIFEST_COLS = ["id", "relpath", "sha256", "label", "group"]


class Report:
    def __init__(self):
        self.checks: list[dict] = []

    def add(self, name: str, status: str, detail: str = "") -> None:
        assert status in ("PASS", "FAIL", "WARN")
        self.checks.append(dict(name=name, status=status, detail=detail))

    @property
    def ok(self) -> bool:
        return not any(c["status"] == "FAIL" for c in self.checks)

    def text(self) -> str:
        lines = []
        for c in self.checks:
            lines.append(f"[{c['status']}] {c['name']}" + (f" - {c['detail']}" if c["detail"] else ""))
        n_f = sum(c["status"] == "FAIL" for c in self.checks)
        n_w = sum(c["status"] == "WARN" for c in self.checks)
        lines.append(f"\nRESULT: {'OK' if self.ok else 'FAILED'} ({n_f} fail, {n_w} warn, "
                     f"{len(self.checks)} checks)")
        return "\n".join(lines)


def _load(run: Path, rep: Report):
    missing = [f for f in REQUIRED_FILES if not (run / f).exists()]
    if missing:
        rep.add("required files", "FAIL", "missing: " + ", ".join(missing))
        return None
    rep.add("required files", "PASS")
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    man = pd.read_csv(run / "manifest.csv", dtype={"id": str, "group": str, "label": str})
    pred = pd.read_csv(run / "predictions.csv",
                       dtype={"model": str, "id": str, "y_true": str, "y_pred": str})
    bad_p = [c for c in REQUIRED_PRED_COLS if c not in pred.columns]
    bad_m = [c for c in REQUIRED_MANIFEST_COLS if c not in man.columns]
    if bad_p or bad_m:
        rep.add("schema", "FAIL", f"predictions missing {bad_p}; manifest missing {bad_m}")
        return None
    rep.add("schema", "PASS")
    return cfg, metrics, man, pred


def verify_run(run_dir, data_root=None, tol: float = 1e-9) -> Report:
    run = Path(run_dir)
    rep = Report()
    loaded = _load(run, rep)
    if loaded is None:
        return rep
    cfg, metrics, man, pred = loaded

    # --- provenance fields -------------------------------------------------------------
    if "seed" in cfg:
        rep.add("seed recorded", "PASS", f"seed={cfg['seed']}")
    else:
        rep.add("seed recorded", "FAIL", "config.json has no 'seed'")
    env = json.loads((run / "env.json").read_text(encoding="utf-8"))
    commit = env.get("git_commit")
    if not commit:
        rep.add("git commit recorded", "WARN", "run was not made from a git checkout; code version unknown")
    elif str(commit).endswith("+dirty"):
        rep.add("git commit recorded", "WARN", f"{commit}: uncommitted changes were present")
    else:
        rep.add("git commit recorded", "PASS", str(commit)[:12])

    # --- ids ---------------------------------------------------------------------------
    if man["id"].duplicated().any():
        rep.add("manifest ids unique", "FAIL", f"{int(man['id'].duplicated().sum())} duplicated ids")
    else:
        rep.add("manifest ids unique", "PASS", f"{len(man)} files")
    unknown = set(pred["id"]) - set(man["id"])
    if unknown:
        rep.add("prediction ids in manifest", "FAIL", f"{len(unknown)} unknown ids, e.g. {sorted(unknown)[:3]}")
    else:
        rep.add("prediction ids in manifest", "PASS")
    dup_pred = pred.duplicated(["model", "id"]).sum()
    if dup_pred:
        rep.add("one prediction per (model, id)", "FAIL", f"{int(dup_pred)} duplicates")
    else:
        rep.add("one prediction per (model, id)", "PASS")

    models = list(metrics.get("models", {}).keys())
    classes = metrics.get("classes") or sorted(set(man["label"]))
    label_of = dict(zip(man["id"], man["label"]))
    group_of = dict(zip(man["id"], man["group"]))

    # --- recorded labels match the manifest -------------------------------------------
    mism = (pred["y_true"] != pred["id"].map(label_of)).sum()
    if mism:
        rep.add("y_true agrees with manifest labels", "FAIL", f"{int(mism)} rows differ")
    else:
        rep.add("y_true agrees with manifest labels", "PASS")

    # --- recompute metrics --------------------------------------------------------------
    recomputed: dict[str, dict] = {}
    for m in models:
        sub = pred[pred["model"] == m]
        if sub.empty:
            rep.add(f"metrics recomputed [{m}]", "FAIL", "no predictions for this model")
            continue
        if len(sub) < len(man):
            rep.add(f"coverage [{m}]", "WARN", f"{len(sub)} of {len(man)} manifest files have a prediction")
        s = summarize(sub["y_true"].tolist(), sub["y_pred"].tolist(), classes)
        recomputed[m] = s
        worst = 0.0
        for k in HEADLINE_KEYS:
            reported = metrics["models"][m].get(k)
            if reported is None or (isinstance(reported, float) and math.isnan(reported)):
                rep.add(f"metrics recomputed [{m}] {k}", "FAIL", "not reported")
                continue
            worst = max(worst, abs(float(reported) - s[k]))
        if worst > tol:
            rep.add(f"metrics recomputed [{m}]", "FAIL",
                    f"reported vs recomputed differ by up to {worst:.3g}")
        else:
            rep.add(f"metrics recomputed [{m}]", "PASS",
                    f"acc={s['accuracy']:.4f} bal_acc={s['balanced_accuracy']:.4f} macro_f1={s['macro_f1']:.4f}")

    # --- group leakage across folds ------------------------------------------------------
    for m in models:
        sub = pred[pred["model"] == m]
        if sub.empty or "fold" not in sub:
            continue
        g = sub["id"].map(group_of)
        nfold = sub.groupby(g)["fold"].nunique()
        straddle = int((nfold > 1).sum())
        if straddle:
            rep.add(f"no group in two folds [{m}]", "FAIL",
                    f"{straddle} group(s) appear as test data in more than one fold, so they were in training for another")
        else:
            rep.add(f"no group in two folds [{m}]", "PASS", f"{nfold.shape[0]} groups")

    if cfg.get("has_patient_ids") is False:
        rep.add("patient ids available", "WARN",
                "no patient/subject ids: each file was its own group. Smoke test only - cannot support clinical claims.")
    elif cfg.get("has_patient_ids") is True:
        rep.add("patient ids available", "PASS")
    else:
        rep.add("patient ids available", "WARN", "config.json does not say whether groups are patients")

    # --- duplicate audio across folds ----------------------------------------------------
    dups = man.groupby("sha256")["id"].apply(list)
    dups = dups[dups.map(len) > 1]
    if len(dups):
        cross = 0
        for m in models:
            sub = pred[pred["model"] == m].set_index("id")["fold"]
            for ids in dups:
                f = {sub.get(i) for i in ids if i in sub.index}
                if len(f) > 1:
                    cross += 1
        rep.add("duplicate audio", "WARN" if cross else "PASS",
                f"{len(dups)} duplicate-content groups; {cross} straddle folds" if cross else f"{len(dups)} groups, none straddle folds")
    else:
        rep.add("duplicate audio", "PASS", "no byte-identical files")

    # --- shuffled-label controls ----------------------------------------------------------
    controls = cfg.get("controls", {})
    k = len(classes)
    chance = 1.0 / k
    for real, ctrl in controls.items():
        if ctrl not in recomputed:
            rep.add(f"shuffled control for [{real}]", "FAIL", f"control model {ctrl!r} has no predictions")
            continue
        n = recomputed[ctrl]["n"]
        band = 4.0 * math.sqrt(chance * (1 - chance) / max(n, 1)) + 0.02
        ba = recomputed[ctrl]["balanced_accuracy"]
        if ba > chance + band:
            rep.add(f"shuffled control for [{real}]", "FAIL",
                    f"control balanced accuracy {ba:.3f} is above chance {chance:.3f} + {band:.3f}: leakage or a bug")
        else:
            rep.add(f"shuffled control for [{real}]", "PASS", f"{ba:.3f} (chance {chance:.3f}, band +{band:.3f})")
    real_models = [m for m in models if m not in controls.values()]
    for m in real_models:
        if m not in controls and m != cfg.get("trivial_baseline"):     # the baseline is a floor, not a claim
            rep.add(f"shuffled control for [{m}]", "WARN", "no shuffled-label control registered")

    # --- trivial-feature baseline ----------------------------------------------------------
    triv = cfg.get("trivial_baseline")
    if triv and triv in recomputed:
        tb = recomputed[triv]["balanced_accuracy"]
        for m in real_models:
            if m == triv or m not in recomputed:
                continue
            ba = recomputed[m]["balanced_accuracy"]
            if ba <= tb + 0.02:
                rep.add(f"beats trivial baseline [{m}]", "WARN",
                        f"bal_acc {ba:.3f} vs non-cardiac-features baseline {tb:.3f}: no clear gain over a dataset shortcut")
            else:
                rep.add(f"beats trivial baseline [{m}]", "PASS", f"{ba:.3f} vs {tb:.3f}")
        if tb > chance + 0.25:
            rep.add("dataset shortcut level", "WARN",
                    f"non-cardiac features alone reach balanced accuracy {tb:.3f} (chance {chance:.3f}); "
                    "the label is partly predictable from recording properties")
    else:
        rep.add("trivial baseline", "WARN", "no trivial-feature baseline registered, results cannot be put in context")

    # --- data checksums ---------------------------------------------------------------------
    if data_root:
        root = Path(data_root)
        bad = missing_f = 0
        for _, r in man.iterrows():
            p = root / r["relpath"]
            if not p.exists():
                missing_f += 1
            elif sha256_file(p) != r["sha256"]:
                bad += 1
        if bad:
            rep.add("data checksums", "FAIL", f"{bad} file(s) differ from the manifest")
        elif missing_f:
            rep.add("data checksums", "WARN", f"{missing_f} file(s) not found under {root}")
        else:
            rep.add("data checksums", "PASS", f"{len(man)} files match")
    else:
        rep.add("data checksums", "WARN", "skipped (pass --data-root to check files against the manifest)")

    (run / "verify_report.json").write_text(json.dumps(
        dict(ok=rep.ok, checks=rep.checks), indent=2), encoding="utf-8")
    return rep


def compare_runs(run_a, run_b) -> Report:
    """Two runs with the same seed/config/data should produce identical predictions."""
    rep = Report()
    a = pd.read_csv(Path(run_a) / "predictions.csv", dtype=str)
    b = pd.read_csv(Path(run_b) / "predictions.csv", dtype=str)
    m = a.merge(b, on=["model", "id"], suffixes=("_a", "_b"))
    if len(m) != len(a) or len(m) != len(b):
        rep.add("same (model, id) rows", "FAIL", f"{len(a)} vs {len(b)} rows, {len(m)} in common")
    else:
        rep.add("same (model, id) rows", "PASS", f"{len(m)} rows")
    diff = int((m["y_pred_a"] != m["y_pred_b"]).sum())
    rep.add("identical predictions", "FAIL" if diff else "PASS",
            f"{diff} of {len(m)} predictions differ" if diff else "")
    return rep
