"""Run folders.  Every experiment writes ONE folder that contains everything a reviewer needs
to re-derive its headline numbers without trusting the person (or model) that produced it:

  config.json       all parameters, seed, the flags that qualify the result (see below)
  env.json          python/package versions, hardware, git commit (+dirty marker)
  manifest.csv      id, relpath, sha256, bytes, label, group, source_group, n_samples, fs
  predictions.csv   model, id, fold, y_true, y_pred  (+ per-class probability columns)
  metrics.json      headline numbers per model as REPORTED by the training code
  log.txt           human-readable log
  verify_report.json  written by `heartsim verify`

Qualifying flags in config.json that verify.py reads:
  has_patient_ids   False -> results are smoke-test only (e.g. the Yaseen dataset has no patient ids)
  controls          mapping real-model name -> its shuffled-label control model name
  trivial_baseline  name of the model that uses only non-cardiac features
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from pathlib import Path

from .env import env_info


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def new_run_dir(root, name: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    d = Path(root) / f"{stamp}_{name}"
    d.mkdir(parents=True, exist_ok=False)
    return d


class RunLog:
    def __init__(self, run_dir, repo_dir=None):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._log = open(self.dir / "log.txt", "a", encoding="utf-8")
        self.repo_dir = repo_dir

    def log(self, msg: str) -> None:
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        self._log.write(line + "\n")
        self._log.flush()

    def write_json(self, name: str, obj) -> None:
        with open(self.dir / name, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True, default=_jsonable)

    def write_env(self) -> None:
        self.write_json("env.json", env_info(self.repo_dir))

    def write_csv(self, name: str, rows: list[dict], fieldnames: list[str]) -> None:
        with open(self.dir / name, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    def close(self) -> None:
        self._log.close()


def _jsonable(o):
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)


MANIFEST_FIELDS = ["id", "relpath", "sha256", "bytes", "label", "group", "n_samples", "fs"]
PRED_BASE_FIELDS = ["model", "id", "fold", "y_true", "y_pred"]
