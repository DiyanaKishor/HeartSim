"""Plumbing test: the whole smoke pipeline on a tiny SYNTHETIC dataset.  This checks that the
harness runs, writes a verifiable folder and is deterministic.  It says nothing about real
diagnostic performance."""
import json

import pytest

from heartsim.smoke import run_smoke
from heartsim.verify import compare_runs, verify_run

from helpers import status, write_synth_dataset

SIM = dict(grid_n=48, sponge_width=8)


def test_pipeline_end_to_end_and_deterministic(tmp_path):
    data = write_synth_dataset(tmp_path / "data", n_per_class=12, n_classes=3)
    out = tmp_path / "runs"
    r1 = run_smoke(str(data), str(out), "a", seed=3, n_splits=3, sim_overrides=SIM, repo_dir=str(tmp_path))
    rep = verify_run(r1, data_root=data)
    assert rep.ok, rep.text()
    assert status(rep, "data checksums") == ["PASS"]
    assert status(rep, "patient ids") == ["WARN"]                 # files used as their own groups
    cfg = json.loads((r1 / "config.json").read_text())
    assert cfg["seed"] == 3 and cfg["controls"] == {"spectral": "spectral_shuffled", "sim": "sim_shuffled"}
    # rerun with identical settings (features now come from the cache) -> identical predictions
    r2 = run_smoke(str(data), str(out), "b", seed=3, n_splits=3, sim_overrides=SIM, repo_dir=str(tmp_path))
    assert compare_runs(r1, r2).ok
    # a different seed changes the folds/forests, so predictions need not match
    m = json.loads((r1 / "metrics.json").read_text())
    assert set(m["models"]) == {"trivial", "spectral", "sim", "spectral_shuffled", "sim_shuffled"}


def test_groups_csv_marks_patient_ids_present(tmp_path):
    data = write_synth_dataset(tmp_path / "data", n_per_class=12, n_classes=3)
    rows = ["file,group"]
    for cls in ("N", "AS", "MR"):
        for i in range(12):
            rows.append(f"{cls}/{cls}_{i:03d}.wav,P{cls}{i // 2}")      # 2 recordings per 'patient'
    (tmp_path / "groups.csv").write_text("\n".join(rows))
    r = run_smoke(str(data), str(tmp_path / "runs"), "g", seed=1, n_splits=3,
                  groups_csv=str(tmp_path / "groups.csv"), sim_overrides=SIM, repo_dir=str(tmp_path))
    rep = verify_run(r, data_root=data)
    assert rep.ok, rep.text()
    assert status(rep, "patient ids") == ["PASS"]
    assert all(s == "PASS" for s in status(rep, "no group in two folds"))


def test_band_too_close_to_sim_nyquist_is_rejected(tmp_path):
    data = write_synth_dataset(tmp_path / "data", n_per_class=4, n_classes=3)
    with pytest.raises(ValueError):
        run_smoke(str(data), str(tmp_path / "runs"), "x", n_splits=2,
                  sim_overrides=dict(grid_n=48), pre_overrides=dict(band_hi_hz=480.0))
