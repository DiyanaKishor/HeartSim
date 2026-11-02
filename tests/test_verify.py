"""verify.py must PASS an honest run and FAIL each kind of tampering / leakage."""
import json

import pandas as pd

from heartsim.report import make_report
from heartsim.verify import compare_runs, verify_run

from helpers import make_run, status


def test_honest_run_passes(tmp_path):
    run = make_run(tmp_path / "r")
    rep = verify_run(run)
    assert rep.ok, rep.text()
    assert status(rep, "metrics recomputed [real]") == ["PASS"]
    assert status(rep, "shuffled control") == ["PASS"]
    assert status(rep, "no group in two folds") == ["PASS"] * 3


def test_edited_headline_number_fails(tmp_path):
    run = make_run(tmp_path / "r")
    m = json.loads((run / "metrics.json").read_text())
    m["models"]["real"]["accuracy"] += 0.05
    (run / "metrics.json").write_text(json.dumps(m))
    rep = verify_run(run)
    assert not rep.ok and status(rep, "metrics recomputed [real]") == ["FAIL"]


def test_edited_prediction_fails(tmp_path):
    run = make_run(tmp_path / "r")
    p = pd.read_csv(run / "predictions.csv", dtype=str)
    i = p.index[(p.model == "real") & (p.y_true != p.y_pred)][0]
    p.loc[i, "y_pred"] = p.loc[i, "y_true"]
    p.to_csv(run / "predictions.csv", index=False)
    assert not verify_run(run).ok


def test_group_straddling_two_folds_fails(tmp_path):
    run = make_run(tmp_path / "r", group_size=2)
    p = pd.read_csv(run / "predictions.csv", dtype={"id": str, "model": str})
    # id0000 and id0001 share a group; move one of them to a different fold
    j = p.index[(p.model == "real") & (p.id == "id0001")][0]
    p.loc[j, "fold"] = (p.loc[j, "fold"] + 1) % 4
    p.to_csv(run / "predictions.csv", index=False)
    rep = verify_run(run)
    assert not rep.ok and "FAIL" in status(rep, "no group in two folds [real]")


def test_control_that_beats_chance_fails(tmp_path):
    run = make_run(tmp_path / "r")
    p = pd.read_csv(run / "predictions.csv", dtype=str)
    m = p.model == "real_shuffled"
    p.loc[m, "y_pred"] = p.loc[m, "y_true"]            # a "control" that knows the answer = leakage
    p.to_csv(run / "predictions.csv", index=False)
    metrics = json.loads((run / "metrics.json").read_text())
    metrics["models"]["real_shuffled"] = {k: 1.0 for k in ("accuracy", "balanced_accuracy", "macro_f1")}
    (run / "metrics.json").write_text(json.dumps(metrics))
    rep = verify_run(run)
    assert not rep.ok and status(rep, "shuffled control for [real]") == ["FAIL"]


def test_missing_control_warns(tmp_path):
    rep = verify_run(make_run(tmp_path / "r", with_controls=False))
    assert rep.ok and "WARN" in status(rep, "shuffled control")


def test_no_patient_ids_warns_but_does_not_fail(tmp_path):
    rep = verify_run(make_run(tmp_path / "r", patient_ids=False))
    assert rep.ok and status(rep, "patient ids") == ["WARN"]


def test_model_not_beating_trivial_baseline_warns(tmp_path):
    run = make_run(tmp_path / "r")
    cfg = json.loads((run / "config.json").read_text())
    cfg["trivial_baseline"] = "real"          # now "real" is the baseline; compare 'trivial' to it
    (run / "config.json").write_text(json.dumps(cfg))
    rep = verify_run(run)
    assert "WARN" in status(rep, "beats trivial baseline [trivial]")


def test_checksums(tmp_path):
    data = tmp_path / "data"
    run = make_run(tmp_path / "r", data_dir=data)
    assert status(verify_run(run, data_root=data), "data checksums") == ["PASS"]
    (data / "C0" / "f0000.bin").write_bytes(b"tampered")
    rep = verify_run(run, data_root=data)
    assert not rep.ok and status(rep, "data checksums") == ["FAIL"]


def test_missing_files_fail(tmp_path):
    run = make_run(tmp_path / "r")
    (run / "predictions.csv").unlink()
    assert not verify_run(run).ok


def test_compare_runs(tmp_path):
    a = make_run(tmp_path / "a", seed=1)
    b = make_run(tmp_path / "b", seed=1)
    c = make_run(tmp_path / "c", seed=2)
    assert compare_runs(a, b).ok
    assert not compare_runs(a, c).ok


def test_report_zip_contains_only_allowed_files(tmp_path):
    import zipfile
    run = make_run(tmp_path / "r")
    (run / "secret.wav").write_bytes(b"audio")
    z = make_report(run, tmp_path / "out.zip")
    names = [n.split("/", 1)[1] for n in zipfile.ZipFile(z).namelist()]
    assert "secret.wav" not in names and "predictions.csv" in names
