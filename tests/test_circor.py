"""CirCor support: file parsing, patient grouping, discovery, the full run and its verification."""
import json

import numpy as np
import pandas as pd
import pytest

from heartsim import circor
from heartsim.metrics import aggregate_patients
from heartsim.smoke import run_smoke
from heartsim.verify import compare_runs, verify_run

from helpers import status
from helpers_circor import write_synth_circor

SIM = dict(grid_n=48, sponge_width=8)


# ----------------------------------------------------------------------------- parsers
def test_read_tsv_and_hea_match_the_real_format(tmp_path):
    p = tmp_path / "2530_AV.tsv"
    p.write_text("0\t0.242\t0\n0.242\t0.400176\t1\n0.400176\t0.497088\t2\n0.497088\t0.620176\t3\n")
    a = circor.read_tsv(p)
    assert a.shape == (4, 3) and a[1, 2] == 1 and a[1, 0] == pytest.approx(0.242)
    h = tmp_path / "2530_AV.hea"
    h.write_text("2530_AV 1 4000 94400\n2530_AV.wav 16+44 1 16 0 0 0 0 AV\n")
    assert circor.parse_hea(h) == (4000, 94400)
    (tmp_path / "empty.tsv").write_text("")
    assert circor.read_tsv(tmp_path / "empty.tsv").shape == (0, 3)
    (tmp_path / "bad.tsv").write_text("1\t2\n")
    with pytest.raises(ValueError):
        circor.read_tsv(tmp_path / "bad.tsv")


def test_excerpt_starts_at_first_s1_and_handles_edges():
    fs = 4000
    tsv = np.array([[0, 0.242, 0], [0.242, 0.4, 1], [0.4, 0.5, 2]])
    assert circor.excerpt_range(tsv, fs, 20 * fs, 4.0) == (int(round(0.242 * fs)), int(round(0.242 * fs)) + 4 * fs)
    # S1 too close to the end: the excerpt is moved back so it still has full length
    late = np.array([[9.5, 9.7, 1]])
    assert circor.excerpt_range(late, fs, 10 * fs, 4.0) == (6 * fs, 10 * fs)
    # no S1 at all -> sample 0 ; no rows at all -> sample 0
    assert circor.excerpt_range(np.array([[0, 1, 0.0]]), fs, 10 * fs, 4.0)[0] == 0
    assert circor.excerpt_range(np.zeros((0, 3)), fs, 10 * fs, 4.0)[0] == 0
    # shorter than 75% of the requested length -> rejected
    assert circor.excerpt_range(tsv, fs, int(2.9 * fs), 4.0) is None
    assert circor.excerpt_range(tsv, fs, int(3.1 * fs), 4.0) is not None


# ----------------------------------------------------------------------------- metadata and grouping
def test_additional_id_links_are_merged_transitively(tmp_path):
    facts = write_synth_circor(tmp_path)
    meta = circor.load_metadata(tmp_path / "training_data.csv")
    assert "Recording locations" in meta.columns            # trailing colon removed
    assert (meta["Additional ID"] == "").sum() > 0          # the text 'nan' is treated as missing
    groups, n_merged = circor.patient_groups(meta)
    linked = facts["linked"]
    assert len({groups[p] for p in linked}) == 1 and n_merged == len(linked)
    assert len(set(groups.values())) == facts["n_patients"] - len(linked) + 1


def test_additional_id_pointing_outside_the_table_still_groups(tmp_path):
    csv = tmp_path / "t.csv"
    csv.write_text("Patient ID,Murmur,Outcome,Additional ID\n1,Present,Abnormal,99.0\n2,Absent,Normal,99\n3,Absent,Normal,nan\n")
    groups, n = circor.patient_groups(circor.load_metadata(csv))
    assert groups["1"] == groups["2"] != groups["3"] and n == 2


def test_discover_drops_and_counts_and_parses_names(tmp_path):
    facts = write_synth_circor(tmp_path)
    items, info = circor.discover_circor(tmp_path, "murmur")
    assert info["classes"] == ["Absent", "Present"]
    assert info["dropped"]["label_Unknown"] >= 1 and info["dropped"]["no_tsv"] == 1
    assert all(it["patient"] != facts["unknown_murmur"] for it in items)
    assert facts["no_tsv"] not in {it["relpath"].split("/")[1][:-4] for it in items}
    suffixed = [it for it in items if it["relpath"].endswith("1000_AV_1.wav")]
    assert len(suffixed) == 1 and suffixed[0]["location"] == "AV"
    # the label really is patient-level
    lab = {}
    for it in items:
        assert lab.setdefault(it["patient"], it["label"]) == it["label"]
    out_items, out_info = circor.discover_circor(tmp_path, "outcome")
    assert out_info["classes"] == ["Abnormal", "Normal"]
    assert out_info["dropped"].get("label_Unknown") is None            # Unknown murmur is fine for Outcome
    assert len(out_items) > len(items)
    with pytest.raises(SystemExit):
        circor.discover_circor(tmp_path, "bogus")
    with pytest.raises(SystemExit):
        circor.discover_circor(tmp_path / "nowhere", "outcome")


def test_meta_matrix_has_no_audio_and_handles_missing_numbers(tmp_path):
    write_synth_circor(tmp_path)
    items, _ = circor.discover_circor(tmp_path, "outcome")
    items[0]["meta"]["Height"] = ""
    X, names = circor.build_meta_matrix(items)
    assert X.shape == (len(items), len(names)) and X[0, names.index("height")] == -1.0
    assert not np.isnan(X).any()


# ----------------------------------------------------------------------------- patient aggregation
def test_aggregate_patients_rule_and_label_guard():
    classes = ["Abnormal", "Normal"]
    pats = ["a", "a", "b", "b", "b"]
    ids = ["r1", "r2", "r3", "r4", "r5"]
    y = ["Abnormal", "Abnormal", "Normal", "Normal", "Normal"]
    P = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.5, 0.5], [0.3, 0.7]])
    p, t, yp = aggregate_patients(pats, ids, y, P, classes)
    assert p == ["a", "b"] and t == ["Abnormal", "Normal"]
    assert yp == ["Abnormal", "Normal"]                    # a: mean (0.65, 0.35) ; b: mean (0.33, 0.67)
    # exact tie goes to the first class
    assert aggregate_patients(["x", "x"], ["1", "2"], ["Normal"] * 2,
                              np.array([[0.7, 0.3], [0.3, 0.7]]), classes)[2] == ["Abnormal"]
    with pytest.raises(ValueError):
        aggregate_patients(["a", "a"], ["1", "2"], ["Normal", "Abnormal"], P[:2], classes)


# ----------------------------------------------------------------------------- the whole run
@pytest.fixture(scope="module")
def circor_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("circor")
    facts = write_synth_circor(tmp / "data")
    out = tmp / "runs"
    run = run_smoke(str(tmp / "data"), str(out), "c", seed=2, n_splits=3, sim_overrides=SIM,
                    repo_dir=str(tmp), dataset="circor", target="outcome", excerpt_s=3.0)
    return tmp, facts, out, run


def test_circor_run_is_verifiable_and_patient_grouped(circor_run):
    tmp, facts, _, run = circor_run
    rep = verify_run(run, data_root=tmp / "data")
    assert rep.ok, rep.text()
    assert status(rep, "patient ids") == ["PASS"]
    assert status(rep, "one label per patient") == ["PASS"] and status(rep, "one group per patient") == ["PASS"]
    assert status(rep, "data checksums") == ["PASS"]
    assert all(s == "PASS" for s in status(rep, "patient-level metrics"))
    assert len(status(rep, "patient-level metrics")) == 5 + 1          # 5 audio-model rows + the meta baseline
    m = json.loads((run / "metrics.json").read_text())
    assert set(m["models"]) == {"trivial", "meta", "spectral", "sim", "spectral_shuffled", "sim_shuffled"}
    assert m["classes"] == ["Abnormal", "Normal"]
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["baselines"] == ["trivial", "meta"] and cfg["dataset"] == "circor"
    assert cfg["circor"]["dropped"]["too_short"] == 1 and cfg["circor"]["dropped"]["no_tsv"] == 1
    # the linked patients ended up in one group and never straddle folds
    man = pd.read_csv(run / "manifest.csv", dtype=str)
    assert man.loc[man.patient.isin(facts["linked"]), "group"].nunique() == 1
    assert {"patient", "aux_sha256", "ex_start", "ex_end"} <= set(man.columns)
    assert all(s == "PASS" for s in status(rep, "no group in two folds"))
    assert any("beats meta baseline" in c["name"] for c in rep.checks)


def test_circor_run_is_deterministic_and_reuses_cache(circor_run):
    tmp, _, out, run = circor_run
    run2 = run_smoke(str(tmp / "data"), str(out), "c2", seed=2, n_splits=3, sim_overrides=SIM,
                     repo_dir=str(tmp), dataset="circor", target="outcome", excerpt_s=3.0)
    assert compare_runs(run, run2).ok


def test_changing_excerpt_length_changes_the_features_not_the_cache_hit(circor_run):
    tmp, _, out, run = circor_run
    run3 = run_smoke(str(tmp / "data"), str(out), "c3", seed=2, n_splits=3, sim_overrides=SIM,
                     repo_dir=str(tmp), dataset="circor", target="outcome", excerpt_s=2.0, max_per_class=15)
    m1 = pd.read_csv(run / "manifest.csv")
    m3 = pd.read_csv(run3 / "manifest.csv")
    assert (m3.ex_end - m3.ex_start).max() <= 2 * 4000 and (m1.ex_end - m1.ex_start).max() == 3 * 4000


def test_tampered_patient_level_number_fails(circor_run, tmp_path):
    import shutil
    _, _, _, run = circor_run
    r = tmp_path / "copy"
    shutil.copytree(run, r)
    m = json.loads((r / "metrics.json").read_text())
    m["models"]["sim"]["patient_level"]["accuracy"] += 0.05
    (r / "metrics.json").write_text(json.dumps(m))
    rep = verify_run(r)
    assert not rep.ok and status(rep, "patient-level metrics [sim]") == ["FAIL"]


def test_patient_with_two_labels_or_two_groups_fails(circor_run, tmp_path):
    import shutil
    _, _, _, run = circor_run
    r = tmp_path / "copy2"
    shutil.copytree(run, r)
    man = pd.read_csv(r / "manifest.csv", dtype=str)
    pid = man.patient.iloc[0]
    rows = man.index[man.patient == pid]
    assert len(rows) >= 2
    man.loc[rows[0], "group"] = "SOMEWHERE_ELSE"
    man.to_csv(r / "manifest.csv", index=False)
    assert status(verify_run(r), "one group per patient") == ["FAIL"]
    man = pd.read_csv(r / "manifest.csv", dtype=str)
    man.loc[rows[0], "group"] = man.loc[rows[1], "group"]
    other = "Normal" if man.loc[rows[0], "label"] == "Abnormal" else "Abnormal"
    man.loc[rows[0], "label"] = other
    man.to_csv(r / "manifest.csv", index=False)
    assert status(verify_run(r), "one label per patient") == ["FAIL"]


def test_edited_tsv_is_caught_by_the_checksum(circor_run, tmp_path):
    import shutil
    tmp, _, _, run = circor_run
    data = tmp_path / "data_copy"
    shutil.copytree(tmp / "data", data)
    some = next((data / "training_data").glob("*.tsv"))
    some.write_text(some.read_text() + "99\t99.5\t1\n")
    rep = verify_run(run, data_root=data)
    assert not rep.ok and status(rep, "data checksums") == ["FAIL"]


def test_murmur_target_runs_too(tmp_path):
    write_synth_circor(tmp_path / "data")
    run = run_smoke(str(tmp_path / "data"), str(tmp_path / "runs"), "m", seed=1, n_splits=3,
                    sim_overrides=SIM, repo_dir=str(tmp_path), dataset="circor", target="murmur",
                    excerpt_s=3.0)
    rep = verify_run(run, data_root=tmp_path / "data")
    assert rep.ok, rep.text()
    assert json.loads((run / "metrics.json").read_text())["classes"] == ["Absent", "Present"]


# ----------------------------------------------------------------------------- AUROC and class weights
def test_macro_auroc_matches_sklearn_including_ties():
    from sklearn.metrics import roc_auc_score
    from heartsim.metrics import macro_auroc
    r = np.random.default_rng(1)
    y = r.choice(["a", "b", "c"], 300)
    P = np.round(r.dirichlet([1, 1, 1], 300), 1)
    P = P / P.sum(axis=1, keepdims=True)                   # many exact ties on purpose
    want = np.mean([roc_auc_score(y == c, P[:, j]) for j, c in enumerate("abc")])
    assert macro_auroc(y, P, ["a", "b", "c"]) == pytest.approx(want, abs=1e-12)
    assert np.isnan(macro_auroc(["x", "x"], np.array([[1.0, 0.0], [1.0, 0.0]]), ["x", "y"]))


def test_balancing_raises_minority_recall_on_imbalanced_data():
    from heartsim.smoke import _fit_predict
    r = np.random.default_rng(0)
    n = 600
    y = np.where(r.random(n) < 0.12, "Present", "Absent")
    X = r.standard_normal((n, 6))
    X[y == "Present", 0] += 0.9                            # weak but real signal
    classes = ["Absent", "Present"]
    tr, te = np.arange(0, 400), np.arange(400, n)
    plain, p_plain = _fit_predict(X, y, tr, te, 0, classes, False, False)
    bal, p_bal = _fit_predict(X, y, tr, te, 0, classes, False, True)
    rec = lambda yp: np.mean(yp[y[te] == "Present"] == "Present")
    assert rec(plain) < 0.3 < rec(bal)                     # plain forest ignores the minority, balanced does not
    assert (bal == np.array(classes)[p_bal.argmax(axis=1)]).all()      # predictions are the argmax of the probabilities
    assert np.allclose(p_bal.sum(axis=1), 1.0)


def test_run_reports_auroc_and_verify_recomputes_it(circor_run, tmp_path):
    import shutil
    _, _, _, run = circor_run
    rep = verify_run(run)
    assert rep.ok, rep.text()
    assert status(rep, "auroc recomputed") == ["PASS"] * 6
    m = json.loads((run / "metrics.json").read_text())
    assert all("auroc" in v and "auroc" in v["patient_level"] for v in m["models"].values())
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["balance"] is True and "prior-corrected" in cfg["classifier"]
    r = tmp_path / "tampered"
    shutil.copytree(run, r)
    m["models"]["sim"]["auroc"] = 0.99
    (r / "metrics.json").write_text(json.dumps(m))
    assert status(verify_run(r), "auroc recomputed [sim]") == ["FAIL"]
    m = json.loads((run / "metrics.json").read_text())
    m["models"]["sim"]["patient_level"]["auroc"] += 0.05
    (r / "metrics.json").write_text(json.dumps(m))
    assert status(verify_run(r), "patient-level metrics [sim]") == ["FAIL"]


def test_folder_datasets_keep_unweighted_forests_by_default(tmp_path):
    from helpers import write_synth_dataset
    data = write_synth_dataset(tmp_path / "data", n_per_class=12, n_classes=3)
    r = run_smoke(str(data), str(tmp_path / "runs"), "f", seed=3, n_splits=3, sim_overrides=SIM, repo_dir=str(tmp_path))
    assert json.loads((r / "config.json").read_text())["balance"] is False
    rep = verify_run(r, data_root=data)
    assert rep.ok, rep.text()
    assert status(rep, "auroc recomputed")                  # AUROC is reported for every dataset layout
