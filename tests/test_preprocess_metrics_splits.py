import numpy as np
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score

from heartsim.metrics import summarize
from heartsim.preprocess import PreprocessParams, bandpass, normalize, preprocess, quality_flags
from heartsim.splits import assert_disjoint_groups, grouped_folds, leave_one_group_out

from helpers import synth_beats


# ---- preprocess ---------------------------------------------------------------------
def test_dc_offset_is_removed():
    x, fs = synth_beats()
    y = preprocess(x + 0.3, fs, PreprocessParams(normalize="none"))
    assert abs(y.mean()) < 1e-3


def test_bandpass_attenuates_out_of_band_tones():
    fs = 8000
    t = np.arange(0, 2, 1 / fs)
    p = PreprocessParams(normalize="none")
    inband = preprocess(np.sin(2 * np.pi * 100 * t), fs, p)
    high = preprocess(np.sin(2 * np.pi * 1500 * t), fs, p)
    low = preprocess(np.sin(2 * np.pi * 5 * t), fs, p)
    r = lambda v: np.sqrt(np.mean(v[2000:-2000] ** 2))
    assert 20 * np.log10(r(inband) / r(high)) > 40
    assert 20 * np.log10(r(inband) / r(low)) > 25


def test_bad_band_raises():
    with pytest.raises(ValueError):
        bandpass(np.zeros(100), 1000, 25, 600)


def test_normalisation_levels():
    x = np.random.default_rng(0).standard_normal(5000) * 7
    assert np.isclose(np.sqrt(np.mean(normalize(x, "rms", 0.2) ** 2)), 0.2)
    assert np.isclose(np.percentile(np.abs(normalize(x, "p99", 0.2)), 99), 0.2)
    with pytest.raises(ValueError):
        normalize(x, "nope", 1.0)


def test_filter_before_normalise_limits_noise_shrinkage():
    """The audit found that normalising BEFORE filtering halved the heart-sound amplitude at
    -5 dB white-noise SNR (relative gain 0.49).  With band-pass first it must stay above 0.7."""
    x, fs = synth_beats(seconds=2.4, seed=3)
    noise = np.random.default_rng(1).standard_normal(len(x))
    noisy = x + noise * np.sqrt(np.mean(x ** 2) / 10 ** (-5 / 10))
    p = PreprocessParams(normalize="none")
    g = lambda s, mode: np.std(preprocess(s, fs, PreprocessParams(normalize=mode))) / np.std(preprocess(s, fs, p))
    assert g(noisy, "rms") / g(x, "rms") > 0.7


def test_quality_flags_detect_clipping_and_flatline():
    t = np.arange(0, 1, 1 / 4000)
    clipped = np.clip(3 * np.sin(2 * np.pi * 50 * t), -1, 1)
    q = quality_flags(clipped, 4000)
    assert q["clip_fraction"] > 0.2
    assert quality_flags(np.zeros(1000), 4000)["flatline_fraction"] == 1.0


# ---- metrics: numpy implementation must agree with scikit-learn -------------------------
def test_metrics_agree_with_sklearn():
    rng = np.random.default_rng(5)
    classes = ["a", "b", "c", "d"]
    yt = rng.choice(classes, 400)
    yp = np.where(rng.random(400) < 0.6, yt, rng.choice(classes, 400))
    s = summarize(yt.tolist(), yp.tolist(), classes)
    assert np.isclose(s["accuracy"], accuracy_score(yt, yp))
    assert np.isclose(s["balanced_accuracy"], balanced_accuracy_score(yt, yp))
    assert np.isclose(s["macro_f1"], f1_score(yt, yp, labels=classes, average="macro"))
    rec = recall_score(yt, yp, labels=classes, average=None)
    assert np.allclose([s["per_class"][c]["sensitivity"] for c in classes], rec)


# ---- splits -----------------------------------------------------------------------------
def test_grouped_folds_never_split_a_group():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(60), 3)
    labels = np.repeat(rng.choice(["x", "y", "z"], 60), 3)
    seen_test = np.zeros(len(groups), int)
    for _, tr, te in grouped_folds(labels, groups, n_splits=5, seed=1):
        assert not (set(groups[tr]) & set(groups[te]))
        seen_test[te] += 1
    assert np.all(seen_test == 1)       # every sample is tested exactly once


def test_leakage_assertion_fires():
    with pytest.raises(AssertionError):
        assert_disjoint_groups(["a", "b"], ["b", "c"])


def test_leave_one_group_out_covers_all_groups():
    g = np.array(list("aabbbcc"))
    out = list(leave_one_group_out(g))
    assert [o[0] for o in out] == ["a", "b", "c"]
    for name, tr, te in out:
        assert set(g[te]) == {name} and name not in set(g[tr])
