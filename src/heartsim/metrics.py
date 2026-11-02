"""Classification metrics implemented in plain numpy.

verify.py recomputes every headline number with THIS module, while the training scripts
report numbers computed by scikit-learn.  Two independent implementations disagreeing
is the signal that something is wrong (a bug, an edited file, or a mismatched label order).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def confusion(y_true, y_pred, classes):
    idx = {c: i for i, c in enumerate(classes)}
    m = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        m[idx[t], idx[p]] += 1
    return m


def summarize(y_true, y_pred, classes) -> dict:
    cm = confusion(y_true, y_pred, classes)
    n = cm.sum()
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    pred_n = cm.sum(axis=0).astype(float)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, pred_n, out=np.zeros_like(tp), where=pred_n > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    fp = pred_n - tp
    tn = n - support - fp
    spec = np.divide(tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) > 0)
    present = support > 0
    return dict(
        n=int(n),
        accuracy=float(tp.sum() / n) if n else float("nan"),
        balanced_accuracy=float(recall[present].mean()) if present.any() else float("nan"),
        macro_f1=float(f1[present].mean()) if present.any() else float("nan"),
        per_class={c: dict(sensitivity=float(recall[i]), specificity=float(spec[i]),
                           support=int(support[i])) for i, c in enumerate(classes)},
        confusion=cm.tolist(),
        classes=list(classes),
    )


HEADLINE_KEYS = ("accuracy", "balanced_accuracy", "macro_f1")


def patient_mean_proba(patients, ids, y_true, proba):
    """Mean class-probability vector per patient.  Recordings are visited in (patient, id) order so
    the arithmetic is identical wherever this is called.  Returns (patient_ids, labels, matrix).
    Raises ValueError if one patient carries two different labels."""
    proba = np.asarray(proba, dtype=np.float64)
    order = sorted(range(len(ids)), key=lambda i: (str(patients[i]), str(ids[i])))
    by_pat: dict[str, list[int]] = {}
    for i in order:
        by_pat.setdefault(str(patients[i]), []).append(i)
    out_p, out_t, rows = [], [], []
    for p, idx in by_pat.items():
        labels = {y_true[i] for i in idx}
        if len(labels) != 1:
            raise ValueError(f"patient {p} has more than one label: {sorted(labels)}")
        out_p.append(p)
        out_t.append(labels.pop())
        rows.append(proba[idx].mean(axis=0))
    return out_p, out_t, np.vstack(rows)


def aggregate_patients(patients, ids, y_true, proba, classes):
    """Recording-level -> patient-level.  A patient's prediction is the class with the highest
    MEAN probability over their recordings (mean rounded to 9 decimals, ties go to the class that
    comes first in `classes`).  Used both by the training code and by verify.py: the metric
    arithmetic is independent, this decision rule is the one shared definition.

    Returns (patient_ids, y_true_per_patient, y_pred_per_patient), sorted by patient id.
    """
    p, t, mean = patient_mean_proba(patients, ids, y_true, proba)
    y = [classes[int(np.argmax(np.round(m, 9)))] for m in mean]
    return p, t, y


def _auc_rank(pos_mask, score) -> float:
    """AUROC from average ranks (Mann-Whitney U); ties count one half.  nan if a side is empty."""
    pos_mask = np.asarray(pos_mask, dtype=bool)
    n_pos = int(pos_mask.sum())
    n_neg = len(pos_mask) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(np.asarray(score, dtype=np.float64))
    return float((r[pos_mask].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def macro_auroc(y_true, proba, classes) -> float:
    """One-vs-rest AUROC averaged over the classes that have both positives and negatives.
    For two classes this is simply the AUROC of the second class."""
    y = np.asarray(y_true)
    proba = np.asarray(proba, dtype=np.float64)
    vals = [a for j, c in enumerate(classes)
            for a in [_auc_rank(y == c, proba[:, j])] if not np.isnan(a)]
    return float(np.mean(vals)) if vals else float("nan")
