"""Classification metrics implemented in plain numpy.

verify.py recomputes every headline number with THIS module, while the training scripts
report numbers computed by scikit-learn.  Two independent implementations disagreeing
is the signal that something is wrong (a bug, an edited file, or a mismatched label order).
"""
from __future__ import annotations

import numpy as np


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


def aggregate_patients(patients, ids, y_true, proba, classes):
    """Recording-level -> patient-level.  A patient's prediction is the class with the highest
    MEAN probability over their recordings (mean rounded to 9 decimals, ties go to the class that
    comes first in `classes`).  Used both by the training code and by verify.py: the metric
    arithmetic is independent, this decision rule is the one shared definition.

    Returns (patient_ids, y_true_per_patient, y_pred_per_patient), sorted by patient id.
    Raises ValueError if one patient carries two different labels.
    """
    proba = np.asarray(proba, dtype=np.float64)
    order = sorted(range(len(ids)), key=lambda i: (str(patients[i]), str(ids[i])))
    by_pat: dict[str, list[int]] = {}
    for i in order:
        by_pat.setdefault(str(patients[i]), []).append(i)
    out_p, out_t, out_y = [], [], []
    for p, idx in by_pat.items():
        labels = {y_true[i] for i in idx}
        if len(labels) != 1:
            raise ValueError(f"patient {p} has more than one label: {sorted(labels)}")
        mean = np.round(proba[idx].mean(axis=0), 9)
        out_p.append(p)
        out_t.append(labels.pop())
        out_y.append(classes[int(np.argmax(mean))])
    return out_p, out_t, out_y
