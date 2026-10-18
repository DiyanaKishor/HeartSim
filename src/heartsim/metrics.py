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
