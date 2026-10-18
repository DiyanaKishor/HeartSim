"""Group-aware splitting.  The rule: a group (patient, or recording source/database) is never
split across train and test.  Splitting by cycle, frame or file when several come from the
same patient is the classic way to get an impressive and meaningless accuracy.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def grouped_folds(labels, groups, n_splits: int = 5, seed: int = 0):
    """Yield (fold_index, train_idx, test_idx) with disjoint groups and balanced classes."""
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(skf.split(np.zeros(len(labels)), labels, groups)):
        assert_disjoint_groups(groups[tr], groups[te])
        yield k, tr, te


def leave_one_group_out(groups):
    """Yield (group_name, train_idx, test_idx).  Use with recording-database ids to measure
    cross-database generalisation (the published cross-corpus specificity drops are large)."""
    groups = np.asarray(groups)
    for g in sorted(set(groups.tolist())):
        te = np.where(groups == g)[0]
        tr = np.where(groups != g)[0]
        yield g, tr, te


def assert_disjoint_groups(train_groups, test_groups) -> None:
    overlap = set(np.asarray(train_groups).tolist()) & set(np.asarray(test_groups).tolist())
    if overlap:
        raise AssertionError(f"group leakage: {len(overlap)} group(s) in both train and test, "
                             f"e.g. {sorted(overlap)[:3]}")
