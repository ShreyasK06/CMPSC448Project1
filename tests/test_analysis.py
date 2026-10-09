"""Tests for the pure helpers in src.analysis (stats, cv_tfidf, integrity)."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.analysis.cv_tfidf import grouped_folds, inner_split, mean_std
from src.analysis.integrity import normalise, replace_section, split_overlaps, text_hash
from src.analysis.stats import (bootstrap_weights, cluster_bootstrap_ci, cluster_confusions, holm,
                                mcnemar_exact, metrics_from_confusion)


def _data(n_ids=40, seed=0):
    rng = np.random.default_rng(seed)
    sid = np.repeat([f"id{i}" for i in range(n_ids)], 5)
    y = np.tile(np.arange(5), n_ids)
    pred = np.where(rng.random(len(y)) < 0.7, y, rng.integers(0, 5, len(y)))
    return y, pred, sid


def test_metrics_from_confusion_matches_sklearn():
    y, pred, sid = _data()
    _, C = cluster_confusions(y, pred, sid)
    acc, f1 = metrics_from_confusion(C.sum(0))
    assert np.isclose(acc, accuracy_score(y, pred))
    assert np.isclose(f1, f1_score(y, pred, average="macro", labels=range(5), zero_division=0))


def test_bootstrap_weights_rows_sum_to_n():
    W = bootstrap_weights(30, 50, seed=1)
    assert W.shape == (50, 30) and np.all(W.sum(1) == 30)


def test_bootstrap_ci_contains_point_and_is_deterministic():
    y, pred, sid = _data()
    _, C = cluster_confusions(y, pred, sid)
    W = bootstrap_weights(C.shape[0], 2000, seed=3)
    ci = cluster_bootstrap_ci(C, W)
    acc = accuracy_score(y, pred)
    assert ci["acc"][0] <= acc <= ci["acc"][1]
    assert ci == cluster_bootstrap_ci(C, bootstrap_weights(C.shape[0], 2000, seed=3))


def test_perfect_classifier_ci_is_degenerate():
    y, _, sid = _data()
    _, C = cluster_confusions(y, y, sid)
    ci = cluster_bootstrap_ci(C, bootstrap_weights(C.shape[0], 200, seed=0))
    assert ci["acc"] == (1.0, 1.0) and ci["f1"] == (1.0, 1.0)


def test_mcnemar_exact_known_value():
    # 10 vs 0 discordant pairs -> two-sided exact p = 2 * 0.5**10
    a = np.array([True] * 10 + [True] * 5)
    b = np.array([False] * 10 + [True] * 5)
    p, na, nb = mcnemar_exact(a, b)
    assert (na, nb) == (10, 0) and np.isclose(p, 2 * 0.5 ** 10)
    assert mcnemar_exact(a, a)[0] == 1.0


def test_holm_monotone_and_ge_raw():
    p = [0.001, 0.02, 0.04, 0.5]
    adj, rej = holm(p)
    assert all(a >= r for a, r in zip(adj, p)) and np.isclose(adj[0], 0.004) and not rej[3]


def test_grouped_folds_no_leakage_and_cover_everything():
    groups = np.repeat(np.arange(50), 5)
    folds = grouped_folds(groups, 5)
    held = np.concatenate([te for _, te in folds])
    assert sorted(held) == list(range(len(groups)))
    for tr, te in folds:
        assert not set(groups[tr]) & set(groups[te])


def test_inner_split_grouped_and_reproducible():
    groups = np.repeat(np.arange(40), 5)
    a, b = inner_split(groups), inner_split(groups)
    assert np.array_equal(a[0], b[0]) and not set(groups[a[0]]) & set(groups[a[1]])


def test_mean_std_uses_ddof1():
    m, s = mean_std([1, 2, 3])
    assert m == 2 and np.isclose(s, 1.0)


def test_integrity_helpers():
    assert normalise("Hello,  WORLD!\n") == "hello world"
    assert text_hash("A b") == text_hash("a   B.")
    df = pd.DataFrame({"split": ["a", "b", "b"], "source_id": ["x", "y", "x"]})
    assert split_overlaps(df, "split") == {"a/b": 1}
    assert replace_section("# T\n\n## A\nold\n\n## B\nkeep\n", "## A", "## A\nnew") == "# T\n\n## A\nnew\n\n## B\nkeep\n"
    assert replace_section("# T\n", "## A", "## A\nnew").endswith("## A\nnew\n")
