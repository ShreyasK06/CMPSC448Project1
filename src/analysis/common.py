"""Shared helpers for the T4-T7 analysis modules (canonical row order, section writers, CI helpers)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.integrity import replace_section
from src.analysis.stats import (bootstrap_weights, cluster_bootstrap_ci, cluster_confusions,
                                metrics_from_confusion)
from src.preprocess import tokenize

MAIN = "data/processed/raid_main.parquet"
RQ3 = "data/processed/raid_rq3.parquet"
CAP = 256  # neural / TF-IDF token cap for the output text


def canon_order(df):
    """Row order used by every runs/*.npz via stats.load_preds: lexsort by (source_id as str, label)."""
    return np.lexsort((df["label"].to_numpy(), df["source_id"].astype(str).to_numpy()))


def canon(df):
    return df.iloc[canon_order(df)].reset_index(drop=True)


def load_splits(path=MAIN):
    df = pd.read_parquet(path)
    return {n: df[df["split"] == n].reset_index(drop=True) for n in ("train", "val", "test")}


def n_tokens(texts):
    return np.array([len(tokenize(t)) for t in texts])


def ci_row(y, pred, sid, n_boot=2000, seed=0):
    """-> (acc, f1, acc_ci, f1_ci) with cluster bootstrap over source_ids (any row subset)."""
    ids, C = cluster_confusions(np.asarray(y), np.asarray(pred), np.asarray(sid).astype(str))
    W = bootstrap_weights(len(ids), n_boot, seed)
    acc, f1 = metrics_from_confusion(C.sum(0))
    ci = cluster_bootstrap_ci(C, W)
    return float(acc), float(f1), ci["acc"], ci["f1"]


def fmt_ci(point, ci):
    return f"{100 * point:.2f} [{100 * ci[0]:.2f}, {100 * ci[1]:.2f}]"


def write_section(header, body, analysis_md="reports/analysis.md", results_md=None, results_body=None):
    """Idempotent: replaces the section starting with `header` or appends it. results.md is overwritten by
    src/aggregate.py, so re-run the analysis modules after aggregate to re-append the results sections."""
    for p, b in ((analysis_md, body), (results_md, results_body)):
        if p is None or b is None:
            continue
        p = Path(p)
        old = p.read_text(encoding="utf-8") if p.exists() else "# Analysis\n\n"
        p.write_text(replace_section(old, header, b), encoding="utf-8")


def md_table(headers, rows):
    L = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    L += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return L


def length_bins(values, n_bins=10):
    """Equal-frequency bin index (0..n_bins-1) using quantile edges of `values`; duplicate edges collapsed."""
    v = np.asarray(values, float)
    edges = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return np.searchsorted(edges, v, side="right")


def length_matched_indices(labels, bins, seed=0):
    """Per bin, sample min-over-classes count of rows from every class (without replacement), so each class
    has an identical bin histogram. Returns sorted row indices."""
    labels, bins = np.asarray(labels), np.asarray(bins)
    rng = np.random.default_rng(seed)
    out = []
    classes = np.unique(labels)
    for b in np.unique(bins):
        pools = [np.flatnonzero((bins == b) & (labels == c)) for c in classes]
        m = min(len(p) for p in pools)
        for p in pools:
            out.append(rng.choice(p, size=m, replace=False))
    return np.sort(np.concatenate(out)) if out else np.array([], dtype=int)
