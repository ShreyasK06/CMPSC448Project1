"""T2: cluster-bootstrap CIs and exact McNemar tests on the TEST split (identical rows for all models).

Clusters are source_ids (each id contributes its 5 generations, one per model class). Everything here is
computed from saved predictions (runs/*.npz); nothing is retrained and no selection is done on TEST.

python -m src.analysis.stats [--n-boot 10000] [--seed 0]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.analysis.integrity import replace_section

K = 5
SPLIT_LABEL = "TEST split: test"


# ----------------------------------------------------------------- pure helpers
def load_preds(path, split="test"):
    """Load an npz in the neural format -> dict(y, pred, prob, sid), rows sorted by (source_id, y)."""
    z = np.load(path, allow_pickle=False)
    y, pred, sid = z[f"{split}_y"], z[f"{split}_pred"], z[f"{split}_source_id"].astype(str)
    prob = z[f"{split}_prob"]
    order = np.lexsort((y, sid))
    return {"y": y[order], "pred": pred[order], "prob": prob[order], "sid": sid[order]}


def cluster_confusions(y, pred, sid, k=K):
    """Per-cluster confusion counts. Returns (ids, C) with C of shape (n_clusters, k, k) [true, pred]."""
    ids, inv = np.unique(sid, return_inverse=True)
    C = np.zeros((len(ids), k, k), dtype=np.int64)
    np.add.at(C, (inv, y, pred), 1)
    return ids, C


def metrics_from_confusion(conf):
    """conf: (..., k, k) counts. Returns (accuracy, macro_f1) with zero_division=0 (sklearn convention)."""
    conf = np.asarray(conf, dtype=np.float64)
    tp = np.diagonal(conf, axis1=-2, axis2=-1)
    n = conf.sum(axis=(-2, -1))
    fp = conf.sum(axis=-2) - tp
    fn = conf.sum(axis=-1) - tp
    den = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, den, out=np.zeros_like(den), where=den > 0)
    return tp.sum(-1) / n, f1.mean(-1)


def bootstrap_weights(n_clusters, n_boot, seed):
    """(n_boot, n_clusters) multiplicity matrix: each row = bincount of n_clusters draws with replacement."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    W = np.zeros((n_boot, n_clusters), dtype=np.float64)
    np.add.at(W, (np.arange(n_boot)[:, None], draws), 1.0)
    return W


def cluster_bootstrap_ci(C, W, alpha=0.05):
    """Percentile CIs for (acc, macro-F1) from per-cluster confusions C and bootstrap weights W."""
    conf = np.tensordot(W, C.astype(np.float64), axes=(1, 0))  # (n_boot, k, k)
    acc, f1 = metrics_from_confusion(conf)
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {"acc": tuple(np.percentile(acc, q)), "f1": tuple(np.percentile(f1, q))}


def mcnemar_table(correct_a, correct_b):
    a = np.asarray(correct_a, bool)
    b = np.asarray(correct_b, bool)
    return [[int((a & b).sum()), int((a & ~b).sum())], [int((~a & b).sum()), int((~a & ~b).sum())]]


def mcnemar_exact(correct_a, correct_b):
    """Exact (binomial) McNemar p-value; returns (p, n_a_only, n_b_only)."""
    from statsmodels.stats.contingency_tables import mcnemar
    t = mcnemar_table(correct_a, correct_b)
    return float(mcnemar(t, exact=True).pvalue), t[0][1], t[1][0]


def holm(pvals, alpha=0.05):
    from statsmodels.stats.multitest import multipletests
    rej, adj, _, _ = multipletests(pvals, alpha=alpha, method="holm")
    return adj, rej


# ----------------------------------------------------------------- driver
def _check_aligned(ref, other, name):
    if not (np.array_equal(ref["sid"], other["sid"]) and np.array_equal(ref["y"], other["y"])):
        raise ValueError(f"{name}: rows not aligned with reference test rows")


def collect(runs="runs"):
    """-> dict[(family, variant)] -> {'runs': {label: preds}, 'ens': preds (neural only)}."""
    runs = Path(runs)
    out = {}
    for v in ("output", "both"):
        out[("tfidf", v)] = {"runs": {"": load_preds(runs / f"tfidf_{v}_base.npz")}}
        out[("tfidf+char", v)] = {"runs": {"": load_preds(runs / f"tfidf_char_{v}_base.npz")}}
        for m in ("cnn", "bilstm"):
            out[(m, v)] = {"runs": {f"s{s}": load_preds(runs / f"{m}_{v}_base_s{s}.npz") for s in range(3)}}
    ref = out[("tfidf", "output")]["runs"][""]
    for key, d in out.items():
        for rk, r in d["runs"].items():
            _check_aligned(ref, r, f"{key}{rk}")
        if len(d["runs"]) > 1:
            prob = np.mean([r["prob"] for r in d["runs"].values()], axis=0)
            d["ens"] = {**ref, "prob": prob, "pred": prob.argmax(1)}
    return out


def _fmt_ci(point, ci):
    return f"{100 * point:.2f} [{100 * ci[0]:.2f}, {100 * ci[1]:.2f}]"


def _correct(r):
    return r["pred"] == r["y"]


def _primary(data, fam, v):
    d = data[(fam, v)]
    return d["ens"] if "ens" in d else d["runs"][""]


def _seed_run(data, fam, v, s):
    d = data[(fam, v)]["runs"]
    return d[f"s{s}"] if f"s{s}" in d else d[""]


def run(runs="runs", n_boot=10000, seed=0, results_md="reports/results.md", analysis_md="reports/analysis.md"):
    data = collect(runs)
    ref = data[("tfidf", "output")]["runs"][""]
    ids, _ = cluster_confusions(ref["y"], ref["pred"], ref["sid"])
    W = bootstrap_weights(len(ids), n_boot, seed)

    rows = []  # (family, variant, label, acc, f1, ci)
    for (fam, v), d in data.items():
        items = list(d["runs"].items()) + ([("3-seed mean-prob ensemble", d["ens"])] if "ens" in d else [])
        for lab, r in items:
            _, C = cluster_confusions(r["y"], r["pred"], r["sid"])
            acc, f1 = metrics_from_confusion(C.sum(0))
            rows.append((fam, v, lab or "single fit", float(acc), float(f1), cluster_bootstrap_ci(C, W)))

    tests = []  # (name, p, a_only, b_only, per-seed p-range or None)
    for v in ("output", "both"):
        for a, b in (("tfidf", "cnn"), ("tfidf", "bilstm"), ("cnn", "bilstm")):
            p, na, nb = mcnemar_exact(_correct(_primary(data, a, v)), _correct(_primary(data, b, v)))
            ps = [mcnemar_exact(_correct(_seed_run(data, a, v, s)), _correct(_seed_run(data, b, v, s)))[0] for s in range(3)]
            tests.append((f"{a} vs {b} [{v}]", p, na, nb, (min(ps), max(ps))))
    for m in ("tfidf", "cnn", "bilstm"):
        p, na, nb = mcnemar_exact(_correct(_primary(data, m, "output")), _correct(_primary(data, m, "both")))
        rng_ = None
        if m != "tfidf":
            ps = [mcnemar_exact(_correct(_seed_run(data, m, "output", s)), _correct(_seed_run(data, m, "both", s)))[0]
                  for s in range(3)]
            rng_ = (min(ps), max(ps))
        tests.append((f"{m}: output vs both", p, na, nb, rng_))
    adj, rej = holm([t[1] for t in tests])

    def ci_table():
        L = ["| model | variant | fit | test acc % [95% CI] | test macro-F1 % [95% CI] |", "|---|---|---|---|---|"]
        for fam, v, lab, acc, f1, ci in rows:
            L.append(f"| {fam} | {v} | {lab} | {_fmt_ci(acc, ci['acc'])} | {_fmt_ci(f1, ci['f1'])} |")
        return L

    def mc_table(detail):
        h = "| comparison (A vs B) | A-only correct | B-only correct | exact McNemar p | Holm-adj p | reject @0.05 |"
        if detail:
            h += " per-seed p range |"
        L = [h, "|---|---|---|---|---|---|" + ("---|" if detail else "")]
        for t, pa, r in zip(tests, adj, rej):
            line = f"| {t[0]} | {t[2]} | {t[3]} | {t[1]:.3g} | {pa:.3g} | {'yes' if r else 'no'} |"
            if detail:
                line += f" {'-' if t[4] is None else f'{t[4][0]:.3g} to {t[4][1]:.3g}'} |"
            L.append(line)
        return L

    note = (f"Cluster bootstrap: resample the {len(ids)} test source_ids with replacement ({n_boot} resamples, seed {seed}; "
            "all models share the same resamples), keep all 5 rows of each drawn id, percentile 95% CI. "
            f"All numbers are on the [{SPLIT_LABEL}] ({len(ref['y'])} rows) split.")
    sd = ("Neural per-seed rows differ only by training seed; the +/- reported elsewhere in this file for neural models is "
          "seed-only variance (n=3, ddof=1). TF-IDF is one deterministic fit (no seed), so its uncertainty here is the "
          "bootstrap CI over test source_ids only. The two are different quantities; the bootstrap CI for a neural row is "
          "test-sampling uncertainty conditional on that seed/ensemble.")

    res_sec = ["## Statistical tests on TEST split (cluster bootstrap CIs, exact McNemar)", "", note, "", sd, ""] + ci_table() + [
        "", "McNemar tests (exact binomial) on identical test rows, [TEST split: test]; neural = 3-seed mean-prob ensemble "
        "prediction; Holm correction across all 9 tests. Rows within a source_id are not independent, so these p-values are "
        "somewhat optimistic; details in reports/analysis.md.", ""] + mc_table(False) + [""]
    an_sec = ["## Statistics detail (T2)", "", "Generated by `python -m src.analysis.stats`. " + note, "", sd, "",
              "### Bootstrap CIs", ""] + ci_table() + [
        "", "### Exact McNemar (A vs B) with Holm correction", "",
        "Primary p = TF-IDF single fit / neural 3-seed mean-prob ensemble. The per-seed column pairs seed s of A with seed s of B "
        "(TF-IDF is the single fit) for s = 0,1,2 and gives min to max p. `A-only correct` = rows A gets right and B wrong.", ""
        ] + mc_table(True) + [""]

    for path, hdr, sec in ((Path(results_md), "## Statistical tests on TEST split", res_sec),
                           (Path(analysis_md), "## Statistics detail (T2)", an_sec)):
        old = path.read_text(encoding="utf-8") if path.exists() else "# Analysis\n\n"
        path.write_text(replace_section(old, hdr, "\n".join(sec)), encoding="utf-8")
    print("\n".join(an_sec))
    return rows, tests, adj


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    run(a.runs, a.n_boot, a.seed)


if __name__ == "__main__":
    main()
