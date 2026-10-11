"""RQ3 report: leave-one-domain-out generalization.

python -m src.analysis.t15_domain_holdout -> section '## Domain holdout (RQ3)' in reports/{analysis,results}.md

For each of the 8 text domains a model is trained on the other 7 (train ids) and early-stopped / C-selected on their val ids, then scored on
EVERY row of the held-out domain (400 ids, 2,000 rows). The in-domain control is the existing all-domains model scored on the held-out
domain's original TEST-split ids only (it saw the other ids in training); the holdout model is scored on exactly the same ids, so the
drop column is a paired comparison. Cluster bootstrap over source_ids (2,000 resamples).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.common import md_table, write_section
from src.analysis.stats import bootstrap_weights, load_preds
from src.analysis.t8_report import ensemble, load_group

HEADER = "## Domain holdout (RQ3)"
LABELS = ["chatgpt", "cohere-chat", "llama-chat", "mistral-chat", "mpt-chat"]
N_BOOT = 2000
HO = "test_holdout"
# (display name, kind, TF-IDF truncation key or neural control tag)
MODELS = [("TF-IDF (256-token cap)", "tfidf", "K256"), ("TF-IDF (no truncation)", "tfidf", "Kfull"),
          ("CNN (tuned, 2048-token cap)", "cnn", "ext_len2048"), ("BiLSTM (tuned, 512-token cap)", "bilstm", "ext_len512")]


def _cluster(correct, sid):
    ids, inv = np.unique(sid, return_inverse=True)
    return np.bincount(inv, weights=correct.astype(float)), np.bincount(inv).astype(float)


def _ci(c, n, W):
    b = (W @ c) / (W @ n)
    return np.percentile(b, [2.5, 97.5])


def _fmt(p, ci):
    return f"{100 * p:.1f} [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"


def _load_ho(path):
    """Like stats.load_preds(path, HO), tolerating source_id stored as an object array (first TF-IDF holdout files)."""
    z = np.load(path, allow_pickle=True)  # our own files
    y, pred, sid, prob = z[f"{HO}_y"], z[f"{HO}_pred"], z[f"{HO}_source_id"].astype(str), z[f"{HO}_prob"]
    o = np.lexsort((y, sid))
    return {"y": y[o], "pred": pred[o], "prob": prob[o], "sid": sid[o]}


def _tfidf_holdout(runs, dom, k, tag="ho"):
    p = Path(runs) / f"tfidf_output_{tag}_{dom}_{k}.npz"
    return _load_ho(p) if p.exists() else None


def _neural_holdout(runs, model, dom, tag="ho"):
    files = [Path(runs) / f"{model}_output_{tag}_{dom}_s{s}.npz" for s in range(5)]
    files = [f for f in files if f.exists()]
    if not files:
        return None, []
    ps = [load_preds(f, HO) for f in files]
    for q in ps[1:]:
        assert np.array_equal(q["sid"], ps[0]["sid"]) and np.array_equal(q["y"], ps[0]["y"])
    prob = np.mean([q["prob"] for q in ps], axis=0)
    ens = dict(y=ps[0]["y"], sid=ps[0]["sid"], prob=prob, pred=prob.argmax(1))
    return ens, [float((q["pred"] == q["y"]).mean()) for q in ps]


def _controls(runs):
    """Control predictions on the all-domains TEST split, display name -> dict(y, sid, pred)."""
    ref = load_preds(Path(runs) / "tfidf_output_base.npz")
    tf = np.load(Path(runs) / "tfidf_ablations_test_pred.npz")
    out = {"TF-IDF (256-token cap)": dict(y=ref["y"], sid=ref["sid"], pred=tf["K256"].astype(int)),
           "TF-IDF (no truncation)": dict(y=ref["y"], sid=ref["sid"], pred=tf["Kfull"].astype(int))}
    for name, kind, tag in MODELS[2:]:
        g = [x for x in load_group(runs, kind, "output", tag) if x["seed"] < 3]
        if len(g) == 3:
            e = ensemble(g)
            assert np.array_equal(e["sid"], ref["sid"])
            out[name] = e
    return out


def per_model(runs, df, kind, k, ctrl, tag="ho"):
    """-> list of per-domain result dicts for one model."""
    test_ids = {d: set(g.loc[g["split"] == "test", "source_id"].astype(str)) for d, g in df.groupby("domain")}
    rows = []
    for dom in sorted(df["domain"].unique()):
        if kind == "tfidf":
            ho, seeds = _tfidf_holdout(runs, dom, k, tag), []
        else:
            ho, seeds = _neural_holdout(runs, kind, dom, tag)
        if ho is None:
            continue
        sid = ho["sid"]
        W = bootstrap_weights(len(np.unique(sid)), N_BOOT, 0)
        corr = ho["pred"] == ho["y"]
        c, n = _cluster(corr, sid)
        row = dict(dom=dom, acc=float(corr.mean()), ci=_ci(c, n, W), seeds=seeds,
                   recall=[float(corr[ho["y"] == j].mean()) for j in range(5)],
                   top_share=float(np.bincount(ho["pred"], minlength=5).max() / len(ho["pred"])))
        if ctrl is not None:
            ids = list(test_ids[dom])
            m, mh = np.isin(ctrl["sid"], ids), np.isin(sid, ids)
            assert m.sum() == mh.sum() and np.array_equal(ctrl["sid"][m], sid[mh]) and np.array_equal(ctrl["y"][m], ho["y"][mh])
            cc, nn = _cluster(ctrl["pred"][m] == ctrl["y"][m], ctrl["sid"][m])
            ch, _ = _cluster(corr[mh], sid[mh])
            Wt = bootstrap_weights(len(nn), N_BOOT, 1)
            row.update(n_rows_ctrl=int(m.sum()), ctrl=float(cc.sum() / nn.sum()), ctrl_ci=_ci(cc, nn, Wt),
                       ho_same=float(ch.sum() / nn.sum()), drop=float((cc.sum() - ch.sum()) / nn.sum()),
                       drop_ci=_ci(cc - ch, nn, Wt))
        rows.append(row)
    return rows


def build(runs="runs", data="data/processed/raid_main.parquet"):
    df = pd.read_parquet(data)
    ctrls = _controls(runs)
    tables, classes = [], []
    for name, kind, k in MODELS:
        rows = per_model(runs, df, kind, k, ctrls.get(name))
        if not rows:
            continue
        T = []
        for r in rows:
            sd = f" (per seed {100 * np.mean(r['seeds']):.1f} ± {100 * np.std(r['seeds'], ddof=1):.1f})" if len(r["seeds"]) > 1 else ""
            has = "ctrl" in r
            T.append([r["dom"], _fmt(r["acc"], r["ci"]) + sd,
                      _fmt(r["ctrl"], r["ctrl_ci"]) if has else "-", f"{100 * r['ho_same']:.1f}" if has else "-",
                      f"{100 * r['drop']:+.1f} [{100 * r['drop_ci'][0]:+.1f}, {100 * r['drop_ci'][1]:+.1f}]" if has else "-",
                      r.get("n_rows_ctrl", "-"), f"{100 * r['top_share']:.0f}"])
        if len(rows) == 8 and all("ctrl" in r for r in rows):
            T.append(["**mean of 8 domains**", f"{100 * np.mean([r['acc'] for r in rows]):.1f}",
                      f"{100 * np.mean([r['ctrl'] for r in rows]):.1f}", f"{100 * np.mean([r['ho_same'] for r in rows]):.1f}",
                      f"{100 * np.mean([r['drop'] for r in rows]):+.1f}", "-", f"{100 * np.mean([r['top_share'] for r in rows]):.0f}"])
        tables.append((name, len(rows), T))
        rec = np.array([r["recall"] for r in rows])
        classes.append([name, len(rows)] + [f"{100 * rec[:, j].mean():.0f} ({100 * rec[:, j].min():.0f}-{100 * rec[:, j].max():.0f})"
                                           for j in range(5)])
    return tables, classes


def grouped(runs):
    """Optional literal example from the brief: tag hog_<domain> (src.domain_holdout --tag hog_<domain>, src.train --tag hog_<domain>)."""
    out = []
    for p in sorted(Path(runs).glob("tfidf_output_hog_*_K256.npz")):
        dom = p.stem.split("_")[3]
        for name, kind, k in MODELS:
            ho = _tfidf_holdout(runs, dom, k, "hog") if kind == "tfidf" else _neural_holdout(runs, kind, dom, "hog")[0]
            if ho is None:
                continue
            c, n = _cluster(ho["pred"] == ho["y"], ho["sid"])
            W = bootstrap_weights(len(n), N_BOOT, 0)
            out.append([dom, name, _fmt(float((ho["pred"] == ho["y"]).mean()), _ci(c, n, W)),
                        f"{100 * np.bincount(ho['pred'], minlength=5).max() / len(ho['pred']):.0f}"])
    return out


def body(tables, classes, grp):
    L = [HEADER, "",
         "Generated by `python -m src.analysis.t15_domain_holdout`. For each text domain a model is trained on the other 7 domains (their "
         "train ids), early-stopped or C-selected on those domains' val ids only, with the vocabulary built from the seen-domain train rows "
         "only and hyper-parameters reused (not re-tuned). It is then scored on every row of the held-out domain (400 source_ids, "
         "2,000 rows): column 2 (for the neural models the headline and CI are for the probability ensemble over the available seeds; "
         "the per-seed mean ± sd is in brackets). "
         "The in-domain control is the existing all-domains model (same hyper-parameters; neural = 3-seed ensemble) scored on the held-out "
         "domain's original TEST-split ids only, since it was trained on the other ids; the holdout model is scored on exactly those ids "
         "(column 4) and the drop is control minus holdout, paired, with a cluster-bootstrap 95% CI over source_ids (2,000 resamples). Those "
         "control sets are small (45-65 source_ids, column 6 = rows), so the drop CIs are wide; the full 400-id column is the steadier "
         "estimate. Chance = 20%. `top pred %` = share of the held-out predictions that fall in the single most-predicted class (20% = "
         "balanced, 100% = total collapse). One split per fold, so these are not cross-validated estimates of a new-domain deployment; "
         "McNemar p-values are not shown because the rows of a source_id are not independent.", ""]
    for name, nrows, T in tables:
        L += [f"### {name}" + ("" if nrows == 8 else f" ({nrows} of 8 domains available)"), ""]
        L += md_table(["held-out domain", "held-out acc % [95% CI], all 400 ids", "in-domain control acc % [CI] (all-domains model)",
                       "holdout model, same ids acc %", "drop pp [95% CI]", "control rows", "top pred %"], T)
        L.append("")
    if classes:
        L += ["### Per-class recall on the held-out domain (mean over folds, %; range over folds in brackets)", ""]
        L += md_table(["model", "folds"] + LABELS, classes)
        L.append("")
    if grp:
        L += ["### Grouped example from the brief", "", "Train on the listed domains only, test on the held-out one.", ""]
        L += md_table(["held-out domain", "model", "held-out acc % [95% CI]", "top pred %"], grp)
        L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--data", default="data/processed/raid_main.parquet")
    ap.add_argument("--analysis", default="reports/analysis.md")
    ap.add_argument("--results", default="reports/results.md")
    a = ap.parse_args(argv)
    tables, classes = build(a.runs, a.data)
    b = body(tables, classes, grouped(a.runs))
    write_section(HEADER, b, analysis_md=a.analysis, results_md=a.results, results_body=b)
    print(b)


if __name__ == "__main__":
    main()
