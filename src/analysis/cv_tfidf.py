"""T3: grouped 5-fold CV for TF-IDF + LR over train+val source_ids (TEST is never touched).

Outer: GroupKFold(5) by source_id on train+val rows (2,720 ids). Inside each outer fold, C is chosen on an inner
grouped split (GroupShuffleSplit 80/20 of the outer-train ids, macro-F1), the vectorizer is fit on the inner-train
rows for that selection, then vectorizer + LR are refit on the full outer-train rows at the chosen C and scored on
the held-out fold. Neural models are not cross-validated.

python -m src.analysis.cv_tfidf [--variants output both] [--folds 5]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from threadpoolctl import threadpool_limits

from src import baselines as B
from src.analysis.integrity import replace_section


def grouped_folds(groups, k=5):
    """List of (train_idx, held_idx) from GroupKFold; asserts no group appears on both sides."""
    groups = np.asarray(groups)
    out = []
    for tr, te in GroupKFold(n_splits=k).split(np.zeros(len(groups)), groups=groups):
        assert not (set(groups[tr]) & set(groups[te])), "group leakage across folds"
        out.append((tr, te))
    return out


def inner_split(groups, test_size=0.2, seed=42):
    groups = np.asarray(groups)
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(
        np.zeros(len(groups)), groups=groups))
    assert not (set(groups[tr]) & set(groups[va]))
    return tr, va


def mean_std(xs):
    xs = np.asarray(xs, float)
    return float(xs.mean()), (float(xs.std(ddof=1)) if len(xs) > 1 else 0.0)


def select_C(df_tr, df_va, variant, c_grid=B.C_GRID):
    vec = B.make_vectorizer()
    X = vec.fit_transform(B.build_texts(df_tr, variant))
    Xv = vec.transform(B.build_texts(df_va, variant))
    y, yv = df_tr["label"].to_numpy(), df_va["label"].to_numpy()
    scores = [B._metrics(yv, B._fit_lr(X, y, C).predict(Xv))["macro_f1"] for C in c_grid]
    return c_grid[int(np.argmax(scores))], scores


def cv_variant(df, variant, k=5):
    groups = df["source_id"].to_numpy()
    folds = []
    for i, (tr, te) in enumerate(grouped_folds(groups, k)):
        dtr, dte = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
        itr, iva = inner_split(dtr["source_id"].to_numpy())
        C, _ = select_C(dtr.iloc[itr].reset_index(drop=True), dtr.iloc[iva].reset_index(drop=True), variant)
        vec = B.make_vectorizer()
        X = vec.fit_transform(B.build_texts(dtr, variant))
        m = B._fit_lr(X, dtr["label"].to_numpy(), C)
        met = B._metrics(dte["label"].to_numpy(), m.predict(vec.transform(B.build_texts(dte, variant))))
        folds.append({"fold": i, "C": C, "n_train_ids": int(dtr.source_id.nunique()),
                      "n_held_ids": int(dte.source_id.nunique()), **met})
        print(f"  [{variant}] fold {i}: C={C} acc={100 * met['accuracy']:.2f} f1={100 * met['macro_f1']:.2f}", flush=True)
    return folds


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/raid_main.parquet")
    ap.add_argument("--variants", nargs="+", default=["output", "both"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out-json", default="reports/baselines/cv_tfidf.json")
    ap.add_argument("--results-md", default="reports/results.md")
    ap.add_argument("--analysis-md", default="reports/analysis.md")
    ap.add_argument("--fixed-json-dir", default="runs")
    a = ap.parse_args(argv)
    df = pd.read_parquet(a.data)
    dv = df[df["split"].isin(["train", "val"])].reset_index(drop=True)
    res = {}
    with threadpool_limits(limits=1):
        for v in a.variants:
            res[v] = cv_variant(dv, v, a.folds)
    Path(a.out_json).write_text(json.dumps(res, indent=1), encoding="utf-8")

    L = ["| model | variant | CV acc % [grouped 5-fold on train+val ids] | CV macro-F1 % [same] | chosen C per fold | "
         "val acc % [VAL split, fixed] | test acc % [TEST split, fixed] | test macro-F1 % [TEST split, fixed] |",
         "|---|---|---|---|---|---|---|---|"]
    for v, folds in res.items():
        ma, sa = mean_std([f["accuracy"] for f in folds])
        mf, sf = mean_std([f["macro_f1"] for f in folds])
        fx = json.loads((Path(a.fixed_json_dir) / f"tfidf_{v}_base.json").read_text())
        L.append(f"| tfidf_lr (word 1-2gram) | {v} | {100 * ma:.2f} ± {100 * sa:.2f} | {100 * mf:.2f} ± {100 * sf:.2f} | "
                 f"{', '.join(str(f['C']) for f in folds)} | {100 * fx['val_accuracy']:.2f} | "
                 f"{100 * fx['test_accuracy']:.2f} | {100 * fx['test_macro_f1']:.2f} |")
    n_ids = dv["source_id"].nunique()
    desc = (f"GroupKFold({a.folds}) by source_id over train+val ({n_ids} ids, {len(dv)} rows); mean ± sample std (ddof=1) over folds. "
            "Within each fold C is picked from the grid on an inner grouped 80/20 split of the outer-train ids (macro-F1), then the "
            "vectorizer and LR are refit on all outer-train rows; the held-out fold is only scored. The TEST split is not used by the CV. "
            "The fixed-split columns are the single train/val/test result (C chosen on VAL), shown for comparison. "
            "No neural CV was run.")
    sec = ["## TF-IDF grouped cross-validation", "", desc, ""] + L + [""]
    for p in (Path(a.results_md), Path(a.analysis_md)):
        old = p.read_text(encoding="utf-8") if p.exists() else "# Analysis\n\n"
        p.write_text(replace_section(old, "## TF-IDF grouped cross-validation", "\n".join(sec)), encoding="utf-8")
    print("\n".join(sec))


if __name__ == "__main__":
    main()
