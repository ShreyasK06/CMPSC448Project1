"""T10 (SECONDARY, beyond spec): TF-IDF + LR learning curve on the OUTPUT text.

Points: 25%, 50%, 100% of the 2,240 train source_ids of raid_main (nested seeded subsets, src.scaling.subsample_ids) and the
expanded train set of raid_scaled.parquet. Val and test are identical for every point (verified). For each point: word 1-2gram
TF-IDF fit on that point's train only, C chosen on VAL macro-F1 (src.baselines.C_GRID), TEST scored once at the chosen C.
Single BLAS thread, n_jobs=1. TEST CIs: cluster bootstrap over test source_ids (shared resamples); McNemar vs the 100% point.

python -m src.analysis.t10_scaling_tfidf      (idempotent; rewrites reports/baselines/scaling_tfidf.json and the results section)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src import baselines as B
from src.analysis.common import canon, ci_row
from src.analysis.stats import holm, mcnemar_exact
from src.phase2_spec import MAIN_DATA, SCALED_DATA
from src.scaling import subsample_ids

OUT_JSON = "reports/baselines/scaling_tfidf.json"
OUT_NPZ = "runs/tfidf_scaling_test_pred.npz"


def train_for_point(main_df, scaled_df, name):
    """name in {'25%','50%','100%','expanded'} -> train dataframe (rows) for that point."""
    if name == "expanded":
        return scaled_df[scaled_df["split"] == "train"].reset_index(drop=True)
    tr = main_df[main_df["split"] == "train"]
    f = float(name.rstrip("%")) / 100
    keep = set(subsample_ids(tr["source_id"].unique(), f))
    return tr[tr["source_id"].astype(str).isin(keep)].reset_index(drop=True)


def fit_point(tr, va, te):
    """-> dict with val-chosen C, val/test metrics, test predictions (canonical test order)."""
    ytr, yva, yte = tr["label"].to_numpy(), va["label"].to_numpy(), te["label"].to_numpy()
    vec = B.make_vectorizer()
    Xtr = vec.fit_transform(B.build_texts(tr, "output", True))
    Xva = vec.transform(B.build_texts(va, "output", True))
    Xte = vec.transform(B.build_texts(te, "output", True))
    curve, tests = {}, {}
    for C in B.C_GRID:
        m = B._fit_lr(Xtr, ytr, C)
        curve[C] = B._metrics(yva, m.predict(Xva))
        tests[C] = m.predict(Xte).astype(int)  # kept for all C, but only the val-chosen one is ever used/reported
        del m
    bc = B.C_GRID[int(np.argmax([curve[C]["macro_f1"] for C in B.C_GRID]))]
    return dict(C=bc, val=curve[bc], val_curve={str(k): v["macro_f1"] for k, v in curve.items()},
                n_features=int(Xtr.shape[1]), pred=tests[bc], test_y=yte)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", nargs="+", default=["25%", "50%", "100%", "expanded"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--no-section", action="store_true", help="do not (re)write the markdown sections")
    a = ap.parse_args(argv)
    main_df = pd.read_parquet(MAIN_DATA)
    scaled_df = pd.read_parquet(SCALED_DATA)
    va = main_df[main_df["split"] == "val"].reset_index(drop=True)
    te = canon(main_df[main_df["split"] == "test"].reset_index(drop=True))
    te_s = canon(scaled_df[scaled_df["split"] == "test"].reset_index(drop=True))
    va_s = scaled_df[scaled_df["split"] == "val"].reset_index(drop=True)
    assert te["id"].tolist() == te_s["id"].tolist() and sorted(va["id"]) == sorted(va_s["id"]), "val/test differ"
    sid = te["source_id"].astype(str).to_numpy()
    yte = te["label"].to_numpy()

    out, preds = {}, {}
    with threadpool_limits(limits=1):
        for nm in a.points:
            t0 = time.time()
            tr = train_for_point(main_df, scaled_df, nm)
            r = fit_point(tr, va, te)
            acc, f1, acc_ci, f1_ci = ci_row(yte, r["pred"], sid, n_boot=a.n_boot)
            out[nm] = dict(n_train_ids=int(tr["source_id"].nunique()), n_train_rows=len(tr), C=r["C"],
                           n_features=r["n_features"], val_acc=r["val"]["accuracy"], val_macro_f1=r["val"]["macro_f1"],
                           test_acc=acc, test_macro_f1=f1, test_acc_ci=list(acc_ci), test_macro_f1_ci=list(f1_ci),
                           val_curve_macro_f1=r["val_curve"])
            preds[nm] = r["pred"]
            print(f"{nm}: ids={out[nm]['n_train_ids']} C={r['C']} val_acc={100 * r['val']['accuracy']:.2f} "
                  f"val_f1={100 * r['val']['macro_f1']:.2f} test_acc={100 * acc:.2f} test_f1={100 * f1:.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    if "100%" in out:
        print(f"100% point reproduces base test acc 75.75: {round(100 * out['100%']['test_acc'], 2) == 75.75}")
    mc = {}
    if "100%" in preds:
        names = [n for n in preds if n != "100%"]
        ps = []
        for n in names:
            p, ao, bo = mcnemar_exact(preds["100%"] == yte, preds[n] == yte)
            mc[n] = dict(p=p, base_only=ao, other_only=bo)
            ps.append(p)
        if ps:
            adj, rej = holm(ps)
            for n, pa, rj in zip(names, adj, rej):
                mc[n].update(holm_p=float(pa), reject=bool(rj))
    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_JSON).write_text(json.dumps(dict(
        note=("TF-IDF word 1-2gram + LR on the output text; C chosen on VAL macro-F1; test scored once at the chosen C. "
              "Subsets of raid_main train are nested and seeded (seed 42); 'expanded' = raid_scaled train. "
              "Val/test identical for all points. CIs: cluster bootstrap over test source_ids."),
        points=out, mcnemar_vs_100pct=mc), indent=1), encoding="utf-8")
    np.savez_compressed(OUT_NPZ, **{k.replace("%", "pct"): v for k, v in preds.items()})
    if not a.no_section:
        from src.analysis import t8_report
        t8_report.write_all()


if __name__ == "__main__":
    main()
