"""T1: re-fit TF-IDF+LR (val-chosen C, same flow as src.baselines) and save per-row predictions.

Saves runs/tfidf_{variant}_base.npz and runs/tfidf_char_{variant}_base.npz with the same keys as the
neural npz (test_y, test_pred, test_prob, test_source_id) plus val_* arrays, and a small .json.
C is chosen on VAL macro-F1 inside src.baselines.run_variant; TEST is only predicted afterwards.

python -m src.analysis.tfidf_preds [--variants output both] [--char 0 1]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from threadpoolctl import threadpool_limits

from src import baselines as B

EXPECTED_TEST_ACC = {("output", False): 75.75, ("both", False): 76.71,
                     ("output", True): 74.54, ("both", True): 74.42}

_orig_make = B.make_vectorizer


def _single_job_vectorizer(char=False, **kw):
    vec = _orig_make(char=char, **kw)
    if hasattr(vec, "n_jobs"):
        vec.n_jobs = 1  # no parallel processes (low-RAM machine)
    return vec


B.make_vectorizer = _single_job_vectorizer


def fit_and_save(splits, variant, char, out_dir="runs"):
    # BLAS threads = 1: src.baselines originally ran each LR fit in a loky worker (1 BLAS thread). Multi-thread
    # BLAS changes float32 summation order and lbfgs stops at a different point (val acc 75.38 vs 75.04 at C=10).
    with threadpool_limits(limits=1):
        res, art = B.run_variant(splits, variant, char=char, eval_test=True, n_jobs=1)
    vec, model = art["vectorizer"], art["model"]
    arrays = {}
    for name in ("val", "test"):
        df = splits[name]
        X = vec.transform(B.build_texts(df, variant, True))
        prob = model.predict_proba(X).astype(np.float32)
        arrays[f"{name}_y"] = df["label"].to_numpy().astype(np.int64)
        arrays[f"{name}_pred"] = model.classes_[prob.argmax(1)].astype(np.int64)
        arrays[f"{name}_prob"] = prob
        arrays[f"{name}_source_id"] = df["source_id"].to_numpy().astype(str)
    stem = f"tfidf_{'char_' if char else ''}{variant}_base"
    np.savez_compressed(Path(out_dir) / f"{stem}.npz", **arrays)
    summary = {"variant": variant, "char": char, "best_C": res["best_C"], "val_curve": res["val_curve"],
               "val_accuracy": res["tfidf_lr"]["val"]["accuracy"],
               "test_accuracy": res["tfidf_lr"]["test"]["test"]["accuracy"],
               "test_macro_f1": res["tfidf_lr"]["test"]["test"]["macro_f1"], "n_features": res["n_features"]}
    (Path(out_dir) / f"{stem}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return stem, summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/raid_main.parquet")
    ap.add_argument("--variants", nargs="+", default=["output", "both"])
    ap.add_argument("--char", nargs="+", type=int, default=[0, 1], help="0 = word only, 1 = word+char")
    ap.add_argument("--out-dir", default="runs")
    a = ap.parse_args(argv)
    df = pd.read_parquet(a.data)
    splits = {n: df[df["split"] == n].reset_index(drop=True) for n in ("train", "val", "test")}
    ok = True
    for char in a.char:
        for v in a.variants:
            stem, s = fit_and_save(splits, v, bool(char), a.out_dir)
            acc = round(100 * s["test_accuracy"], 2)
            exp = EXPECTED_TEST_ACC.get((v, bool(char)))
            flag = "MATCH" if exp == acc else f"MISMATCH (expected {exp})"
            ok &= exp == acc
            print(f"{stem}: C={s['best_C']} val_acc={100 * s['val_accuracy']:.2f} test_acc={acc} "
                  f"test_macroF1={100 * s['test_macro_f1']:.2f} {flag}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
