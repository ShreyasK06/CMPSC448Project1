"""Domain-holdout evaluation (RQ3): train on some text domains, test on a domain never seen in training.

Split rules (no leakage):
  train        = existing 'train' split ids of the seen domains (default: the 7 other domains)
  val          = existing 'val' split ids of the SAME seen domains (early stopping / C selection never sees the held-out domain)
  test_holdout = ALL rows of the held-out domain (every split: 400 ids / 2,000 rows)
Vocabulary / TF-IDF vocabulary are fit on the seen-domain train rows only. Hyper-parameters are reused, not re-tuned.

TF-IDF runner (CPU):  python -m src.domain_holdout [--domains news ...] [--no-truncate] [--train-domains a b c --domains poetry]
Neural runs use `python -m src.train --holdout-domain DOMAIN` (see GPU_PHASE6.md).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src import baselines as B

DATA = "data/processed/raid_main.parquet"
HOLDOUT = "test_holdout"


def holdout_splits(df, holdout, train_domains=None):
    """df: raid_main-style frame with columns split, domain, source_id. -> dict train / val / test_holdout."""
    doms = sorted(df["domain"].unique())
    if holdout not in doms:
        raise ValueError(f"unknown domain {holdout!r}; choose from {doms}")
    seen = sorted(set(train_domains)) if train_domains else [d for d in doms if d != holdout]
    if holdout in seen:
        raise ValueError("held-out domain must not be in train_domains")
    bad = [d for d in seen if d not in doms]
    if bad:
        raise ValueError(f"unknown train domains {bad}")
    out = {
        "train": df[(df["split"] == "train") & df["domain"].isin(seen)].reset_index(drop=True),
        "val": df[(df["split"] == "val") & df["domain"].isin(seen)].reset_index(drop=True),
        HOLDOUT: df[df["domain"] == holdout].reset_index(drop=True),
    }
    out["train"] = out["train"].assign(split="train")  # vocab guard checks split == 'train'
    seen_ids = set(out["train"]["source_id"].astype(str)) | set(out["val"]["source_id"].astype(str))
    if seen_ids & set(out[HOLDOUT]["source_id"].astype(str)):
        raise AssertionError("source_id overlap between seen-domain train/val and the held-out domain")
    return out


def run_tfidf(df, holdout, train_domains=None, truncate=True, out_dir="runs", tag=None):
    splits = holdout_splits(df, holdout, train_domains)
    with threadpool_limits(limits=1):
        res, art = B.run_variant(splits, "output", char=False, eval_test=True, truncate=truncate,
                                 test_splits=(HOLDOUT,), n_jobs=1)
    te = splits[HOLDOUT]
    X = art["vectorizer"].transform(B.build_texts(te, "output", truncate))
    prob = art["model"].predict_proba(X).astype(np.float32)
    k = "K256" if truncate else "Kfull"
    tag = tag or f"ho_{holdout}"
    stem = f"tfidf_output_{tag}_{k}"
    np.savez_compressed(Path(out_dir) / f"{stem}.npz",
                        test_holdout_y=te["label"].to_numpy().astype(np.int64),
                        test_holdout_pred=art["model"].classes_[prob.argmax(1)].astype(np.int64),
                        test_holdout_prob=prob, test_holdout_source_id=te["source_id"].astype(str).to_numpy(dtype=str))
    summ = {"holdout": holdout, "train_domains": sorted(splits["train"]["domain"].unique().tolist()), "truncate": truncate,
            "best_C": res["best_C"], "n": res["n"], "val": res["tfidf_lr"]["val"],
            "test_holdout": res["tfidf_lr"]["test"][HOLDOUT]}
    (Path(out_dir) / f"{stem}.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    return stem, summ


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--domains", nargs="+", default=None, help="held-out domains (default: all 8)")
    ap.add_argument("--train-domains", nargs="+", default=None, help="seen domains (default: all others); needs one --domains value")
    ap.add_argument("--tag", default=None, help="run tag (default ho_<domain>; use hog_<domain> for grouped runs)")
    ap.add_argument("--no-truncate", action="store_true")
    ap.add_argument("--out-dir", default="runs")
    a = ap.parse_args(argv)
    df = pd.read_parquet(a.data)
    doms = a.domains or sorted(df["domain"].unique())
    if a.train_domains and len(doms) != 1:
        ap.error("--train-domains needs exactly one held-out domain")
    for d in doms:
        stem, s = run_tfidf(df, d, a.train_domains, not a.no_truncate, a.out_dir, a.tag)
        t = s["test_holdout"]
        print(f"{stem}: C={s['best_C']} val_acc={100 * s['val']['accuracy']:.2f} "
              f"held-out acc={100 * t['accuracy']:.2f} macro-F1={100 * t['macro_f1']:.2f}", flush=True)


if __name__ == "__main__":
    main()
