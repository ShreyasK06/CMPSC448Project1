"""Phase 3 baselines: majority class and TF-IDF + LogisticRegression, per RQ2 input variant.

Vectorizer is fit on TRAIN only. C is tuned on VAL macro-F1. The test split(s) are only
evaluated when ``eval_test`` / ``--eval-test`` is set.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import FeatureUnion

from src.preprocess import tokenize

VARIANTS = ("prompt", "output", "both")
C_GRID = (0.01, 0.1, 1, 10, 100, 1000)  # 1000 added: best C=100 sat at the old grid edge
LABEL_NAMES = ["chatgpt", "cohere-chat", "llama-chat", "mistral-chat", "mpt-chat"]
SEED = 42


# ---------------------------------------------------------------- text construction
def _prep(text, max_tokens):
    toks = tokenize(text if isinstance(text, str) else "")
    if max_tokens is not None:
        toks = toks[:max_tokens]
    return " ".join(toks)


def build_texts(df, variant, truncate=True, max_prompt=64, max_output=256):
    """Return list[str] of space-joined tokens for the RQ2 variant ('<NL>' stays a token)."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    mp, mo = (max_prompt, max_output) if truncate else (None, None)
    out = []
    for pr, gen in zip(df["prompt"], df["generation"]):
        if variant == "prompt":
            out.append(_prep(pr, mp))
        elif variant == "output":
            out.append(_prep(gen, mo))
        else:
            out.append(_prep(pr, mp) + " <SEP> " + _prep(gen, mo))
    return out


# ---------------------------------------------------------------- model pieces
def make_vectorizer(char=False, max_features=200_000, min_df=2):
    """Word 1-2 gram TF-IDF over pre-tokenized text (split on whitespace, case kept, so
    punctuation and <NL> tokens are features). Optionally union with char_wb 2-5 grams."""
    word = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False,
                           ngram_range=(1, 2), sublinear_tf=True, min_df=min_df,
                           max_features=max_features, dtype=np.float32)
    if not char:
        return word
    ch = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=False,
                         sublinear_tf=True, min_df=min_df, max_features=max_features,
                         dtype=np.float32)
    return FeatureUnion([("word", word), ("char", ch)], n_jobs=2)


def _fit_lr(X, y, C):
    m = LogisticRegression(C=C, max_iter=2000, solver="lbfgs", random_state=SEED)
    return m.fit(X, y)


def top_features(model, vectorizer, k=20):
    """Top-k highest-weighted n-grams per class: {class_name: [(feature, weight), ...]}."""
    names = np.asarray(vectorizer.get_feature_names_out())
    out = {}
    for i, c in enumerate(model.classes_):
        w = model.coef_[i]
        idx = np.argsort(-w)[:k]
        out[LABEL_NAMES[int(c)]] = [(str(names[j]), float(w[j])) for j in idx]
    return out


# ---------------------------------------------------------------- metrics
def _metrics(y, p):
    return {"accuracy": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro", labels=list(range(5)), zero_division=0))}


def _per_domain_acc(df, y, p):
    d = pd.DataFrame({"domain": df["domain"].values, "ok": (np.asarray(y) == np.asarray(p))})
    return {str(k): float(v) for k, v in d.groupby("domain")["ok"].mean().items()}


def _report(df, y, p, extra=False):
    r = _metrics(y, p)
    if extra:
        r["confusion_matrix"] = confusion_matrix(y, p, labels=list(range(5))).tolist()
        r["per_domain_accuracy"] = _per_domain_acc(df, y, p)
    return r


# ---------------------------------------------------------------- core
def run_variant(splits, variant, char=False, eval_test=False, truncate=True, test_splits=("test",),
                c_grid=C_GRID, n_jobs=-1, top_k=20):
    """splits: dict name -> df; must contain 'train' and 'val' (+ test_splits if eval_test).
    Returns (result_dict, artifacts_dict with vectorizer/model)."""
    t0 = time.time()
    tr, va = splits["train"], splits["val"]
    ytr, yva = tr["label"].to_numpy(), va["label"].to_numpy()

    vec = make_vectorizer(char=char)
    Xtr = vec.fit_transform(build_texts(tr, variant, truncate))  # TRAIN only
    Xva = vec.transform(build_texts(va, variant, truncate))

    # majority baseline
    maj = Counter(ytr.tolist()).most_common(1)[0][0]
    majority = {"label": int(maj), "val": _report(va, yva, np.full(len(yva), maj))}

    models = Parallel(n_jobs=n_jobs)(delayed(_fit_lr)(Xtr, ytr, C) for C in c_grid)
    curve = {}
    for C, m in zip(c_grid, models):
        curve[str(C)] = _metrics(yva, m.predict(Xva))
    best_i = int(np.argmax([curve[str(C)]["macro_f1"] for C in c_grid]))  # first max on ties
    bestC, model = c_grid[best_i], models[best_i]  # fit on train only with best C (val not merged)

    res = {
        "variant": variant, "char": char, "truncate": truncate, "eval_test": eval_test,
        "n": {k: int(len(v)) for k, v in splits.items() if k in ("train", "val") or eval_test},
        "n_features": int(Xtr.shape[1]), "best_C": bestC, "val_curve": curve,
        "majority": majority,
        "tfidf_lr": {"train": _report(tr, ytr, model.predict(Xtr)),
                     "val": _report(va, yva, model.predict(Xva), extra=True)},
        "top_features": top_features(model, vec, top_k),
    }
    if eval_test:
        majority["test"], res["tfidf_lr"]["test"] = {}, {}
        for name in test_splits:  # evaluated exactly once
            te = splits[name]
            yte = te["label"].to_numpy()
            majority["test"][name] = _metrics(yte, np.full(len(yte), maj))
            res["tfidf_lr"]["test"][name] = _report(te, yte, model.predict(
                vec.transform(build_texts(te, variant, truncate))), extra=True)
    res["seconds"] = round(time.time() - t0, 1)
    return res, {"vectorizer": vec, "model": model}


def tag_for(variant, char, rq3, truncate=True):
    return variant + ("_char" if char else "") + ("_rq3" if rq3 else "") + ("" if truncate else "_full")


def run(data, variants=VARIANTS, char=False, eval_test=False, rq3=False, truncate=True,
        out_dir="reports/baselines", n_jobs=-1):
    df = pd.read_parquet(data)
    col = "split_rq3" if rq3 else "split"
    names = ["train", "val"] + (["test_indomain", "test_code"] if rq3 else ["test"])
    splits = {n: df[df[col] == n].reset_index(drop=True) for n in names}
    test_splits = tuple(n for n in names if n.startswith("test"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for v in variants:
        res, _ = run_variant(splits, v, char, eval_test, truncate, test_splits, n_jobs=n_jobs)
        res["rq3"] = rq3
        tag = tag_for(v, char, rq3, truncate)
        (out / f"{tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        results[tag] = res
    print_table(results)
    return results


def print_table(results):
    rows = []
    for tag, r in results.items():
        def row(model, split, m):
            rows.append((tag, model, split, m["accuracy"], m["macro_f1"], r["best_C"] if model == "tfidf_lr" else ""))
        row("majority", "val", r["majority"]["val"])
        for s in ("train", "val"):
            row("tfidf_lr", s, r["tfidf_lr"][s])
        for s, m in r["majority"].get("test", {}).items():
            row("majority", s, m)
        for s, m in r["tfidf_lr"].get("test", {}).items():
            row("tfidf_lr", s, m)
    print(f"{'tag':<18}{'model':<10}{'split':<14}{'acc':>8}{'macroF1':>9}{'bestC':>8}")
    for tag, model, split, a, f, c in rows:
        print(f"{tag:<18}{model:<10}{split:<14}{a:>8.4f}{f:>9.4f}{str(c):>8}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/processed/raid_main.parquet")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--char", action="store_true", help="add char_wb 2-5 gram TF-IDF")
    ap.add_argument("--eval-test", action="store_true", help="also evaluate test split(s), once")
    ap.add_argument("--rq3", action="store_true", help="use split_rq3 (test_indomain / test_code)")
    ap.add_argument("--no-truncate", action="store_true", help="use full text instead of 64/256 tokens")
    ap.add_argument("--out-dir", default="reports/baselines")
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args(argv)
    data = a.data
    if a.rq3 and a.data == "data/processed/raid_main.parquet":
        data = "data/processed/raid_rq3.parquet"
    return run(data, a.variants, a.char, a.eval_test, a.rq3, not a.no_truncate, a.out_dir, a.n_jobs)


if __name__ == "__main__":
    main()
