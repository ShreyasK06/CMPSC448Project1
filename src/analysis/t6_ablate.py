"""T6: TF-IDF (word 1-2gram, output text) ablations. Train and test are processed identically per ablation.

Ablations: truncation to first K tokens (K in 32,64,128,256,full), lowercase, format-normalised, self-id masked.
C is chosen on VAL macro-F1 separately per ablation (grid src.baselines.C_GRID); TEST scored once per ablation.
K=256 is the base model (must reproduce 75.75 test acc). Ablations other than K use the base cap of 256 tokens.
Single BLAS thread, n_jobs=1.

python -m src.analysis.t6_ablate      (idempotent; re-run after src/aggregate.py to re-append to results.md)
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from src import baselines as B
from src.analysis.common import canon, ci_row, fmt_ci, load_splits, md_table, write_section
from src.analysis.stats import holm, load_preds, mcnemar_exact
from src.preprocess import tokenize

HDR = "## TF-IDF ablations (T6)"
SELF_ID = [r"\bcohere\b", r"\bas an ai(?: language model)?\b", r"^\s*sure[!,.]?\s+here(?:'s| is| are)?"]
_SELF_ID_RE = [re.compile(p, re.I | re.M) for p in SELF_ID]
_MD_LINE = re.compile(r"^[ \t]*(?:#{1,6}[ \t]+|>[ \t]?|[-*+•][ \t]+|\d+[.)][ \t]+)", re.M)
_MD_INLINE = re.compile(r"[*_`~]+")


def mask_self_id(text):
    """Delete model self-identification phrases (Cohere, 'as an AI [language model]', 'Sure! Here ...')."""
    t = text if isinstance(text, str) else ""
    for r in _SELF_ID_RE:
        t = r.sub(" ", t)
    return t


def format_normalise(text):
    """Strip markdown (heading/quote/bullet/numbered-list markers, emphasis chars), turn every newline into a space and
    collapse whitespace, so no <NL> token and no list/markdown structure survives."""
    t = text if isinstance(text, str) else ""
    t = _MD_LINE.sub("", t)
    t = _MD_INLINE.sub("", t)
    return " ".join(t.split())


def make_text_fn(name):
    """name -> function(raw text) -> space-joined token string."""
    if name.startswith("K"):
        k = None if name == "Kfull" else int(name[1:])
        return lambda t: " ".join(tokenize(t)[:k])
    if name == "lowercase":
        return lambda t: " ".join(tokenize(t, lowercase=True)[:256])
    if name == "format_norm":
        return lambda t: " ".join(tokenize(format_normalise(t))[:256])
    if name == "self_id_masked":
        return lambda t: " ".join(tokenize(mask_self_id(t))[:256])
    raise ValueError(name)


ABLATIONS = ["K32", "K64", "K128", "K256", "Kfull", "lowercase", "format_norm", "self_id_masked"]


def run_one(name, tr, va, te):
    f = make_text_fn(name)
    ytr, yva = tr["label"].to_numpy(), va["label"].to_numpy()
    vec = B.make_vectorizer()
    Xtr = vec.fit_transform([f(t) for t in tr["generation"]])
    Xva = vec.transform([f(t) for t in va["generation"]])
    Xte = vec.transform([f(t) for t in te["generation"]])
    curve, models = {}, []
    for C in B.C_GRID:
        m = B._fit_lr(Xtr, ytr, C)
        models.append(m)
        curve[C] = B._metrics(yva, m.predict(Xva))
    bi = int(np.argmax([curve[C]["macro_f1"] for C in B.C_GRID]))
    m = models[bi]
    return {"name": name, "C": B.C_GRID[bi], "val": curve[B.C_GRID[bi]], "val_curve": {str(k): v for k, v in curve.items()},
            "n_features": int(Xtr.shape[1]), "pred": m.predict(Xte).astype(int)}


def main():
    sp = load_splits()
    tr, va, te = sp["train"], sp["val"], canon(sp["test"])
    ref = load_preds("runs/tfidf_output_base.npz")
    assert np.array_equal(te["source_id"].astype(str).to_numpy(), ref["sid"])
    yte, sid = te["label"].to_numpy(), te["source_id"].astype(str).to_numpy()
    out = {}
    with threadpool_limits(limits=1):
        for nm in ABLATIONS:
            t0 = time.time()
            out[nm] = run_one(nm, tr, va, te)
            r = out[nm]
            print(f"{nm}: C={r['C']} val_acc={100 * r['val']['accuracy']:.2f} test_acc={100 * (r['pred'] == yte).mean():.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    base = out["K256"]
    print(f"K256 preds identical to saved base preds: {np.array_equal(base['pred'], ref['pred'])}")
    Path("reports/baselines/ablations.json").write_text(
        json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "pred"} for k, v in out.items()}, indent=1), encoding="utf-8")
    np.savez_compressed("runs/tfidf_ablations_test_pred.npz", **{k: v["pred"] for k, v in out.items()})

    tests = [n for n in ABLATIONS if n != "K256"]
    pvals, cnt = [], {}
    for n in tests:
        p, a_only, b_only = mcnemar_exact(base["pred"] == yte, out[n]["pred"] == yte)
        pvals.append(p)
        cnt[n] = (a_only, b_only)
    adj, rej = holm(pvals)
    rows = []
    for n in ABLATIONS:
        r = out[n]
        a, f1, ac, fc = ci_row(yte, r["pred"], sid)
        if n == "K256":
            mc = "(base)"
        else:
            i = tests.index(n)
            mc = f"{cnt[n][0]}/{cnt[n][1]}; p={pvals[i]:.3g}; Holm p={adj[i]:.3g}{' *' if rej[i] else ''}"
        rows.append([n, r["C"], f"{100 * r['val']['accuracy']:.2f} / {100 * r['val']['macro_f1']:.2f}", fmt_ci(a, ac),
                     fmt_ci(f1, fc), mc, r["n_features"]])
    tab = md_table(["ablation", "C (VAL)", "VAL acc / macro-F1 %", "TEST acc % [95% CI]", "TEST macro-F1 % [95% CI]",
                    "McNemar vs base K256 (base-only/ablation-only correct; exact p; Holm over 7)", "n features"], rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ks = [("K32", 32), ("K64", 64), ("K128", 128), ("K256", 256), ("Kfull", 512)]
    xs, ys, lo, hi = [], [], [], []
    for n, x in Ks:
        a, _, ac, _ = ci_row(yte, out[n]["pred"], sid)
        xs.append(x)
        ys.append(100 * a)
        lo.append(100 * (a - ac[0]))
        hi.append(100 * (ac[1] - a))
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(["32", "64", "128", "256", "full"])
    ax.set_xlabel("output tokens kept (first K)")
    ax.set_ylabel("TEST accuracy % (95% cluster CI)")
    ax.set_title("TF-IDF (word, output): accuracy vs truncation")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig("reports/figures/tfidf_acc_vs_K.png", dpi=150)

    L = [HDR, "",
         "Generated by `python -m src.analysis.t6_ablate`. TF-IDF word 1-2gram + LR on the OUTPUT text; train, val and test are "
         "processed identically in every row; C chosen on VAL macro-F1 per ablation (grid 0.01..1000); vectorizer fit on train only. "
         "K = keep first K tokens (tokeniser tokens, <NL> counts as a token); `K256` is the base model (predictions identical to "
         "runs/tfidf_output_base.npz), `Kfull` = no truncation. lowercase / format_norm / self_id_masked use the base 256-token cap. "
         "format_norm = strip markdown headings/quotes/bullets/numbered-list markers and `*_~` emphasis, newlines to spaces, collapse "
         "whitespace (no <NL>). self_id_masked = delete matches of `" + "`, `".join(SELF_ID) + "` (case-insensitive). Split for all "
         "metrics: [TEST split: test], CIs by cluster bootstrap over source_ids; `*` = Holm-significant at 0.05. "
         "Figure: reports/figures/tfidf_acc_vs_K.png (`full` plotted at x=512).", "", *tab, ""]
    body = "\n".join(L)
    write_section(HDR, body, results_md="reports/results.md", results_body=body)
    print(body)


if __name__ == "__main__":
    main()
