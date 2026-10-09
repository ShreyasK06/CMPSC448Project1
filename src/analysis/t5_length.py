"""T5: length-only and shallow-stylometry baselines, length-quintile accuracy, length-matched TEST subset.

python -m src.analysis.t5_length      (idempotent; re-run after src/aggregate.py to re-append to results.md)
Hyperparameters (C, HGB depth/lr/iters) are chosen on VAL macro-F1 only; TEST is scored once.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.analysis.common import (CAP, canon, ci_row, fmt_ci, length_bins, length_matched_indices, load_splits,
                                 md_table, n_tokens, write_section)
from src.analysis.stats import load_preds

HDR = "## Length controls and shallow baselines (T5)"
LEN_FEATS = ["word_count", "char_count", "hit_cap", "line_count"]
STYLO_FEATS = LEN_FEATS + [
    "mean_word_len", "type_token_ratio", "punct_ratio", "digit_ratio", "upper_ratio", "newline_count",
    "blank_line_count", "bullet_count", "markdown_count", "ends_terminal_punct", "starts_boilerplate",
    "comma_ratio", "mean_sentence_len", "mean_line_len", "exclaim_count", "colon_count", "quote_ratio", "starts_upper"]
_BOIL = re.compile(r"^\s*(sure|certainly|here|of course|okay|ok|title|note)\b", re.I)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.M)
_MD = re.compile(r"\*\*|__|^#{1,6}\s|`|^>\s", re.M)
_WORD = re.compile(r"\w+(?:['’]\w+)*")


def text_features(text, ntok):
    """Feature dict for one output text (pure function; raw text, not tokenised)."""
    t = text if isinstance(text, str) else ""
    words = _WORD.findall(t)
    nw, nc = len(words), len(t)
    lines = [l for l in t.split("\n") if l.strip()]
    sents = [s for s in re.split(r"[.!?]+\s", t) if s.strip()]
    nchar = max(nc, 1)
    return {
        "word_count": nw, "char_count": nc, "hit_cap": float(ntok >= CAP), "line_count": len(lines),
        "mean_word_len": float(np.mean([len(w) for w in words])) if nw else 0.0,
        "type_token_ratio": len({w.lower() for w in words}) / nw if nw else 0.0,
        "punct_ratio": sum(1 for c in t if not c.isalnum() and not c.isspace()) / nchar,
        "digit_ratio": sum(c.isdigit() for c in t) / nchar,
        "upper_ratio": sum(c.isupper() for c in t) / nchar,
        "newline_count": t.count("\n"), "blank_line_count": t.count("\n\n"),
        "bullet_count": len(_BULLET.findall(t)), "markdown_count": len(_MD.findall(t)),
        "ends_terminal_punct": float(t.rstrip()[-1:] in {".", "!", "?", '"', "'", ")"}),
        "starts_boilerplate": float(bool(_BOIL.match(t))),
        "comma_ratio": t.count(",") / nchar, "mean_sentence_len": nw / max(len(sents), 1),
        "mean_line_len": nw / max(len(lines), 1), "exclaim_count": t.count("!"), "colon_count": t.count(":"),
        "quote_ratio": (t.count('"') + t.count("“") + t.count("”")) / nchar,
        "starts_upper": float(t.lstrip()[:1].isupper()),
    }


def feature_matrix(df, feats):
    ntok = n_tokens(df["generation"])
    rows = [text_features(g, k) for g, k in zip(df["generation"], ntok)]
    X = pd.DataFrame(rows)[feats].to_numpy(float)
    # stored raw lengths in the parquet are authoritative for the length columns
    if "word_count" in feats:
        X[:, feats.index("word_count")] = df["word_len"].to_numpy()
    if "char_count" in feats:
        X[:, feats.index("char_count")] = df["char_len"].to_numpy()
    return X, ntok


def _mf1(y, p):
    return f1_score(y, p, average="macro", labels=list(range(5)), zero_division=0)


def fit_lr(Xtr, ytr, Xva, yva, Cs=(0.001, 0.01, 0.1, 1, 10, 100)):
    sc = StandardScaler().fit(Xtr)
    best = None
    for C in Cs:
        m = LogisticRegression(C=C, max_iter=5000, random_state=42).fit(sc.transform(Xtr), ytr)
        s = _mf1(yva, m.predict(sc.transform(Xva)))
        if best is None or s > best[0]:
            best = (s, C, m)
    s, C, m = best
    return (lambda X: m.predict_proba(sc.transform(X))), {"C": C, "val_f1": s}


def fit_hgb(Xtr, ytr, Xva, yva):
    best = None
    for depth in (3, None):
        for lr in (0.05, 0.1):
            for it in (100, 300):
                m = HistGradientBoostingClassifier(max_depth=depth, learning_rate=lr, max_iter=it,
                                                   random_state=42, early_stopping=False).fit(Xtr, ytr)
                s = _mf1(yva, m.predict(Xva))
                if best is None or s > best[0]:
                    best = (s, dict(max_depth=depth, lr=lr, max_iter=it), m)
    s, hp, m = best
    return m.predict_proba, {**hp, "val_f1": s}


def fit_shallow_models(tr, va):
    """Fit the four shallow models; returns (models dict name->(predict_proba, hp, feat_family), X dict)."""
    ytr, yva = tr["label"].to_numpy(), va["label"].to_numpy()
    X = {fam: {"train": feature_matrix(tr, f)[0], "val": feature_matrix(va, f)[0]}
         for fam, f in (("len", LEN_FEATS), ("stylo", STYLO_FEATS))}
    models = {}
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, fam in (("length-only LR", "len"), ("stylometry LR", "stylo")):
            f, hp = fit_lr(X[fam]["train"], ytr, X[fam]["val"], yva)
            models[name] = (f, hp, fam)
        for name, fam in (("length-only HGB", "len"), ("stylometry HGB", "stylo")):
            f, hp = fit_hgb(X[fam]["train"], ytr, X[fam]["val"], yva)
            models[name] = (f, hp, fam)
    return models, X


def main():
    sp = load_splits()
    tr, va = sp["train"], sp["val"]
    te = canon(sp["test"])
    te_ref = load_preds("runs/tfidf_output_base.npz")
    assert np.array_equal(te["source_id"].astype(str).to_numpy(), te_ref["sid"])
    assert np.array_equal(te["label"].to_numpy(), te_ref["y"])
    yva, yte = va["label"].to_numpy(), te["label"].to_numpy()
    sid = te["source_id"].astype(str).to_numpy()
    models, X = fit_shallow_models(tr, va)
    fams = {"len": LEN_FEATS, "stylo": STYLO_FEATS}
    Xte, ntok_te = {}, None
    for fam, f in fams.items():
        Xte[fam], ntok_te = feature_matrix(te, f)

    res = {}
    for name, (f, hp, fam) in models.items():
        res[name] = dict(hp=hp, val_pred=f(X[fam]["val"]).argmax(1), pred=f(Xte[fam]).argmax(1))

    rows = []
    for name, r in res.items():
        va_acc, va_f1 = float((r["val_pred"] == yva).mean()), _mf1(yva, r["val_pred"])
        a, f1, ac, fc = ci_row(yte, r["pred"], sid)
        hp = ", ".join(f"{k}={v}" for k, v in r["hp"].items() if k != "val_f1")
        rows.append([name, hp, f"{100 * va_acc:.2f} / {100 * va_f1:.2f}", fmt_ci(a, ac), fmt_ci(f1, fc)])
    tf = load_preds("runs/tfidf_output_base.npz")
    zt = np.load("runs/tfidf_output_base.npz")
    cnn_p = np.mean([load_preds(f"runs/cnn_output_base_s{s}.npz")["prob"] for s in range(3)], 0)
    bl_p = np.mean([load_preds(f"runs/bilstm_output_base_s{s}.npz")["prob"] for s in range(3)], 0)
    allm = {"length-only LR": res["length-only LR"]["pred"], "length-only HGB": res["length-only HGB"]["pred"],
            "stylometry LR": res["stylometry LR"]["pred"], "stylometry HGB": res["stylometry HGB"]["pred"],
            "TF-IDF (word, output)": tf["pred"], "CNN 3-seed ens (output)": cnn_p.argmax(1),
            "BiLSTM 3-seed ens (output)": bl_p.argmax(1)}
    va_acc = float((zt["val_pred"] == zt["val_y"]).mean())
    va_f1 = _mf1(zt["val_y"], zt["val_pred"])
    a, f1, ac, fc = ci_row(yte, tf["pred"], sid)
    ref_rows = [["TF-IDF (word, output)", "C=100 (val)", f"{100 * va_acc:.2f} / {100 * va_f1:.2f}", fmt_ci(a, ac), fmt_ci(f1, fc)]]
    for nm in ("CNN 3-seed ens (output)", "BiLSTM 3-seed ens (output)"):
        a, f1, ac, fc = ci_row(yte, allm[nm], sid)
        ref_rows.append([nm, "-", "n/a (no val probs saved)", fmt_ci(a, ac), fmt_ci(f1, fc)])
    t_main = md_table(["model", "hyperparams (chosen on VAL)", "VAL acc / macro-F1 %", "TEST acc % [95% cluster CI]",
                       "TEST macro-F1 % [95% CI]"], rows + ref_rows)

    # (c) raw-length quintiles on TEST (word_len = raw words of the output)
    wl = te["word_len"].to_numpy()
    q = length_bins(wl, 5)
    q_rows = []
    cols = ("TF-IDF (word, output)", "CNN 3-seed ens (output)", "BiLSTM 3-seed ens (output)", "stylometry HGB", "length-only LR")
    for b in range(5):
        m = q == b
        mix = " / ".join(f"{100 * (yte[m] == c).mean():.0f}" for c in range(5))
        q_rows.append([f"Q{b + 1}", f"{wl[m].min()}-{wl[m].max()}", int(m.sum()),
                       *[f"{100 * (allm[k][m] == yte[m]).mean():.1f}" for k in cols], mix])
    t_q = md_table(["quintile", "raw words", "n", "TF-IDF acc %", "CNN acc %", "BiLSTM acc %", "stylo-HGB acc %",
                    "length-only LR acc %", "class mix % (chatgpt/cohere/llama/mistral/mpt)"], q_rows)

    # (d) length-matched subset
    bins = length_bins(wl, 10)
    idx = length_matched_indices(yte, bins, seed=0)
    sub = {k: [] for k in allm}
    for sd in range(20):  # robustness over resampling seeds
        ix = length_matched_indices(yte, bins, seed=sd)
        for k, p in allm.items():
            sub[k].append((p[ix] == yte[ix]).mean())
    lm_rows = []
    for k, p in allm.items():
        a, f1, ac, fc = ci_row(yte[idx], p[idx], sid[idx])
        lm_rows.append([k, f"{100 * (p == yte).mean():.2f}", fmt_ci(a, ac),
                        f"{100 * np.mean(sub[k]):.2f} +/- {100 * np.std(sub[k], ddof=1):.2f}"])
    t_lm = md_table(["model", "full TEST acc %", "length-matched TEST acc % (seed 0) [95% cluster CI]",
                     "mean +/- sd over 20 resample seeds"], lm_rows)
    cnt = np.bincount(yte[idx], minlength=5)

    cap = ntok_te >= CAP
    cap_rows = [[k, f"{100 * (p[~cap] == yte[~cap]).mean():.2f}", f"{100 * (p[cap] == yte[cap]).mean():.2f}"]
                for k, p in allm.items()]
    t_cap = md_table(["model", f"acc % below cap (<{CAP} tokens, n={int((~cap).sum())})",
                      f"acc % at cap (>={CAP} tokens, n={int(cap.sum())})"], cap_rows)
    capmix = " / ".join(f"{100 * (yte[cap] == c).sum() / (yte == c).sum():.0f}" for c in range(5))

    L = [HDR, "",
         "Generated by `python -m src.analysis.t5_length`. All hyperparameters chosen on VAL macro-F1; TEST scored once. Features "
         "use the OUTPUT text only (word/char counts are the stored raw `word_len`/`char_len` columns; `hit_cap` = tokeniser "
         f"count >= {CAP}, i.e. truncated by the neural/TF-IDF cap). Stylometry = {len(STYLO_FEATS)} features: "
         + ", ".join(STYLO_FEATS) + ". LR uses standardised features (C grid 0.001..100); HGB grid depth {3,None} x lr "
         "{0.05,0.1} x iters {100,300}. CIs: cluster bootstrap over test source_ids (2000 resamples, seed 0). Chance = 20%.", "",
         "### (a)(b) Shallow baselines vs trained models", "", *t_main, "",
         "(Neural 3-seed ensembles saved only test probabilities, so no VAL number is shown for them.)", "",
         "### (c) Accuracy within raw-length quintiles [TEST split: test]", "",
         "Quintile edges are TEST quantiles of raw word count (descriptive only).", "", *t_q, "",
         "### (d) Length-matched TEST subset [TEST split: test]", "",
         "Rows resampled so every class has the same histogram over 10 equal-frequency raw-word-count bins (min class count per "
         f"bin, no replacement, seed 0): {len(idx)} rows ({', '.join(str(c) for c in cnt)} per class). Models are NOT retrained; "
         "their predictions on the subset are scored.", "", *t_lm, "",
         "Below vs at the 256-token cap [TEST split: test]. Share of each class's test rows that hit the cap "
         f"(chatgpt/cohere/llama/mistral/mpt, %): {capmix}.", "", *t_cap, ""]
    body = "\n".join(L)
    write_section(HDR, body, results_md="reports/results.md", results_body=body)
    print(body)


if __name__ == "__main__":
    main()
