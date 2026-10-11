"""RQ4 follow-up: which stylometric feature groups carry the signal, and does (length-robust) lexical diversity add anything?

python -m src.analysis.t16_feature_groups -> section '## Feature-group ablation and lexical diversity (RQ4)' in reports/{analysis,results}.md

Same data, split, HGB grid (chosen on VAL macro-F1) and TEST rows as T5 (`stylometry HGB` = all 22 features). The 22 features are split into
groups; three length-controlled diversity features are added as a separate group. Plain type-token ratio falls with text length, so the
diversity features are computed on fixed-size windows / a fixed-length prefix:
  mattr50    moving-average type-token ratio over 50-word windows (lowercased words)
  ttr100     type-token ratio of the first 100 words
  hapax100   share of the first 100 words that occur exactly once
(texts shorter than the window use all their words, so a residual length dependence remains for very short outputs).
Reported: each group alone, all-but-one group, TEST accuracy, the T5 length-matched TEST subset (models are not retrained), and group
permutation importance on the full TEST split.
"""
from __future__ import annotations

import argparse
import re
import warnings
from collections import Counter

import numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from src.analysis.common import (canon, ci_row, fmt_ci, length_bins, length_matched_indices, load_splits, md_table,
                                 write_section)
from src.analysis.t5_length import STYLO_FEATS, feature_matrix, fit_hgb

HDR = "## Feature-group ablation and lexical diversity (RQ4)"
_WORD = re.compile(r"\w+(?:['’]\w+)*")
GROUPS = {
    "length": ["word_count", "char_count", "hit_cap"],
    "layout (lines, newlines, bullets, markdown)": ["line_count", "newline_count", "blank_line_count", "bullet_count",
                                                    "markdown_count", "mean_line_len"],
    "punctuation / characters": ["punct_ratio", "digit_ratio", "upper_ratio", "comma_ratio", "exclaim_count", "colon_count",
                                 "quote_ratio"],
    "opening / ending": ["ends_terminal_punct", "starts_boilerplate", "starts_upper"],
    "lexical (word length, sentence length, TTR)": ["mean_word_len", "mean_sentence_len", "type_token_ratio"],
}
DIV = ["mattr50", "ttr100", "hapax100"]
DIV_NAME = "diversity (length-robust)"
ALL_NAMES = list(GROUPS) + [DIV_NAME]


def mattr(words, w=50):
    n = len(words)
    if n == 0:
        return 0.0
    if n <= w:
        return len(set(words)) / n
    c = Counter(words[:w])
    tot, k = len(c) / w, 1
    for i in range(w, n):
        o = words[i - w]
        c[o] -= 1
        if c[o] == 0:
            del c[o]
        c[words[i]] += 1
        tot += len(c) / w
        k += 1
    return tot / k


def diversity_features(text):
    words = [x.lower() for x in _WORD.findall(text if isinstance(text, str) else "")]
    first = words[:100]
    n = len(first)
    cnt = Counter(first)
    return [mattr(words), len(cnt) / n if n else 0.0, sum(1 for v in cnt.values() if v == 1) / n if n else 0.0]


def matrix(df):
    X, _ = feature_matrix(df, STYLO_FEATS)
    D = np.array([diversity_features(g) for g in df["generation"]], float)
    return np.hstack([X, D]), STYLO_FEATS + DIV


def cols_of(names, feats):
    idx = []
    for nme in names:
        idx += [feats.index(f) for f in (GROUPS[nme] if nme in GROUPS else DIV)]
    return sorted(idx)


def mf1(y, p):
    return f1_score(y, p, average="macro", labels=list(range(5)), zero_division=0)


def build():
    sp = load_splits()
    tr, va, te = sp["train"], sp["val"], canon(sp["test"])
    Xtr, feats = matrix(tr)
    Xva, _ = matrix(va)
    Xte, _ = matrix(te)
    ytr, yva, yte = tr["label"].to_numpy(), va["label"].to_numpy(), te["label"].to_numpy()
    sid = te["source_id"].astype(str).to_numpy()
    bins = length_bins(te["word_len"].to_numpy(), 10)
    lm_ix = [length_matched_indices(yte, bins, seed=s) for s in range(20)]

    sets = [("all 22 stylometric features (= T5 stylometry HGB)", list(GROUPS)),
            ("all 22 + 3 diversity features", ALL_NAMES)]
    sets += [(f"only: {g}", [g]) for g in ALL_NAMES]
    sets += [("only: length + diversity", ["length", DIV_NAME])]
    sets += [(f"all 25 minus: {g}", [x for x in ALL_NAMES if x != g]) for g in ALL_NAMES]

    rows, fitted = [], {}
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for label, names in sets:
            ci_ = cols_of(names, feats)
            f, _hp = fit_hgb(Xtr[:, ci_], ytr, Xva[:, ci_], yva)
            pva, pte = f(Xva[:, ci_]).argmax(1), f(Xte[:, ci_]).argmax(1)
            a, f1, ac, fc = ci_row(yte, pte, sid)
            lm = [float((pte[ix] == yte[ix]).mean()) for ix in lm_ix]
            rows.append([label, len(ci_), f"{100 * (pva == yva).mean():.2f} / {100 * mf1(yva, pva):.2f}", fmt_ci(a, ac),
                         fmt_ci(f1, fc), f"{100 * np.mean(lm):.2f} ± {100 * np.std(lm, ddof=1):.2f}"])
            fitted[label] = (f, ci_, float(a))
            print(f"{label}: test acc {100 * a:.2f}", flush=True)

    # group permutation importance on the full TEST split, model = all 25 features
    f, ci_, base_acc = fitted["all 22 + 3 diversity features"]
    rng = np.random.default_rng(0)
    imp = []
    for g in ALL_NAMES:
        gc = cols_of([g], feats)
        drops = []
        for _ in range(10):
            Xp = Xte.copy()
            Xp[:, gc] = Xte[rng.permutation(len(Xp))][:, gc]
            drops.append(base_acc - float((f(Xp[:, ci_]).argmax(1) == yte).mean()))
        imp.append([g, len(gc), f"{100 * np.mean(drops):.2f} ± {100 * np.std(drops, ddof=1):.2f}"])
    return rows, imp, base_acc


def body(rows, imp, base_acc):
    L = [HDR, "",
         "Generated by `python -m src.analysis.t16_feature_groups`. Gradient-boosted trees (same HGB grid as T5, hyper-parameters chosen on "
         "[VAL split] macro-F1, TEST scored once) on the OUTPUT text. The 22 T5 stylometric features are split into the five groups listed "
         "below; a sixth group adds three length-controlled diversity features (`mattr50` = moving-average type-token ratio over "
         "50-word windows, `ttr100` and `hapax100` = type-token ratio and hapax share of the first 100 words). Plain TTR "
         "(`type_token_ratio`, in the lexical group) falls with text length, which differs strongly by model, so a raw-TTR result "
         "would partly re-measure length; the window-based measures are the fair test. Short outputs (< 50 or < 100 words) still "
         "use all their words, so a residual length dependence remains. The last column re-scores the same trained models on the "
         "T5 length-matched TEST subset (820 rows, every class with the same length histogram; mean ± sd over 20 resamples; models "
         "not retrained), which removes length as a usable cue. Chance = 20%. [TEST split: test], CIs by cluster bootstrap over "
         "480 source_ids.", "",
         "### Accuracy by feature set", ""]
    L += md_table(["features", "n", "[VAL split] acc / macro-F1 %", "[TEST split: test] acc % [95% CI]", "TEST macro-F1 % [95% CI]",
                   "length-matched TEST acc %"], rows)
    L += ["", f"### Group permutation importance (all 25 features; TEST acc {100 * base_acc:.2f}%)", "",
          "Drop in TEST accuracy (pp) when one group's columns are jointly shuffled across rows (10 repeats, mean ± sd). "
          "Correlated features share importance, so groups with overlapping information (length, layout, lexical) can understate "
          "each other; the 'only' and 'minus' rows above are the cross-check.", ""]
    L += md_table(["group", "n features", "accuracy drop pp"], imp)
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="reports/analysis.md")
    ap.add_argument("--results", default="reports/results.md")
    a = ap.parse_args(argv)
    b = body(*build())
    write_section(HDR, b, analysis_md=a.analysis, results_md=a.results, results_body=b)
    print(b)


if __name__ == "__main__":
    main()
