"""T4: error analysis (confusions, per-domain accuracy and format statistics, top n-grams, disagreement, ensemble).

python -m src.analysis.t4_errors      (idempotent; re-run after src/aggregate.py to re-append to results.md)

TF-IDF is refit at the VAL-chosen C (runs/tfidf_output_base.json) with a single BLAS thread; its test predictions must
equal runs/tfidf_output_base.npz. All error analysis is on [TEST split: test]. Neural .npz files only hold TEST
probabilities, so no VAL-selected ensemble weight is possible without retraining (see section text).
"""
from __future__ import annotations

import json

import numpy as np
from threadpoolctl import threadpool_limits

from src import baselines as B
from src.analysis.common import CAP, canon, ci_row, fmt_ci, load_splits, md_table, n_tokens, write_section
from src.analysis.stats import load_preds
from src.analysis.t5_length import STYLO_FEATS, feature_matrix, fit_shallow_models

HDR = "## Error analysis (T4)"
NAMES = B.LABEL_NAMES
FORMAT_WORDS = {"sure", "here", "certainly", "title", "note", "chapter", "step", "ingredients", "instructions",
                "directions", "summary", "abstract", "introduction", "conclusion", "review", "sincerely", "answer",
                "yes", "no", "dear", "hello", "hi", "sorry", "thanks", "thank", "please", "ok", "okay"}


def row_normalise(conf):
    conf = np.asarray(conf, float)
    return conf / np.maximum(conf.sum(1, keepdims=True), 1)


def confusion(y, p, k=5):
    c = np.zeros((k, k), int)
    np.add.at(c, (y, p), 1)
    return c


def top_confused_pairs(conf, n=5):
    """Unordered class pairs ranked by total mutual confusion counts: [(i, j, n_i_as_j, n_j_as_i)]."""
    k = len(conf)
    pairs = [(i, j, int(conf[i, j]), int(conf[j, i])) for i in range(k) for j in range(i + 1, k)]
    return sorted(pairs, key=lambda t: -(t[2] + t[3]))[:n]


def tag_ngram(ngram):
    """Heuristic tag for a TF-IDF feature (space-joined tokens). 'formatting' if it contains the <NL> token or a word from
    FORMAT_WORDS (chat boilerplate / section labels); 'punctuation' if any token is a pure symbol; otherwise 'content'."""
    toks = ngram.split()
    low = [t.lower() for t in toks]
    if "<nl>" in low or any(t in FORMAT_WORDS for t in low):
        return "formatting"
    if any(not any(ch.isalnum() for ch in t) for t in toks):
        return "punctuation"
    return "content"


def snippet(text, n=220):
    t = " / ".join(s.strip() for s in str(text).split("\n") if s.strip())
    return (t[:n] + "...") if len(t) > n else t


def main():
    sp = load_splits()
    tr, va, te = sp["train"], sp["val"], canon(sp["test"])
    y, sid, dom = te["label"].to_numpy(), te["source_id"].astype(str).to_numpy(), te["domain"].to_numpy()
    tf = load_preds("runs/tfidf_output_base.npz")
    assert np.array_equal(tf["sid"], sid) and np.array_equal(tf["y"], y)
    cnn_prob = np.mean([load_preds(f"runs/cnn_output_base_s{s}.npz")["prob"] for s in range(3)], 0)
    bl_prob = np.mean([load_preds(f"runs/bilstm_output_base_s{s}.npz")["prob"] for s in range(3)], 0)
    p_tf, p_cnn, p_bl = tf["pred"], cnn_prob.argmax(1), bl_prob.argmax(1)

    # ---- refit TF-IDF at val-chosen C (single BLAS thread) for coefficients
    C = json.load(open("runs/tfidf_output_base.json"))["best_C"]
    with threadpool_limits(limits=1):
        _, art = B.run_variant({"train": tr, "val": va, "test": te}, "output", eval_test=True, c_grid=(C,), n_jobs=1)
        vec, model = art["vectorizer"], art["model"]
        refit_pred = model.classes_[model.predict_proba(vec.transform(B.build_texts(te, "output", True))).argmax(1)]
    same = bool(np.array_equal(refit_pred, p_tf))
    print("refit matches saved TF-IDF preds:", same, "C =", C)

    # ---- confusion matrices
    blocks, top_pairs = [], None
    for nm, p in (("TF-IDF (word, output)", p_tf), ("CNN 3-seed mean-prob ensemble (output)", p_cnn)):
        conf = confusion(y, p)
        rn = row_normalise(conf)
        rows = [[NAMES[i]] + [f"{100 * rn[i, j]:.1f}" for j in range(5)] + [int(conf[i].sum())] for i in range(5)]
        blocks += [f"**{nm}** - row-normalised % [TEST split: test] (rows = true, columns = predicted)", "",
                   *md_table(["true \\ pred"] + NAMES + ["n"], rows), ""]
        pr = top_confused_pairs(conf, 5)
        blocks += ["Most-confused unordered pairs (i->j + j->i errors): " + "; ".join(
            f"{NAMES[i]}<->{NAMES[j]} ({a}+{b}={a + b})" for i, j, a, b in pr), ""]
        if top_pairs is None:
            top_pairs = pr

    # ---- per-domain accuracy and format statistics
    sty_f = fit_shallow_models(tr, va)[0]["stylometry HGB"][0]
    p_sty = sty_f(feature_matrix(te, STYLO_FEATS)[0]).argmax(1)
    wl = te["word_len"].to_numpy()
    ntok = n_tokens(te["generation"])
    gens = te["generation"].astype(str)
    nl = gens.str.count("\n").to_numpy()
    blank = gens.str.contains("\n\n").to_numpy()
    recs = []
    for d in sorted(set(dom)):
        m = dom == d
        cls_len = [wl[m & (y == c)].mean() for c in range(5)]
        cls_nl = [(nl[m & (y == c)] > 0).mean() for c in range(5)]
        acc = [(p[m] == y[m]).mean() for p in (p_tf, p_cnn, p_bl, p_sty)]
        recs.append((acc[0], [d, int(m.sum())] + [f"{100 * a:.1f}" for a in acc],
                     [d, f"{wl[m].mean():.0f}", f"{100 * (ntok[m] >= CAP).mean():.0f}", f"{100 * (nl[m] > 0).mean():.0f}",
                      f"{nl[m].mean():.1f}", f"{100 * blank[m].mean():.0f}", f"{wl[m].mean() / (nl[m].mean() + 1):.1f}",
                      f"{min(cls_len):.0f}-{max(cls_len):.0f}", f"{100 * min(cls_nl):.0f}-{100 * max(cls_nl):.0f}"]))
    recs.sort(key=lambda r: -r[0])
    t_dom = md_table(["domain", "n", "TF-IDF acc %", "CNN-ens acc %", "BiLSTM-ens acc %", "stylometry-HGB acc %"], [r[1] for r in recs])
    t_sta = md_table(["domain (same order)", "mean raw words", "% at 256-token cap", "% with newline", "mean newlines",
                      "% with blank line", "words per line", "per-class mean words (min-max)",
                      "per-class % with newline (min-max)"], [r[2] for r in recs])

    # ---- top-30 positive-coefficient n-grams per class
    names = np.asarray(vec.get_feature_names_out())
    ng_lines, tag_rows = [], []
    for ci, c in enumerate(model.classes_):
        w = model.coef_[ci]
        idx = [j for j in np.argsort(-w)[:30] if w[j] > 0]
        tags = [tag_ngram(str(names[j])) for j in idx]
        tag_rows.append([NAMES[int(c)], len(idx)] + [sum(t == k for t in tags) for k in ("content", "formatting", "punctuation")])
        ng_lines.append(f"- **{NAMES[int(c)]}**: " + ", ".join(f"`{names[j]}` [{t[0]}]" for j, t in zip(idx, tags)))
    t_tag = md_table(["class", "n positive", "content", "formatting/boilerplate", "punctuation"], tag_rows)

    # ---- disagreement
    ok_t, ok_c = p_tf == y, p_cnn == y
    t_dis = md_table(["outcome", "rows (of 2400)"], [["both right", int((ok_t & ok_c).sum())], ["TF-IDF only right", int((ok_t & ~ok_c).sum())],
                                                     ["CNN-ens only right", int((~ok_t & ok_c).sum())], ["both wrong", int((~ok_t & ~ok_c).sum())]])
    bw = ~ok_t & ~ok_c
    same_wrong = int((bw & (p_tf == p_cnn)).sum())

    # ---- ensemble (secondary): fixed weight, plus a flagged diagnostic sweep
    pe = (0.5 * tf["prob"] + 0.5 * cnn_prob).argmax(1)
    a, fm, ac, fc = ci_row(y, pe, sid)
    t_ens = md_table(["TF-IDF prob x w + CNN-ens prob x (1-w)", "TEST acc % [95% CI]", "TEST macro-F1 % [95% CI]"],
                     [["w=0.5 (pre-specified, not tuned)", fmt_ci(a, ac), fmt_ci(fm, fc)]])
    sweep = ", ".join(f"{w:.1f}: {100 * (((w * tf['prob'] + (1 - w) * cnn_prob).argmax(1)) == y).mean():.2f}" for w in np.linspace(0, 1, 11))

    # ---- examples
    ex, rng = [], np.random.default_rng(0)
    for i, j, _, _ in top_pairs[:2]:
        for a_, b_, k in ((i, j, 2), (j, i, 1)):
            cand = np.flatnonzero((y == a_) & (p_tf == b_))
            for r in rng.choice(cand, size=min(len(cand), k), replace=False):
                ex.append(f"- true **{NAMES[a_]}**, TF-IDF predicted **{NAMES[b_]}** (domain {dom[r]}, {wl[r]} words): "
                          f"\"{snippet(te['generation'].iloc[r])}\"")

    L = [HDR, "",
         "Generated by `python -m src.analysis.t4_errors`. All numbers are on [TEST split: test] (2400 rows) unless noted. TF-IDF = word 1-2gram, "
         f"output text, C={C} (chosen on VAL; refit with a single BLAS thread; refit predictions identical to runs/tfidf_output_base.npz: {same}). "
         "CNN/BiLSTM = mean of the 3 seeds' probabilities. Stylometry-HGB is the T5 model (hyperparameters from VAL).", "",
         "### Confusion matrices", "", *blocks,
         "### Per-domain accuracy [TEST split: test]", "", *t_dom, "",
         "### Per-domain length / format statistics [TEST split: test]", "",
         "Output text only; `words per line` = mean raw words / (mean newlines + 1). The last two columns give the range of the class means "
         "inside the domain (a wide range means length / layout alone separates models there).", "", *t_sta, "",
         f"### Top positive-coefficient n-grams per class (TF-IDF word model, C={C}, trained on train)", "",
         "Tag heuristic: [f]ormatting = contains the `<NL>` token or a word from {" + ", ".join(sorted(FORMAT_WORDS)) + "} (chat boilerplate / "
         "section labels); [p]unctuation = any token is a pure symbol (so mixed punctuation+word n-grams count as punctuation); otherwise [c]ontent. "
         "It is a crude lexical rule and does not understand meaning.", "",
         *t_tag, "", *ng_lines, "",
         "### TF-IDF vs CNN disagreement [TEST split: test]", "", *t_dis, "",
         f"Of the {int(bw.sum())} rows both get wrong, {same_wrong} get the same wrong label from both.", "",
         "### Probability-averaging ensemble (SECONDARY, not a headline)", "",
         "The neural .npz files store TEST probabilities only (no VAL probabilities), so a VAL-chosen weight is not possible without retraining "
         "(not done). The ensemble uses a pre-specified equal weight; the sweep is a TEST-side diagnostic and was NOT used to pick anything.", "",
         *t_ens, "", f"Diagnostic sweep, TEST acc % by w_TF-IDF: {sweep}.", "",
         "### Misclassified examples (TF-IDF errors on the top-2 confused pairs)", "", *ex, ""]
    body = "\n".join(L)
    short = L[:2] + ["### Per-domain accuracy [TEST split: test]", "", *t_dom, "", "### TF-IDF vs CNN disagreement [TEST split: test]", "", *t_dis,
                     "", "(Full tables: reports/analysis.md.)", ""]
    write_section(HDR, body, results_md="reports/results.md", results_body="\n".join(short))
    print(body)


if __name__ == "__main__":
    main()
