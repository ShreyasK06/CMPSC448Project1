"""T7: RQ3 diagnosis (text-trained models on code outputs) and the code-only grouped-CV ceiling.

python -m src.analysis.t7_rq3      (idempotent; re-run after src/aggregate.py to re-append to results.md)

Models are trained on text train only (RQ3 split = main split); code ids are never used for training or selection except in the
separate ceiling experiment, which is a grouped 5-fold CV on the 400 code source_ids alone (C chosen on an inner grouped split).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src import baselines as B
from src.analysis.common import CAP, RQ3, canon, md_table, n_tokens, write_section
from src.analysis.cv_tfidf import cv_variant, mean_std
from src.analysis.t4_errors import confusion, row_normalise
from src.analysis.t5_length import STYLO_FEATS, feature_matrix, fit_shallow_models
from src.preprocess import Vocab, tokenize

HDR = "## RQ3 diagnosis: text-trained models on code (T7)"
NAMES = B.LABEL_NAMES


def predicted_distribution(pred, k=5):
    """Share (%) of predictions per class."""
    return 100 * np.bincount(np.asarray(pred), minlength=k) / max(len(pred), 1)


def oov_rate(token_lists, vocab_set):
    """Token-level fraction of tokens not in vocab_set, pooled over all token lists."""
    tot = sum(len(t) for t in token_lists)
    miss = sum(1 for t in token_lists for x in t if x not in vocab_set)
    return miss / max(tot, 1)


def main():
    df = pd.read_parquet(RQ3)
    tr = df[df["split_rq3"] == "train"].reset_index(drop=True)
    va = df[df["split_rq3"] == "val"].reset_index(drop=True)
    ti = canon(df[df["split_rq3"] == "test_indomain"].reset_index(drop=True))
    code = canon(df[df["split_rq3"] == "test_code"].reset_index(drop=True))
    ycode, sidc = code["label"].to_numpy(), code["source_id"].astype(str).to_numpy()

    def load_code(prefix):
        out = []
        for s in range(3):
            z = np.load(f"runs/{prefix}_output_rq3_s{s}.npz")
            o = np.lexsort((z["test_code_y"], z["test_code_source_id"].astype(str)))
            assert np.array_equal(z["test_code_source_id"].astype(str)[o], sidc) and np.array_equal(z["test_code_y"][o], ycode)
            out.append(z["test_code_prob"][o])
        return out

    cnn_runs, bl_runs = load_code("cnn"), load_code("bilstm")

    # TF-IDF (same train as base): refit at the VAL-chosen C on rq3 train, single BLAS thread
    C = json.load(open("runs/tfidf_output_base.json"))["best_C"]
    with threadpool_limits(limits=1):
        _, art = B.run_variant({"train": tr, "val": va, "test": code}, "output", eval_test=True, c_grid=(C,), n_jobs=1)
        vec, model = art["vectorizer"], art["model"]
        p_tf = model.predict(vec.transform(B.build_texts(code, "output", True)))
        p_tf_in = model.predict(vec.transform(B.build_texts(ti, "output", True)))
    yin = ti["label"].to_numpy()

    sty_f = fit_shallow_models(tr, va)[0]["stylometry HGB"][0]  # length/format-only diagnostic
    p_sty = sty_f(feature_matrix(code, STYLO_FEATS)[0]).argmax(1)

    fams = {"TF-IDF (word)": (p_tf, None), "CNN 3-seed ens": (np.mean(cnn_runs, 0).argmax(1), cnn_runs),
            "BiLSTM 3-seed ens": (np.mean(bl_runs, 0).argmax(1), bl_runs), "stylometry-HGB (diagnostic)": (p_sty, None)}
    drows, arows = [], []
    for nm, (p, runs) in fams.items():
        drows.append([nm] + [f"{d:.1f}" for d in predicted_distribution(p)] + [f"{100 * (p == ycode).mean():.2f}"])
        if runs is not None:
            accs = [100 * (r.argmax(1) == ycode).mean() for r in runs]
            m, s = mean_std(accs)
            arows.append([nm.replace("3-seed ens", "single seeds"), f"{m:.2f} +/- {s:.2f}", ", ".join(f"{a:.2f}" for a in accs)])
    t_dist = md_table(["model (trained on text)", *[f"% pred {n}" for n in NAMES], "code acc %"], drows)
    t_seed = md_table(["model", "code acc % mean +/- sd (3 seeds)", "per seed"], arows)

    cm_blocks = []
    for nm in ("TF-IDF (word)", "CNN 3-seed ens"):
        rn = row_normalise(confusion(ycode, fams[nm][0]))
        cm_blocks += [f"**{nm}** on code, row-normalised % (rows = true)", "",
                      *md_table(["true \\ pred"] + NAMES, [[NAMES[i]] + [f"{100 * rn[i, j]:.1f}" for j in range(5)] for i in range(5)]), ""]

    def prof(d):
        g = d["generation"].astype(str)
        nl, nt, y = g.str.count("\n").to_numpy(), n_tokens(d["generation"]), d["label"].to_numpy()
        fence = g.str.contains("```", regex=False).to_numpy()
        w = d["word_len"].to_numpy()
        return [[f"{w[y == c].mean():.0f}", f"{100 * (nt[y == c] >= CAP).mean():.0f}", f"{nl[y == c].mean():.1f}",
                 f"{100 * fence[y == c].mean():.0f}"] for c in range(5)]
    pc, pt = prof(code), prof(ti)
    t_prof = md_table(["class", "text: mean words", "text: % at cap", "text: mean newlines", "text: % with ``` fence",
                       "code: mean words", "code: % at cap", "code: mean newlines", "code: % with ``` fence"],
                      [[NAMES[i]] + pt[i] + pc[i] for i in range(5)])

    vocab = Vocab(json.loads(Path("data/processed/vocab.json").read_text(encoding="utf-8"))["itos"])
    vs = set(vocab.itos)
    tok_c = [tokenize(t)[:CAP] for t in code["generation"]]
    tok_t = [tokenize(t)[:CAP] for t in ti["generation"]]
    uni = {k for k in vec.vocabulary_ if " " not in k}
    t_oov = md_table(["test set (first 256 tokens)", f"% tokens OOV for neural vocab ({len(vocab)} types, train text only)",
                      f"% tokens not a TF-IDF unigram feature ({len(uni)} unigrams)"],
                     [["code (test_code)", f"{100 * oov_rate(tok_c, vs):.2f}", f"{100 * oov_rate(tok_c, uni):.2f}"],
                      ["text (test_indomain)", f"{100 * oov_rate(tok_t, vs):.2f}", f"{100 * oov_rate(tok_t, uni):.2f}"]])

    # ceiling: grouped 5-fold CV of TF-IDF on code only
    codeall = df[df["split_rq3"] == "test_code"].reset_index(drop=True)
    with threadpool_limits(limits=1):
        folds = cv_variant(codeall, "output", 5)
    ma, sa = mean_std([f["accuracy"] for f in folds])
    mf, sf = mean_std([f["macro_f1"] for f in folds])
    cv_json = json.load(open("reports/baselines/cv_tfidf.json"))["output"]
    ta, ts = mean_std([f["accuracy"] for f in cv_json])
    tmf, tsf = mean_std([f["macro_f1"] for f in cv_json])
    t_ceil = md_table(["setting", "acc %", "macro-F1 %"], [
        ["TF-IDF trained on TEXT, evaluated on code (no code in training) [TEST split: test_code]",
         f"{100 * (p_tf == ycode).mean():.2f}", f"{100 * B._metrics(ycode, p_tf)['macro_f1']:.2f}"],
        ["TF-IDF trained on TEXT, evaluated on in-domain text [TEST split: test]",
         f"{100 * (p_tf_in == yin).mean():.2f}", f"{100 * B._metrics(yin, p_tf_in)['macro_f1']:.2f}"],
        ["CEILING: TF-IDF trained AND tested on code, grouped 5-fold CV over the 400 code source_ids "
         f"(mean +/- sd over folds; C per fold {', '.join(str(f['C']) for f in folds)}) [CV on code]",
         f"{100 * ma:.2f} +/- {100 * sa:.2f}", f"{100 * mf:.2f} +/- {100 * sf:.2f}"],
        ["reference: TF-IDF grouped 5-fold CV on text train+val [CV on text]", f"{100 * ta:.2f} +/- {100 * ts:.2f}",
         f"{100 * tmf:.2f} +/- {100 * tsf:.2f}"]])

    L = [HDR, "",
         "Generated by `python -m src.analysis.t7_rq3`. Text-trained models (train = text train; vocab / TF-IDF fit on train only) evaluated on the 2000 "
         "code outputs (400 ids x 5 models) [TEST split: test_code]; output-only input. TF-IDF refit at VAL-chosen C=" + str(C) +
         " with a single BLAS thread. CNN/BiLSTM predictions come from the saved runs/*_output_rq3_s{0,1,2}.npz (not retrained).", "",
         "### Predicted-class distribution on code [TEST split: test_code]", "", "True class mix is 20% each.", "", *t_dist, "", *t_seed, "", *cm_blocks,
         "### Length / format of outputs by class: text vs code (test splits)", "", *t_prof, "",
         "### Out-of-vocabulary rate", "", *t_oov, "",
         "### Ceiling: can code outputs be attributed at all?", "",
         "Grouped CV over the 400 code source_ids (2000 rows), C selected inside each fold on an inner grouped 80/20 split; no text data "
         "is used. Chance = 20%.", "", *t_ceil, ""]
    short = [HDR, "", "[TEST split: test_code] unless noted; full tables in reports/analysis.md.", "", *t_dist, "", *t_ceil, ""]
    write_section(HDR, "\n".join(L), results_md="reports/results.md", results_body="\n".join(short))
    print("\n".join(L))


if __name__ == "__main__":
    main()
