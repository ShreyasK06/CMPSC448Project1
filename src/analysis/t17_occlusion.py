"""RQ4 follow-up: intervene on the CNN's inputs to see which cues its predictions rely on (occlusion).

Train the checkpoints first (3 seeds, tuned CNN, 256-token cap, variant output):
  python -m src.train --model cnn --variant output --seeds 0 1 2 --eval-test --tag ext_occ --dropout 0.1 --nf 300 --emb 256 \
      --weight-decay 0.0001 --save-ckpt --runs-dir quick_out
Then:  python -m src.analysis.t17_occlusion -> section '## Occlusion of formatting, openers and the first tokens (RQ4)'

Each intervention DELETES tokens from the encoded TEST input and re-scores the same trained models (no retraining). Measured per example:
the change in the 3-seed ensemble's probability of the TRUE class and in accuracy. Controls: deleting the LAST 32 tokens and deleting 32
random tokens, so the effect of the first 32 tokens can be separated from simply having a shorter input.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.analysis.common import md_table, write_section
from src.analysis.stats import bootstrap_weights
from src.models import build_model
from src.preprocess import NL_ID, collate_fn
from src.train import DEFAULTS, get_data

HDR = "## Occlusion of formatting, openers and the first tokens (RQ4)"
LABELS = ["chatgpt", "cohere-chat", "llama-chat", "mistral-chat", "mpt-chat"]
MD_TOKENS = {"*", "#", "`", "_", "~", ">"}          # markdown emphasis / heading / quote / code markers (single-char tokens)
OPENERS = {"sure", "certainly", "here", "of", "okay", "ok", "title", "note", "as", "absolutely"}
OPENER_END = {":", ".", "!", "?"}
CFG = dict(DEFAULTS, model="cnn", variant="output", emb=256, nf=300, dropout=0.1, weight_decay=1e-4)


def _del(ids, drop):
    out = [t for t, d in zip(ids, drop) if not d]
    return out if out else [1]


def make_variants(vocab, rng_seed=0):
    itos = vocab.itos
    md_ids = {i for i, t in enumerate(itos) if t in MD_TOKENS}
    low = [t.lower() for t in itos]

    def no_nl(ids):
        return _del(ids, [t == NL_ID for t in ids])

    def no_md(ids):
        return _del(ids, [t in md_ids for t in ids])

    def no_fmt(ids):
        return _del(ids, [t == NL_ID or t in md_ids for t in ids])

    def no_opener(ids):
        if not ids or low[ids[0]] not in OPENERS:
            return list(ids)
        for j, t in enumerate(ids[:12]):
            if itos[t] in OPENER_END:
                return _del(ids, [i <= j for i in range(len(ids))])
        return list(ids)

    def first32(ids):
        return _del(ids, [i < 32 for i in range(len(ids))]) if len(ids) > 32 else list(ids)

    def last32(ids):
        return _del(ids, [i >= len(ids) - 32 for i in range(len(ids))]) if len(ids) > 32 else list(ids)

    rng = np.random.default_rng(rng_seed)

    def rand32(ids):
        if len(ids) <= 32:
            return list(ids)
        gone = set(rng.choice(len(ids), 32, replace=False).tolist())
        return _del(ids, [i in gone for i in range(len(ids))])

    return [("delete newline tokens (<NL>)", no_nl), ("delete markdown marker tokens (* # ` _ ~ >)", no_md),
            ("delete all formatting (newlines + markdown markers)", no_fmt),
            ("delete boilerplate opener (Sure/Here/Certainly... up to first : . ! ?)", no_opener),
            ("delete the first 32 tokens", first32), ("control: delete the last 32 tokens", last32),
            ("control: delete 32 random tokens", rand32)]


@torch.no_grad()
def probs(model, items, labels):
    batch = [(ids, max(1, sum(1 for t in ids if t != 0)), int(l)) for ids, l in zip(items, labels)]
    out = []
    model.eval()
    for i in range(0, len(batch), 256):
        x, ln, _ = collate_fn(batch[i:i + 256])
        out.append(torch.softmax(model(x, ln), 1).numpy())
    return np.concatenate(out)


def build(ckpt_dir="checkpoints", tag="ext_occ", seeds=(0, 1, 2)):
    vocab, ds, _ = get_data(CFG)
    te = ds["test"]
    labels, sids = np.array(te.labels), np.array([str(s) for s in te.source_ids])
    models = []
    for s in seeds:
        m = build_model("cnn", len(vocab), 5, CFG["emb"], CFG["hid"], CFG["nf"], CFG["dropout"])
        m.load_state_dict(torch.load(Path(ckpt_dir) / f"cnn_output_{tag}_s{s}.pt", map_location="cpu"))
        models.append(m)

    def ens(items):
        return np.mean([probs(m, items, labels) for m in models], axis=0)

    base = ens(te.items)
    base_pred = base.argmax(1)
    base_acc = float((base_pred == labels).mean())
    p_true = base[np.arange(len(labels)), labels]
    ids_u, inv = np.unique(sids, return_inverse=True)
    W = bootstrap_weights(len(ids_u), 2000, 0)
    n_cl = np.bincount(inv).astype(float)
    rows, cls_rows = [], []
    for name, fn in make_variants(vocab):
        items = [fn(list(x)) for x in te.items]
        changed = np.array([a != list(b) for a, b in zip(items, te.items)])
        pr = ens(items)
        d = pr[np.arange(len(labels)), labels] - p_true
        corr_d = (pr.argmax(1) == labels).astype(float) - (base_pred == labels).astype(float)
        cd = np.bincount(inv, weights=d)
        ca = np.bincount(inv, weights=corr_d)
        bd, ba = (W @ cd) / (W @ n_cl), (W @ ca) / (W @ n_cl)
        qd, qa = np.percentile(bd, [2.5, 97.5]), np.percentile(ba, [2.5, 97.5])
        rows.append([name, f"{100 * changed.mean():.0f}", f"{100 * d.mean():+.2f} [{100 * qd[0]:+.2f}, {100 * qd[1]:+.2f}]",
                     f"{100 * corr_d.mean():+.2f} [{100 * qa[0]:+.2f}, {100 * qa[1]:+.2f}]",
                     f"{100 * float((pr.argmax(1) == labels).mean()):.2f}"])
        cls_rows.append([name] + [f"{100 * d[labels == j].mean():+.1f}" for j in range(5)])
        print(name, rows[-1][1:], flush=True)
    return rows, cls_rows, base_acc, float(p_true.mean())


def body(rows, cls_rows, base_acc, base_p):
    L = [HDR, "",
         "Generated by `python -m src.analysis.t17_occlusion`. The CNN (tuned config, variant `output`, 256-token cap, 3 seeds, mean-"
         "probability ensemble) is trained once; each intervention then DELETES the listed tokens from the encoded [TEST split: test] "
         "input and re-scores the same models (no retraining). Columns: share of the 2,400 test rows the intervention changes; mean "
         "per-example change in the probability of the TRUE class (percentage points) and in accuracy, both with a 95% cluster-bootstrap "
         f"CI over the 480 source_ids; accuracy after. Intact input: accuracy {100 * base_acc:.2f}%, mean true-class probability "
         f"{100 * base_p:.1f}%. The last two rows are controls: deleting 32 tokens from the end or at random shows how much of an effect "
         "is just 'a shorter input', so the first-32-token effect should be read against them. Interventions edit the input, so they "
         "show what THIS model relies on (a causal test of the model, not of the LLMs); deleting tokens also shifts the remaining "
         "positions and creates unnatural text.", ""]
    L += md_table(["intervention", "% rows changed", "mean Δ P(true class), pp", "mean Δ accuracy, pp", "accuracy after, %"], rows)
    L += ["", "### Mean change in true-class probability by true class (pp)", ""]
    L += md_table(["intervention"] + LABELS, cls_rows)
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--tag", default="ext_occ")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--analysis", default="reports/analysis.md")
    ap.add_argument("--results", default="reports/results.md")
    a = ap.parse_args(argv)
    b = body(*build(a.ckpt_dir, a.tag, tuple(a.seeds)))
    write_section(HDR, b, analysis_md=a.analysis, results_md=a.results, results_body=b)
    print(b)


if __name__ == "__main__":
    main()
