"""Grid search on VAL ONLY (never evaluates test).

Stage-1 grid (default, seed 0):
    python -m src.tune --model cnn --variant output [--max-configs N]
Extended two-stage search (T9), writes reports/tuning/{model}_{variant}_ext.json:
    python -m src.tune --model cnn --variant output --ext [--with-emb-dropout] [--top-k 3]
  stage 1: every grid config, seed 0;  stage 2: the top-k configs re-run with seeds 0,1,2 (seed 0 is reused from stage 1);
  the config with the best MEAN val macro-F1 over those seeds is chosen.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from src.train import run

GRIDS = {
    "cnn": dict(dropout=[0.3, 0.5], nf=[100, 200], emb=[128, 256]),
    "bilstm": dict(dropout=[0.3, 0.5], hid=[128, 256]),
}


def ext_grid(model, with_emb_dropout=False):
    """T9 extended grids (list of param dicts). The first stage-1 grid had its CNN optimum (dropout 0.3, nf 200, emb 256)
    on the low-dropout / high-nf edge, and the BiLSTM optimum (dropout 0.5, hid 256) on the high-hid edge.
    CNN: dropout {0.1,0.2,0.3} x nf {200,300} x emb {256} x weight_decay {0,1e-4} = 12 (x2 with embedding dropout 0.2 = 24).
    BiLSTM (6; the LSTM is the expensive model, so hid 384 is only probed at the incumbent dropout 0.5):
      hid 256: dropout {0.3,0.5} x weight_decay {0,1e-4}  (4)   +   hid 384: dropout 0.5 x weight_decay {0,1e-4}  (2)."""
    if model == "cnn":
        g = [dict(dropout=d, nf=nf, emb=256, weight_decay=wd)
             for d, nf, wd in itertools.product([0.1, 0.2, 0.3], [200, 300], [0.0, 1e-4])]
    elif model == "bilstm":
        g = [dict(dropout=d, hid=256, emb=128, weight_decay=wd) for d, wd in itertools.product([0.3, 0.5], [0.0, 1e-4])]
        g += [dict(dropout=0.5, hid=384, emb=128, weight_decay=wd) for wd in [0.0, 1e-4]]
    else:
        raise ValueError(model)
    if with_emb_dropout:
        g = g + [{**c, "emb_dropout": 0.2} for c in g]
    return g


def select_best(stage2):
    """stage2: list of dict(params, val_macro_f1_by_seed{seed: f1}). Returns the entry with the best mean val macro-F1
    (ties -> first). Pure helper."""
    return max(stage2, key=lambda r: float(np.mean(list(r["val_macro_f1_by_seed"].values()))))


def _fit(a, params, seed, epochs):
    return run(dict(model=a.model, variant=a.variant, seed=seed, eval_test=False, rq3=False, data=a.data,
                    lowercase=a.lowercase, max_epochs=epochs, runs_dir=None, verbose=False,
                    device=a.device, threads=a.threads, **params))


def _row(params, r):
    return dict(params=params, val_acc=r["val_acc"], val_macro_f1=r["val_macro_f1"], best_epoch=r["best_epoch"],
                n_params=r["n_params"], train_time_s=r["train_time_s"])


def main_ext(a):
    combos = ext_grid(a.model, a.with_emb_dropout)
    if a.max_configs:
        combos = combos[: a.max_configs]
    stage1 = []
    for c in combos:
        r = _fit(a, c, 0, a.max_epochs)
        stage1.append(_row(c, r))
        print(f"[stage1 seed0] {c} -> val_f1={r['val_macro_f1']:.4f} val_acc={r['val_acc']:.4f}", flush=True)
    order = sorted(range(len(stage1)), key=lambda i: -stage1[i]["val_macro_f1"])[: a.top_k]
    seeds = [0] if a.quick_seeds else [0, 1, 2]
    stage2 = []
    for i in order:
        c = stage1[i]["params"]
        by_seed = {0: stage1[i]["val_macro_f1"]}
        accs = {0: stage1[i]["val_acc"]}
        for s in seeds[1:]:
            r = _fit(a, c, s, a.max_epochs)
            by_seed[s], accs[s] = r["val_macro_f1"], r["val_acc"]
            print(f"[stage2 seed{s}] {c} -> val_f1={r['val_macro_f1']:.4f}", flush=True)
        stage2.append(dict(params=c, val_macro_f1_by_seed={str(k): v for k, v in by_seed.items()},
                           val_macro_f1_mean=float(np.mean(list(by_seed.values()))),
                           val_macro_f1_std=float(np.std(list(by_seed.values()), ddof=1)) if len(by_seed) > 1 else 0.0,
                           val_acc_mean=float(np.mean(list(accs.values())))))
    best = select_best(stage2)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.model}_{a.variant}_ext.json").write_text(json.dumps(
        dict(model=a.model, variant=a.variant,
             selection=("VAL only, test never touched. Stage 1: whole grid, seed 0. Stage 2: top-%d configs by stage-1 val "
                        "macro-F1 re-run with seeds 0,1,2 (seed 0 reused from stage 1); best = highest mean val macro-F1."
                        % a.top_k),
             with_emb_dropout=a.with_emb_dropout, stage1=stage1, stage2=stage2,
             best=dict(params=best["params"], val_macro_f1_mean=best["val_macro_f1_mean"],
                       val_macro_f1_std=best["val_macro_f1_std"])), indent=1), encoding="utf-8")
    print("BEST (mean val macro-F1 over seeds):", best["params"], f"val_f1={best['val_macro_f1_mean']:.4f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "bilstm"], required=True)
    ap.add_argument("--variant", choices=["prompt", "output", "both"], default="output")
    ap.add_argument("--max-configs", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--data", default="data/processed/raid_main.parquet")
    ap.add_argument("--lowercase", action="store_true")
    ap.add_argument("--out-dir", default="reports/tuning")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--weight-decay", type=float, default=None, help="fixed weight decay for the stage-1 grid (default 1e-5)")
    ap.add_argument("--emb-dropout", type=float, default=None, help="fixed embedding dropout for the stage-1 grid")
    ap.add_argument("--ext", action="store_true", help="T9 extended two-stage search (-> {model}_{variant}_ext.json)")
    ap.add_argument("--with-emb-dropout", action="store_true", help="--ext: also add every config with embedding dropout 0.2")
    ap.add_argument("--top-k", type=int, default=3, help="--ext: configs promoted to the multi-seed stage")
    ap.add_argument("--quick-seeds", action="store_true", help="--ext smoke test: stage 2 uses seed 0 only")
    a = ap.parse_args(argv)
    if a.ext:
        return main_ext(a)
    grid = GRIDS[a.model]
    keys = list(grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*grid.values())]
    if a.max_configs:
        combos = combos[: a.max_configs]
    fixed = {}
    if a.weight_decay is not None:
        fixed["weight_decay"] = a.weight_decay
    if a.emb_dropout is not None:
        fixed["emb_dropout"] = a.emb_dropout
    results = []
    for c in combos:
        r = _fit(a, {**c, **fixed}, 0, a.max_epochs)
        results.append(_row(c, r))
        print(f"{c} -> val_f1={r['val_macro_f1']:.4f} val_acc={r['val_acc']:.4f}", flush=True)
    best = max(results, key=lambda r: r["val_macro_f1"])
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.model}_{a.variant}.json").write_text(json.dumps(
        dict(model=a.model, variant=a.variant, selection="val macro-F1, seed 0, no test data used",
             results=results, best=best), indent=1), encoding="utf-8")
    print("BEST (by val macro-F1):", best["params"], f"val_f1={best['val_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
