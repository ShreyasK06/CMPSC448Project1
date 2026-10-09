"""Light grid search on VAL ONLY (seed 0, never evaluates test).
python -m src.tune --model cnn --variant output [--max-configs N]
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from src.train import run

GRIDS = {
    "cnn": dict(dropout=[0.3, 0.5], nf=[100, 200], emb=[128, 256]),
    "bilstm": dict(dropout=[0.3, 0.5], hid=[128, 256]),
}


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
    a = ap.parse_args(argv)
    grid = GRIDS[a.model]
    keys = list(grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*grid.values())]
    if a.max_configs:
        combos = combos[: a.max_configs]
    results = []
    for c in combos:
        r = run(dict(model=a.model, variant=a.variant, seed=0, eval_test=False, rq3=False, data=a.data,
                     lowercase=a.lowercase, max_epochs=a.max_epochs, runs_dir=None, verbose=False,
                     device=a.device, threads=a.threads, **c))
        results.append(dict(params=c, val_acc=r["val_acc"], val_macro_f1=r["val_macro_f1"],
                            best_epoch=r["best_epoch"], n_params=r["n_params"], train_time_s=r["train_time_s"]))
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
