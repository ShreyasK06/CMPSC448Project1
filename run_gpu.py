#!/usr/bin/env python
"""Run the whole pipeline sequentially (resumable). Works on Linux/Windows, CUDA or CPU.

    python run_gpu.py                 # full pipeline
    python run_gpu.py --quick         # smoke test: 1 epoch, 1 seed, tiny config, output in quick_out/
    python run_gpu.py --skip-tune     # reuse reports/tuning/*.json

Steps (a step is skipped when its output already exists; delete the file to redo it):
  a) tune CNN, then BiLSTM, on variant 'output' (VAL only)           -> reports/tuning/{model}_output.json
  b) train {cnn,bilstm} x {output,both}, seeds 0 1 2, --eval-test    -> runs/{model}_{variant}_base_s{seed}.json
  c) same with --rq3 (tag 'rq3')                                     -> runs/{model}_{variant}_rq3_s{seed}.json
  d) baselines (word, and word+char) with --eval-test, plus rq3      -> reports/baselines/*.json
  e) aggregate                                                       -> reports/results.md

The 'prompt' variant is intentionally NOT trained for the neural models: the prompt is identical
across the 5 generator models, so any classifier is at chance by construction (the TF-IDF baseline
still reports it as a sanity check). Variant 'both' reuses the hyper-parameters tuned on 'output'.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
MODELS = ("cnn", "bilstm")
VARIANTS = ("output", "both")
SEEDS_FULL = [0, 1, 2]
HP_FLAGS = {"dropout": "--dropout", "nf": "--nf", "emb": "--emb", "hid": "--hid"}
STATUS: list[tuple[str, str, float]] = []


def run_step(name, args, log_dir, done=None):
    """Run `python <args>` with output streamed to console and logs/<name>.log. Skip if done() is true."""
    t0 = time.time()
    if done is not None and done():
        print(f"[skip] {name} (output exists)", flush=True)
        STATUS.append((name, "skipped", 0.0))
        return
    log = log_dir / f"{name}.log"
    print(f"[run ] {name}: {' '.join(args)}\n       log -> {log}", flush=True)
    with open(log, "w", encoding="utf-8") as f:
        f.write("$ " + " ".join([PY, *args]) + "\n")
        p = subprocess.Popen([PY, *args], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace",
                             env={**os.environ, "PYTHONUNBUFFERED": "1"})
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
        rc = p.wait()
    dt = time.time() - t0
    if rc != 0:
        STATUS.append((name, f"FAILED rc={rc}", dt))
        print_summary()
        sys.exit(f"step {name} failed (rc={rc}); see {log}. Fix and re-run: finished steps are skipped.")
    STATUS.append((name, "done", dt))


def print_summary():
    print("\n" + "=" * 70 + "\nSUMMARY\n" + "=" * 70)
    for n, s, dt in STATUS:
        print(f"{n:<40}{s:<16}{dt / 60:7.1f} min")


def best_hp(tune_dir, model):
    d = json.loads((tune_dir / f"{model}_output.json").read_text(encoding="utf-8"))
    out = []
    for k, v in d["best"]["params"].items():
        out += [HP_FLAGS[k], str(v)]
    return out


def baseline_done(path, quick):
    if not path.exists():
        return False
    if quick:
        return True
    try:
        r = json.loads(path.read_text(encoding="utf-8"))
        return bool(r.get("eval_test")) and "1000" in r.get("val_curve", {})
    except Exception:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="smoke test (1 epoch, 1 seed, tiny config, writes to quick_out/)")
    ap.add_argument("--skip-tune", action="store_true", help="do not tune; reuse existing tuning JSONs")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda|cuda:N")
    ap.add_argument("--threads", type=int, default=8, help="CPU threads (ignored on CUDA)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--max-epochs", type=int, default=20)
    a = ap.parse_args(argv)

    base = ROOT / "quick_out" if a.quick else ROOT
    logs = base / "logs"
    runs = base / "runs"
    tune_dir = base / "reports" / "tuning"
    bl_dir = base / "reports" / "baselines"
    results_md = base / "reports" / "results.md"
    for d in (logs, runs, tune_dir, bl_dir):
        d.mkdir(parents=True, exist_ok=True)
    seeds = a.seeds or ([0] if a.quick else SEEDS_FULL)
    epochs = 1 if a.quick else a.max_epochs
    dev = ["--device", a.device, "--threads", str(a.threads)]

    try:
        import torch
        print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}"
              + (f" | {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "")
              + f" | --device {a.device}", flush=True)
    except ImportError:
        sys.exit("torch is not installed; see README 'Running on a GPU machine'")
    for f in ("data/processed/raid_main.parquet", "data/processed/raid_rq3.parquet"):
        if not (ROOT / f).exists():
            sys.exit(f"missing {f} (it is tracked in git; did you clone with all files?)")

    # (a) tuning, VAL only
    if not a.skip_tune:
        for m in MODELS:
            tune_args = ["-m", "src.tune", "--model", m, "--variant", "output", "--out-dir", str(tune_dir),
                         "--max-epochs", str(epochs), *dev]
            if a.quick:
                tune_args += ["--max-configs", "1"]
            run_step(f"tune_{m}_output", tune_args, logs, lambda m=m: (tune_dir / f"{m}_output.json").exists())
    for m in MODELS:
        if not (tune_dir / f"{m}_output.json").exists():
            sys.exit(f"{tune_dir / (m + '_output.json')} missing; run without --skip-tune")

    # (b) + (c) final training, test evaluated once at the best-val checkpoint
    for rq3 in (False, True):
        tag = "rq3" if rq3 else "base"
        for m in MODELS:
            hp = ["--emb", "16", "--hid", "16", "--nf", "8"] if a.quick else best_hp(tune_dir, m)
            for v in VARIANTS:
                for s in seeds:
                    args = ["-m", "src.train", "--model", m, "--variant", v, "--seeds", str(s), "--eval-test",
                            "--tag", tag, "--max-epochs", str(epochs), "--runs-dir", str(runs), *hp, *dev]
                    if rq3:
                        args.append("--rq3")
                    stem = f"{m}_{v}_{tag}_s{s}"
                    run_step(f"train_{stem}", args, logs, lambda stem=stem: (runs / f"{stem}.json").exists())

    # (d) baselines (TF-IDF + LR; includes the prompt variant as a chance-level sanity check)
    bl_vars = ["output"] if a.quick else ["prompt", "output", "both"]
    nj = ["--n-jobs", "1"] if a.quick else []
    for char in (False, True):
        tags = [v + ("_char" if char else "") for v in bl_vars]
        args = ["-m", "src.baselines", "--variants", *bl_vars, "--eval-test", "--out-dir", str(bl_dir), *nj]
        if char:
            args.append("--char")
        run_step("baselines_" + ("char" if char else "word"), args, logs,
                 lambda tags=tags: all(baseline_done(bl_dir / f"{t}.json", a.quick) for t in tags))
    if not a.quick:  # RQ3 baselines (word features)
        run_step("baselines_rq3", ["-m", "src.baselines", "--variants", "output", "both", "--rq3", "--eval-test",
                                   "--out-dir", str(bl_dir)], logs,
                 lambda: all(baseline_done(bl_dir / f"{t}_rq3.json", False) for t in ("output", "both")))

    # (e) aggregate (always re-run; cheap)
    run_step("aggregate", ["-m", "src.aggregate", "--runs", str(runs), "--baselines", str(bl_dir),
                           "--out", str(results_md)], logs)

    print_summary()
    print(f"\nResults table: {results_md}\n")
    print(results_md.read_text(encoding="utf-8"))
    print("Copy back: reports/ and runs/*.json (runs/*.npz are also useful). See README.")


if __name__ == "__main__":
    main()
