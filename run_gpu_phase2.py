#!/usr/bin/env python
"""Phase 2 (after run_gpu.py): neural ablations, extended tuning, data scaling. Resumable, CUDA or CPU.

    python run_gpu_phase2.py --quick               # smoke test: 1 epoch, 1 seed, tiny configs, writes ONLY to quick_out/phase2/
    python run_gpu_phase2.py                       # everything, in order T8 -> T9 -> T10
    python run_gpu_phase2.py --only t8             # one or more of t8 t9 t10
    python run_gpu_phase2.py --skip-tune           # T9: reuse reports/tuning/*_output_ext.json (no search)
    python run_gpu_phase2.py --with-emb-dropout    # T9: also search embedding dropout 0.2 (doubles the CNN/BiLSTM grids)

Steps (a step is skipped when its output file exists; delete the file to redo it). All runs: 3 seeds (0 1 2), --eval-test,
test evaluated once at the best-VAL checkpoint, tuning on VAL only. Requires the base results of run_gpu.py
(reports/tuning/{cnn,bilstm}_output.json).

T8  neural ablations (variant output; base config from reports/tuning/*_output.json)
    t8_base_valprobs  base CNN/BiLSTM, output+both, tag 'base_vp' (identical config; also stores VAL probabilities;
                      matches 'base' within seed noise; replaces nothing)
    t8_abl_*          lowercase (cnn, bilstm); output cap 64/128/512 (cnn) and 128/512 (bilstm); format-normalised (cnn)
T9  extended tuning (VAL only): tune_ext_{cnn,bilstm} -> reports/tuning/{cnn,bilstm}_output_ext.json, then final runs
    tag 'ext' (output and both) with the chosen config
T10 data scaling, SECONDARY / beyond spec: CNN with the T9-chosen config (fallback: stage-1 best) at 100% of the 2,240
    train ids (tag scale100) and on the expanded train set data/processed/raid_scaled.parquet (tag scaled); optional
    --scaling-cnn-fractions 25 50 adds nested subsets (tags scalefrac25/50); TF-IDF learning curve (CPU, minutes).
report: python -m src.aggregate (keeps the appended sections) + python -m src.analysis.t8_report
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import run_gpu as G
from src.phase2_spec import ABLATIONS, BASE_VP_TAG, EXT_TAG, HP_FLAGS, SCALED_DATA

ROOT = G.ROOT
MODELS = G.MODELS
VARIANTS = G.VARIANTS
QUICK_HP = ["--emb", "16", "--hid", "16", "--nf", "8"]


def hp_flags(params):
    out = []
    for k, v in params.items():
        out += [HP_FLAGS[k], str(v)]
    return out


def read_best(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))["best"]["params"]


def train_args(model, variant, seed, tag, hp, runs, epochs, dev, extra=()):
    return ["-m", "src.train", "--model", model, "--variant", variant, "--seeds", str(seed), "--eval-test", "--tag", tag,
            "--max-epochs", str(epochs), "--runs-dir", str(runs), *hp, *extra, *dev]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="smoke test (1 epoch, 1 seed, tiny configs, writes to quick_out/phase2/)")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda|cuda:N")
    ap.add_argument("--threads", type=int, default=8, help="CPU threads (ignored on CUDA)")
    ap.add_argument("--skip-tune", action="store_true", help="T9: do not search; reuse existing *_output_ext.json")
    ap.add_argument("--only", nargs="+", choices=["t8", "t9", "t10"], default=["t8", "t9", "t10"])
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--with-emb-dropout", action="store_true", help="T9: also search embedding dropout 0.2")
    ap.add_argument("--top-k", type=int, default=3, help="T9: configs promoted to the 3-seed stage")
    ap.add_argument("--scaling-cnn-fractions", type=int, nargs="*", default=[],
                    help="T10: extra CNN points at these percent of the 2,240 train ids (e.g. 25 50)")
    a = ap.parse_args(argv)

    base = ROOT / "quick_out" / "phase2" if a.quick else ROOT
    logs, runs = base / "logs", base / "runs"
    tune_dir, bl_dir = base / "reports" / "tuning", base / "reports" / "baselines"
    results_md, analysis_md = base / "reports" / "results.md", base / "reports" / "analysis.md"
    ref_tune = ROOT / "reports" / "tuning"  # stage-1 (base) tuning always comes from the real files
    for d in (logs, runs, tune_dir, bl_dir):
        d.mkdir(parents=True, exist_ok=True)
    seeds = a.seeds or ([0] if a.quick else [0, 1, 2])
    epochs = 1 if a.quick else a.max_epochs
    dev = ["--device", a.device, "--threads", str(a.threads)]

    try:
        import torch
        print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}"
              + (f" | {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "")
              + f" | --device {a.device}", flush=True)
    except ImportError:
        sys.exit("torch is not installed; see README 'Running on a GPU machine'")
    for m in MODELS:
        if not (ref_tune / f"{m}_output.json").exists():
            sys.exit(f"{ref_tune / (m + '_output.json')} missing: run run_gpu.py first (phase 2 reuses its tuned base configs)")
    if "t10" in a.only and not (ROOT / SCALED_DATA).exists():
        sys.exit(f"{SCALED_DATA} missing (tracked in git; rebuild with `python -m src.make_scaling_data`, needs data/interim/)")

    def base_hp(m):
        return QUICK_HP if a.quick else hp_flags(read_best(ref_tune / f"{m}_output.json"))

    def train_step(name, model, variant, seed, tag, hp, extra=()):
        stem = f"{model}_{variant}_{tag}_s{seed}"
        G.run_step(f"{name}_{stem}", train_args(model, variant, seed, tag, hp, runs, epochs, dev, extra), logs,
                   lambda: (runs / f"{stem}.json").exists())

    # ------------------------------------------------------------------ T8
    if "t8" in a.only:
        for m in MODELS:   # base re-run that also saves VAL probabilities; same configs as the base runs
            for v in VARIANTS:
                for s in seeds:
                    train_step("t8_base_valprobs", m, v, s, BASE_VP_TAG, base_hp(m))
        for m, tag, extra, _desc in ABLATIONS:
            for s in seeds:
                train_step("t8", m, "output", s, tag, base_hp(m), extra)

    # ------------------------------------------------------------------ T9
    ext = {}
    if "t9" in a.only:
        for m in MODELS:
            f = tune_dir / f"{m}_output_ext.json"
            if not a.skip_tune:
                args = ["-m", "src.tune", "--model", m, "--variant", "output", "--ext", "--out-dir", str(tune_dir),
                        "--max-epochs", str(epochs), "--top-k", "1" if a.quick else str(a.top_k), *dev]
                if a.with_emb_dropout:
                    args.append("--with-emb-dropout")
                if a.quick:
                    args += ["--max-configs", "2", "--quick-seeds"]
                G.run_step(f"tune_ext_{m}_output", args, logs, f.exists)
            if not f.exists():
                print(f"[warn] {f} missing and --skip-tune given: skipping T9 final runs for {m}", flush=True)
                continue
            ext[m] = hp_flags(read_best(f))
            for v in VARIANTS:
                for s in seeds:
                    train_step("t9_final", m, v, s, EXT_TAG, ext[m])

    # ------------------------------------------------------------------ T10
    if "t10" in a.only:
        f = tune_dir / "cnn_output_ext.json"
        if "cnn" in ext:
            cnn_hp, src = ext["cnn"], "T9-chosen"
        elif f.exists():
            cnn_hp, src = hp_flags(read_best(f)), "T9-chosen"
        elif a.quick:
            cnn_hp, src = QUICK_HP, "tiny quick"
        else:
            cnn_hp, src = base_hp("cnn"), "stage-1 best (T9 missing)"
        print(f"[T10] CNN config: {src}: {' '.join(cnn_hp)}", flush=True)
        for s in seeds:
            train_step("t10", "cnn", "output", s, "scale100", cnn_hp)
            train_step("t10", "cnn", "output", s, "scaled", cnn_hp, ["--data", SCALED_DATA])
            for pct in a.scaling_cnn_fractions:
                train_step("t10", "cnn", "output", s, f"scalefrac{pct}", cnn_hp, ["--data-fraction", str(pct / 100)])
        if not a.quick:  # CPU-only, a few minutes
            G.run_step("t10_tfidf_curve", ["-m", "src.analysis.t10_scaling_tfidf", "--no-section"], logs,
                       lambda: (bl_dir / "scaling_tfidf.json").exists())

    # ------------------------------------------------------------------ report (cheap, always re-run)
    real_runs, real_bl = ROOT / "runs", ROOT / "reports" / "baselines"
    G.run_step("report_aggregate", ["-m", "src.aggregate", "--runs", str(runs if a.quick else real_runs),
                                    "--baselines", str(bl_dir if a.quick else real_bl), "--out", str(results_md)], logs)
    G.run_step("report_phase2", ["-m", "src.analysis.t8_report", "--runs", str(runs), "--base-runs", str(real_runs),
                                 "--tuning", str(tune_dir), "--baselines", str(bl_dir if a.quick else real_bl),
                                 "--results", str(results_md), "--analysis", str(analysis_md),
                                 "--fig-dir", str(base / "reports" / "figures")], logs)
    G.print_summary()
    print(f"\nPhase-2 sections are in {results_md} (## Ablations, ## Extended tuning, "
          "## Secondary: learning curve (beyond spec)).")
    print("Copy back: git add runs reports data/processed && commit && push (see README 'Phase 2 on the GPU machine').")


if __name__ == "__main__":
    main()
