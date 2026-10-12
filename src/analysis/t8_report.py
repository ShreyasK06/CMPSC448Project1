"""Phase-2 report sections (T8 ablations, T9 extended tuning) for reports/results.md + analysis.md.

Sections written (idempotent, replaced in place; nothing else in the files is touched):
  '## Ablations'                                  neural ablations vs the base runs  (T8)
  '## Extended tuning'                            T9 grid (VAL only) + final 'ext' runs vs base

Every table says which split each number comes from. Neural rows: mean +/- std (ddof=1) over seeds and a 3-seed mean-prob
ensemble with a cluster-bootstrap CI over test source_ids; comparisons vs base use the exact McNemar test on the ensemble
predictions (same 2,400 test rows) with Holm correction inside each section, plus a paired cluster-bootstrap CI of the
ensemble accuracy difference. Sections whose runs are missing are skipped.

python -m src.analysis.t8_report [--runs runs] [--tuning reports/tuning] [--results reports/results.md] [--analysis reports/analysis.md]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.analysis.common import ci_row, fmt_ci, md_table, write_section
from src.analysis.stats import (bootstrap_weights, cluster_confusions, holm, load_preds, mcnemar_exact)
from src.phase2_spec import ABLATIONS, BASE_VP_TAG, EXT_TAG

H_ABL = "## Ablations"
H_EXT = "## Extended tuning"
PM = "±"
N_BOOT = 2000


# ----------------------------------------------------------------- pure helpers
def pm(vals, digits=2):
    """mean +/- sample std of fractions, in %."""
    v = np.asarray(vals, float) * 100
    if not len(v):
        return "-"
    return f"{v.mean():.{digits}f} {PM} {v.std(ddof=1):.{digits}f}" if len(v) > 1 else f"{v.mean():.{digits}f}"


def load_group(runs, model, variant, tag):
    """All seeds of runs/{model}_{variant}_{tag}_s*.json -> list of dict(seed, res, npz)."""
    out = []
    for s in range(20):
        p = Path(runs) / f"{model}_{variant}_{tag}_s{s}.json"
        if p.exists():
            out.append(dict(seed=s, res=json.loads(p.read_text(encoding="utf-8")), npz=p.with_suffix(".npz")))
    return out


def ensemble(group, split="test"):
    """n-seed mean-probability ensemble -> dict(y, pred, prob, sid), or None if predictions are missing."""
    if not group or not all(g["npz"].exists() for g in group):
        return None
    ps = [load_preds(g["npz"], split) for g in group]
    for p in ps[1:]:
        if not (np.array_equal(p["sid"], ps[0]["sid"]) and np.array_equal(p["y"], ps[0]["y"])):
            raise ValueError("seed predictions are not row-aligned")
    prob = np.mean([p["prob"] for p in ps], axis=0)
    return dict(y=ps[0]["y"], sid=ps[0]["sid"], prob=prob, pred=prob.argmax(1))


def paired_acc_diff(a, b, W):
    """Point estimate and 95% cluster-bootstrap CI of acc(a) - acc(b) (aligned prediction dicts, shared weights W)."""
    _, Ca = cluster_confusions(a["y"], a["pred"], a["sid"])
    _, Cb = cluster_confusions(b["y"], b["pred"], b["sid"])
    ta, tb = np.trace(Ca, axis1=1, axis2=2), np.trace(Cb, axis1=1, axis2=2)
    n = Ca.sum((1, 2))
    d = (ta.sum() - tb.sum()) / n.sum()
    boot = (W @ (ta - tb)) / (W @ n)
    return float(d), tuple(np.percentile(boot, [2.5, 97.5]))


def compare(a, b):
    """Exact McNemar (a vs b) on aligned predictions -> (p, a_only_correct, b_only_correct)."""
    if not (np.array_equal(a["sid"], b["sid"]) and np.array_equal(a["y"], b["y"])):
        raise ValueError("rows not aligned")
    return mcnemar_exact(a["pred"] == a["y"], b["pred"] == b["y"])


def summarise(group):
    """-> dict of formatted VAL/TEST mean +/- std strings."""
    r = [g["res"] for g in group]
    has_test = all("test_acc" in x for x in r)
    return dict(n=len(group), val=f"{pm([x['val_acc'] for x in r])} / {pm([x['val_macro_f1'] for x in r])}",
                test=f"{pm([x['test_acc'] for x in r])} / {pm([x['test_macro_f1'] for x in r])}" if has_test else "-")


def _cmp_cells(ens, ref, W):
    """Return (diff_cell, pending McNemar tuple or None); ref is the reference (base) ensemble."""
    if ens is None or ref is None or W is None:
        return "-", None
    d, ci = paired_acc_diff(ens, ref, W)
    p, ref_only, run_only = compare(ref, ens)
    return f"{100 * d:+.2f} [{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}]", (p, ref_only, run_only)


def _fill_mcnemar(rows, slot):
    """rows: lists whose cell `slot` holds a pending (p, a_only, b_only) tuple or None. Holm over the pending ones."""
    idx = [i for i, r in enumerate(rows) if isinstance(r[slot], tuple)]
    if idx:
        adj, rej = holm([rows[i][slot][0] for i in idx])
        for i, pa, rj in zip(idx, adj, rej):
            p, ao, bo = rows[i][slot]
            rows[i][slot] = f"{ao}/{bo}; p={p:.3g}; Holm p={pa:.3g}{' *' if rj else ''}"
    for r in rows:
        if not isinstance(r[slot], str):
            r[slot] = "-"
    return len(idx)


def _weights(ens_list):
    ref = next((e for e in ens_list if e is not None), None)
    if ref is None:
        return None
    ids, _ = cluster_confusions(ref["y"], ref["pred"], ref["sid"])
    return bootstrap_weights(len(ids), N_BOOT, 0)


def _ci_cells(ens):
    if ens is None:
        return "-", "-"
    a, f, ac, fc = ci_row(ens["y"], ens["pred"], ens["sid"], n_boot=N_BOOT)
    return fmt_ci(a, ac), fmt_ci(f, fc)


# ----------------------------------------------------------------- T8
ABL_HEAD = ["model", "run (tag)", "n seeds", "[VAL split] acc / macro-F1 %", "[TEST split: test] acc / macro-F1 %",
            "3-seed ensemble TEST acc % [95% CI]", "ensemble TEST macro-F1 % [95% CI]",
            "Δ ensemble TEST acc vs base, pp [95% paired CI]",
            "McNemar vs base ensemble (base-only/run-only correct; exact p; Holm over this table)"]


def section_ablations(runs="runs", base_runs=None):
    base_runs = base_runs or runs
    specs = [(m, "base", "base (reference)", base_runs) for m in ("cnn", "bilstm")]
    specs += [(m, BASE_VP_TAG, "base re-run, val probs saved (should match base within seed noise)", runs)
              for m in ("cnn", "bilstm")]
    specs += [(m, tag, desc, runs) for m, tag, _, desc in ABLATIONS]
    groups = [(m, tag, desc, load_group(rd, m, "output", tag)) for m, tag, desc, rd in specs]
    if not any(g for _, tag, _, g in groups if tag != "base"):
        return None
    ens = {(m, tag): ensemble(g) for m, tag, desc, g in groups if g}
    W = _weights(list(ens.values()))
    rows = []
    for m, tag, desc, g in groups:
        if not g:
            continue
        s = summarise(g)
        e = ens.get((m, tag))
        acc_ci, f1_ci = _ci_cells(e)
        if tag == "base":
            diff, mc = "(base)", "(base)"
        else:
            diff, mc = _cmp_cells(e, ens.get((m, "base")), W)
        rows.append([m, f"{desc} (`{tag}`)", s["n"], s["val"], s["test"], acc_ci, f1_ci, diff, mc])
    n_tests = _fill_mcnemar(rows, 8)
    L = [H_ABL, "",
         "Generated by `python -m src.analysis.t8_report`. Neural ablations, variant `output` only, hyper-parameters fixed to the "
         "stage-1 VAL-tuned base config (`reports/tuning/{cnn,bilstm}_output.json`), 3 seeds each, TEST evaluated once at the "
         "best-VAL checkpoint; nothing is selected on TEST. Each ablation changes ONE thing relative to `base` (the vocab is "
         "rebuilt on TRAIN under each ablation): `abl_lc` lowercases text, `abl_lenN` caps the output at N tokens (base: 256; "
         "prompt cap unchanged), `abl_fmt` format-normalises the output (same `format_normalise` as the TF-IDF ablation T6: strip "
         "markdown/bullets/numbering/emphasis, no <NL>, whitespace collapsed). `base_vp` is the base config re-run with the new "
         "code path that also stores VAL probabilities; it replaces nothing and is shown only to size seed/implementation noise "
         "(its differences from `base` are the noise floor against which ablation deltas should be read). "
         "Columns are mean +/- sample std (ddof=1) over seeds; the ensemble averages the seeds' class probabilities. CIs: cluster "
         "bootstrap over the 480 test source_ids (2,000 resamples, shared across rows, paired for the difference column). "
         "`*` = Holm-significant at 0.05 (family = the %d comparisons in this table). Rows within a source_id are not "
         "independent, so McNemar p-values are somewhat optimistic." % n_tests, ""]
    L += md_table(ABL_HEAD, rows) + [""]
    return "\n".join(L)


# ----------------------------------------------------------------- T9
def _params_str(p):
    return ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in p.items())


def section_ext(runs="runs", tuning="reports/tuning", base_runs=None):
    base_runs = base_runs or runs
    tun = {}
    for m in ("cnn", "bilstm"):
        p = Path(tuning) / f"{m}_output_ext.json"
        if p.exists():
            tun[m] = json.loads(p.read_text(encoding="utf-8"))
    finals = {(m, v): load_group(runs, m, v, EXT_TAG) for m in ("cnn", "bilstm") for v in ("output", "both")}
    if not tun and not any(finals.values()):
        return None
    L = [H_EXT, "",
         "Generated by `python -m src.analysis.t8_report`. Stage 1 = whole grid, seed 0; stage 2 = top-3 stage-1 configs re-run with "
         "seeds 0,1,2 (seed 0 reused); the chosen config has the best MEAN [VAL split] macro-F1. Selection used VAL only; TEST was "
         "not touched until the final `ext` runs (3 seeds, variants `output` and `both`, same config for both, TEST at the "
         "best-VAL checkpoint). The old grid is the 'base' tuning (`reports/tuning/*_output.json`); the base runs used "
         "weight_decay 1e-5 and no embedding dropout.", ""]
    for m, d in tun.items():
        L += [f"### {m.upper()} grid (variant output) [VAL split]", "",
              f"Stage 1 ({len(d['stage1'])} configs, seed 0)"
              + (", incl. embedding-dropout 0.2 variants." if d.get("with_emb_dropout") else ", embedding dropout off."), ""]
        s1 = sorted(d["stage1"], key=lambda r: -r["val_macro_f1"])
        L += md_table(["rank", "config", "val acc %", "val macro-F1 %", "best epoch"],
                      [[i + 1, _params_str(r["params"]), f"{100 * r['val_acc']:.2f}", f"{100 * r['val_macro_f1']:.2f}",
                        r["best_epoch"]] for i, r in enumerate(s1)]) + ["", "Stage 2 (top configs, seeds 0,1,2)", ""]
        s2 = sorted(d["stage2"], key=lambda r: -r["val_macro_f1_mean"])
        L += md_table(["config", "val macro-F1 % by seed", "mean ± std", "val acc % (mean)"],
                      [[_params_str(r["params"]), ", ".join(f"{100 * v:.2f}" for v in r["val_macro_f1_by_seed"].values()),
                        f"{100 * r['val_macro_f1_mean']:.2f} {PM} {100 * r['val_macro_f1_std']:.2f}",
                        f"{100 * r['val_acc_mean']:.2f}"] for r in s2])
        L += ["", f"**Chosen ({m}):** {_params_str(d['best']['params'])} "
                  f"(mean val macro-F1 {100 * d['best']['val_macro_f1_mean']:.2f}).", ""]
    if any(finals.values()):
        ens, rows, groups = {}, [], []
        for (m, v), g in finals.items():
            if g:
                bg = load_group(base_runs, m, v, "base")
                ens[(m, v, EXT_TAG)] = ensemble(g)
                ens[(m, v, "base")] = ensemble(bg)
                groups.append((m, v, g, bg))
        W = _weights(list(ens.values()))
        for m, v, g, bg in groups:
            for tag, grp in (("base", bg), (EXT_TAG, g)):
                if not grp:
                    continue
                s = summarise(grp)
                acc_ci, f1_ci = _ci_cells(ens.get((m, v, tag)))
                if tag == "base":
                    diff, mc = "(base)", "(base)"
                else:
                    diff, mc = _cmp_cells(ens.get((m, v, tag)), ens.get((m, v, "base")), W)
                rows.append([m, v, tag, s["n"], s["val"], s["test"], acc_ci, f1_ci, diff, mc])
        n_tests = _fill_mcnemar(rows, 9)
        L += ["### Final runs: `ext` config vs base", "",
              "Per-seed columns are mean ± sample std (ddof=1) over seeds; ensemble = mean of the seeds' probabilities. Holm family = "
              f"the {n_tests} ext-vs-base comparisons here. Splits as labelled.", ""]
        L += md_table(["model", "variant", "run", "n seeds", "[VAL split] acc / macro-F1 %", "[TEST split: test] acc / macro-F1 %",
                       "ensemble TEST acc % [95% CI]", "ensemble TEST macro-F1 % [95% CI]",
                       "Δ ensemble TEST acc vs base, pp [95% paired CI]",
                       "McNemar vs base ensemble (base-only/ext-only correct; exact p; Holm)"], rows) + [""]
    return "\n".join(L)


# ----------------------------------------------------------------- driver
def write_all(runs="runs", tuning="reports/tuning", results_md="reports/results.md",
              analysis_md="reports/analysis.md", base_runs=None):
    done = []
    for hdr, body in ((H_ABL, section_ablations(runs, base_runs)), (H_EXT, section_ext(runs, tuning, base_runs))):
        if body is None:
            print(f"[skip] {hdr}: no runs yet")
            continue
        write_section(hdr, body, analysis_md=analysis_md, results_md=results_md, results_body=body)
        done.append(hdr)
        print(f"[wrote] {hdr}")
    return done


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--base-runs", default=None, help="where the base-tag runs live (default: --runs)")
    ap.add_argument("--tuning", default="reports/tuning")
    ap.add_argument("--results", default="reports/results.md")
    ap.add_argument("--analysis", default="reports/analysis.md")
    a = ap.parse_args(argv)
    write_all(a.runs, a.tuning, a.results, a.analysis, a.base_runs)


if __name__ == "__main__":
    main()
