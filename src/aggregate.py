"""Aggregate runs/*.json and reports/baselines/*.json into markdown tables (mean +/- std over seeds).
python -m src.aggregate [--runs runs] [--baselines reports/baselines] [--out file.md]

Baseline JSONs are read leniently: each file may be a dict or list of dicts with keys like
model/variant/tag/seed and val_acc/val_macro_f1/test_acc/test_macro_f1 (or nested val/test dicts
with acc/macro_f1). Unknown shapes are skipped with a warning on stderr.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PM = "±"


def _first(d, *keys):
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    return None


def _norm(rec, src):
    """-> dict(model, variant, tag, seed, metrics{col: value}) or None."""
    cfg = rec.get("config", rec)
    model = _first(cfg, "model", "name")
    if model is None:
        return None
    m = {"val acc": _first(rec, "val_acc", "val_accuracy"),
         "val macro-F1": _first(rec, "val_macro_f1", "val_f1")}
    for k in ("val", "validation"):
        if isinstance(rec.get(k), dict):
            if m["val acc"] is None:
                m["val acc"] = _first(rec[k], "acc", "accuracy")
            if m["val macro-F1"] is None:
                m["val macro-F1"] = _first(rec[k], "macro_f1", "f1")
    tests = rec.get("test")
    if isinstance(tests, dict) and tests and all(isinstance(v, dict) for v in tests.values()):
        for name, t in tests.items():  # e.g. {"test": {...}} or {"test_indomain": {...}, "test_code": {...}}
            m[f"{name} acc"] = _first(t, "acc", "accuracy")
            m[f"{name} macro-F1"] = _first(t, "macro_f1", "f1")
    else:
        t = tests if isinstance(tests, dict) else {}
        m["test acc"] = _first(rec, "test_acc", "test_accuracy") or _first(t, "acc", "accuracy")
        m["test macro-F1"] = _first(rec, "test_macro_f1", "test_f1") or _first(t, "macro_f1", "f1")
    return dict(model=str(model), variant=str(cfg.get("variant", "-")), tag=str(cfg.get("tag", "base")),
                seed=cfg.get("seed", 0), src=src, metrics={k: v for k, v in m.items() if v is not None})


def _expand_baseline(rec):
    """src.baselines result JSON ({'variant','char','rq3','majority','tfidf_lr'}) -> flat records."""
    if "tfidf_lr" not in rec:
        return [rec]
    tag = "rq3" if rec.get("rq3") else "base"
    out = []
    for name, part in (("majority", rec.get("majority", {})),
                       ("tfidf_lr" + ("+char" if rec.get("char") else ""), rec["tfidf_lr"])):
        out.append(dict(model=name, variant=rec.get("variant"), tag=tag, seed=0,
                        val=part.get("val"), test=part.get("test")))
    return out


def load(runs, baselines):
    recs = []
    for src, d in (("neural", runs), ("baseline", baselines)):
        for f in sorted(glob.glob(str(Path(d) / "*.json"))):
            try:
                data = json.loads(Path(f).read_text(encoding="utf-8"))
            except Exception as e:
                print(f"skip {f}: {e}", file=sys.stderr)
                continue
            recs_in = data if isinstance(data, list) else [data]
            if src == "baseline":
                recs_in = [r2 for r in recs_in if isinstance(r, dict) for r2 in _expand_baseline(r)]
            for rec in recs_in:
                n = _norm(rec, src) if isinstance(rec, dict) else None
                if n is None or not n["metrics"]:
                    print(f"skip {f}: unrecognised format", file=sys.stderr)
                    continue
                recs.append(n)
    return recs


def _fmt(vals):
    if not vals:
        return "-"
    v = np.array(vals) * 100
    return f"{v.mean():.2f} {PM} {v.std(ddof=1):.2f}" if len(v) > 1 else f"{v.mean():.2f}"


def to_markdown(recs):
    groups = defaultdict(list)
    for r in recs:
        groups[(r["src"], r["model"], r["variant"], r["tag"])].append(r)
    present = {k for r in recs for k in r["metrics"]}
    allc = [c for c in ("val acc", "val macro-F1") if c in present]
    for t in sorted({k.rsplit(" ", 1)[0] for k in present if k.startswith("test")}):
        allc += [c for c in (f"{t} acc", f"{t} macro-F1") if c in present]

    def head(c):
        return f"{c} (%) [{'VAL split' if c.startswith('val') else 'TEST split: ' + c.split()[0]}]"

    cols = ["source", "model", "variant", "tag", "n seeds"] + [head(c) for c in allc]
    lines = [f"Values are mean {PM} std (sample std, ddof=1) over seeds, in %. VAL numbers drive early stopping / "
             "model selection; TEST numbers are held-out, computed at the best-val checkpoint.", "",
             "| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for key in sorted(groups):
        rs = groups[key]
        row = list(key) + [str(len({r["seed"] for r in rs}))]
        row += [_fmt([r["metrics"][c] for r in rs if c in r["metrics"]]) for c in allc]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--baselines", default="reports/baselines")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    md = to_markdown(load(a.runs, a.baselines))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(md, encoding="utf-8")
    else:
        sys.stdout.buffer.write(md.encode("utf-8"))


if __name__ == "__main__":
    main()
