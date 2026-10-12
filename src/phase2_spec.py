"""Single source of truth for the phase-2 experiment definitions (t8_report.py reports them).

Every phase-2 run is variant 'output' unless noted, 3 seeds (0,1,2), --eval-test, hyper-parameters = the stage-1 best of
reports/tuning/{model}_output.json (the same ones the base runs use) unless noted. Run files: runs/{model}_{variant}_{tag}_s{seed}.json
"""
from __future__ import annotations

# (model, tag, extra train.py flags, description). Baseline for each is tag 'base' of the same model, variant 'output'.
ABLATIONS = [
    ("cnn", "abl_lc", ["--lowercase"], "lowercase (vocab rebuilt lowercase)"),
    ("bilstm", "abl_lc", ["--lowercase"], "lowercase (vocab rebuilt lowercase)"),
    ("cnn", "abl_len64", ["--max-out-len", "64"], "output cap 64 tokens"),
    ("cnn", "abl_len128", ["--max-out-len", "128"], "output cap 128 tokens"),
    ("cnn", "abl_len512", ["--max-out-len", "512"], "output cap 512 tokens"),
    ("bilstm", "abl_len128", ["--max-out-len", "128"], "output cap 128 tokens"),
    ("bilstm", "abl_len512", ["--max-out-len", "512"], "output cap 512 tokens"),
    ("cnn", "abl_fmt", ["--format-norm"], "format-normalised input (no markdown/bullets, no <NL>, whitespace collapsed)"),
]
BASE_VP_TAG = "base_vp"     # base re-run that also stores val probabilities (replaces nothing)
EXT_TAG = "ext"             # final runs with the T9-chosen config
PHASE2_TAG_PREFIXES = ("abl_", "base_vp", "ext", "scale", "ho_", "hog_")  # ho_/hog_ = domain-holdout (RQ3) runs


def is_phase2_tag(tag: str) -> bool:
