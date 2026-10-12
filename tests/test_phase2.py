"""Tests for the phase-2 tooling: ext grids/selection, train flags (val probs, caps, cache key, fractions), aggregate merge,
report helpers ."""
import json
import pickle

import numpy as np
import pytest
import torch

from src.aggregate import load, merge_preserving, to_markdown
from src.analysis import t8_report as R
from src.analysis.stats import bootstrap_weights, cluster_confusions
from src.models import build_model
from src.phase2_spec import ABLATIONS, is_phase2_tag
from src.train import DEFAULTS, _cache_path, load_splits, run
from src.tune import ext_grid, select_best
from tests.synthetic import make_synthetic_df


# ---------------------------------------------------------------- T9 grids
def test_ext_grid_sizes_and_contents():
    cnn, lstm = ext_grid("cnn"), ext_grid("bilstm")
    assert len(cnn) == 12 and len(lstm) == 6
    assert {c["dropout"] for c in cnn} == {0.1, 0.2, 0.3} and {c["nf"] for c in cnn} == {200, 300}
    assert {c["weight_decay"] for c in cnn} == {0.0, 1e-4} and {c["emb"] for c in cnn} == {256}
    assert {(c["hid"], c["dropout"]) for c in lstm} == {(256, 0.3), (256, 0.5), (384, 0.5)}
    assert len(ext_grid("cnn", True)) == 24 and sum(c.get("emb_dropout") == 0.2 for c in ext_grid("cnn", True)) == 12
    assert len({json.dumps(c, sort_keys=True) for c in cnn}) == 12  # no duplicates


def test_select_best_uses_mean_over_seeds_not_seed0():
    s2 = [dict(params={"a": 1}, val_macro_f1_by_seed={"0": 0.70, "1": 0.60, "2": 0.60}),
          dict(params={"a": 2}, val_macro_f1_by_seed={"0": 0.66, "1": 0.67, "2": 0.68})]
    assert select_best(s2)["params"] == {"a": 2}


# ---------------------------------------------------------------- models / train flags
def test_emb_dropout_default_is_identity_and_active_when_set():
    m0 = build_model("cnn", 50, 5, emb=8, nf=4)
    m1 = build_model("cnn", 50, 5, emb=8, nf=4, emb_dropout=0.5)
    assert isinstance(m0.emb_drop, torch.nn.Identity) and isinstance(m1.emb_drop, torch.nn.Dropout)
    assert list(m0.state_dict()) == list(m1.state_dict())  # same parameters / checkpoint layout


def _cfg(tmp_path, **kw):
    df = make_synthetic_df(40, seed=0)
    p = tmp_path / "d.parquet"
    df.to_parquet(p)
    return dict(model="cnn", variant="output", data=str(p), max_epochs=1, emb=8, nf=4, cache_dir=str(tmp_path / "c"),
                runs_dir=str(tmp_path / "runs"), verbose=False, **kw), df


def test_cache_key_includes_new_options_and_keeps_old_key(tmp_path):
    cfg, _ = _cfg(tmp_path)
    full = {**DEFAULTS, **cfg}
    base = _cache_path(full)
    assert "_mo" not in base.name and "_fn" not in base.name and "_fr" not in base.name  # historical key unchanged
    assert "_mo64" in _cache_path({**full, "max_out_len": 64}).name
    assert _cache_path({**full, "format_norm": True}) != base
    assert _cache_path({**full, "data_fraction": 0.5}) != base
    assert _cache_path({**full, "max_out_len": 64}) != _cache_path({**full, "max_out_len": 128})


def test_run_saves_val_probs_and_test_probs(tmp_path):
    cfg, df = _cfg(tmp_path)
    r = run({**cfg, "eval_test": True})
    z = np.load(tmp_path / "runs" / "cnn_output_base_s0.npz")
    assert {"val_y", "val_pred", "val_prob", "val_source_id", "test_y", "test_pred", "test_prob",
            "test_source_id"} <= set(z.files)
    n_val = int((df.split == "val").sum())
    assert z["val_prob"].shape == (n_val, 5) and np.allclose(z["val_prob"].sum(1), 1, atol=1e-4)
    assert float((z["val_pred"] == z["val_y"]).mean()) == pytest.approx(r["val_acc"])


def test_max_out_len_and_format_norm_change_encoded_lengths(tmp_path):
    cfg, _ = _cfg(tmp_path)
    r = run({**cfg, "max_out_len": 6, "tag": "len6"})
    assert r["config"]["max_out_len"] == 6
    ((name, c),) = {p.name: pickle.loads(p.read_bytes()) for p in (tmp_path / "c").glob("*.pkl")}.items()
    assert "_mo6" in name and max(c["ds"]["train"].lengths) <= 6
    run({**cfg, "format_norm": True, "tag": "fmt"})
    assert len(list((tmp_path / "c").glob("*.pkl"))) == 2


def test_data_fraction_subsamples_train_only_and_nested(tmp_path):
    cfg, _ = _cfg(tmp_path)
    full = {**DEFAULTS, **cfg}
    s100, s50, s25 = (load_splits({**full, "data_fraction": f}) for f in (1.0, 0.5, 0.25))
    t100, t50, t25 = (set(s["train"].source_id) for s in (s100, s50, s25))
    assert t25 < t50 < t100
    for s in (s50, s25):
        assert set(s["val"].source_id) == set(s100["val"].source_id)
        assert set(s["test"].source_id) == set(s100["test"].source_id)
    assert not t50 & (set(s50["val"].source_id) | set(s50["test"].source_id))


# ---------------------------------------------------------------- aggregate
def test_phase2_tag_detection():
    assert all(is_phase2_tag(t) for t in ("abl_lc", "base_vp", "ext", "scaled", "scale100", "scalefrac25"))
    assert not any(is_phase2_tag(t) for t in ("base", "rq3"))
    assert all(is_phase2_tag(t) for _, t, _, _ in ABLATIONS)


def test_merge_preserving_keeps_sections_and_is_idempotent():
    old = "TABLE OLD\n\n| a |\n\n## Stats\nbody1\n\n## Other\nbody2\n"
    new = merge_preserving(old, "TABLE NEW\n")
    assert new.startswith("TABLE NEW") and "## Stats\nbody1" in new and "## Other\nbody2" in new and "TABLE OLD" not in new
    assert merge_preserving(new, "TABLE NEW\n") == new
    assert merge_preserving("", "T\n") == "T\n" and merge_preserving("only table\n", "T\n") == "T\n"


def test_aggregate_excludes_phase2_tags_from_main_table(tmp_path):
    def rec(tag):
        return dict(config=dict(model="cnn", variant="output", tag=tag, seed=0), val_acc=.6, val_macro_f1=.6)
    for t in ("base", "abl_lc", "ext"):
        (tmp_path / f"cnn_output_{t}_s0.json").write_text(json.dumps(rec(t)))
    recs = load(str(tmp_path), str(tmp_path / "nobl"))
    assert {r["tag"] for r in recs} == {"base"}
    assert {r["tag"] for r in load(str(tmp_path), str(tmp_path / "nobl"), include_phase2=True)} == {"base", "abl_lc", "ext"}
    assert "abl_lc" not in to_markdown(recs)


# ---------------------------------------------------------------- report helpers
def _ens(pred, y, sid):
    return dict(y=np.asarray(y), pred=np.asarray(pred), sid=np.asarray(sid, dtype=str), prob=None)


def test_paired_acc_diff_matches_direct_difference_and_ci_brackets_it():
    rng = np.random.default_rng(0)
    n_ids = 60
    sid = np.repeat([f"s{i:03d}" for i in range(n_ids)], 5)
    y = np.tile(np.arange(5), n_ids)
    a = np.where(rng.random(len(y)) < 0.8, y, (y + 1) % 5)
    b = np.where(rng.random(len(y)) < 0.7, y, (y + 1) % 5)
    A, B = _ens(a, y, sid), _ens(b, y, sid)
    W = bootstrap_weights(len(cluster_confusions(y, a, sid)[0]), 500, 0)
    d, ci = R.paired_acc_diff(A, B, W)
    assert d == pytest.approx((a == y).mean() - (b == y).mean())
    assert ci[0] <= d <= ci[1]
    assert R.paired_acc_diff(A, A, W)[0] == 0
    p, ao, bo = R.compare(A, B)
    assert ao == int(((a == y) & (b != y)).sum()) and bo == int(((b == y) & (a != y)).sum()) and 0 <= p <= 1


def test_compare_rejects_misaligned_rows():
    y = np.arange(5)
    with pytest.raises(ValueError):
        R.compare(_ens(y, y, list("abcde")), _ens(y, y, list("abcdf")))


def test_pm_and_fill_mcnemar_holm():
    assert R.pm([0.5]) == "50.00" and R.pm([0.5, 0.7]).startswith("60.00 ± 14.14") and R.pm([]) == "-"
    rows = [["x", (0.01, 10, 2)], ["y", (0.04, 8, 3)], ["z", None]]
    n = R._fill_mcnemar(rows, 1)
    assert n == 2 and rows[2][1] == "-" and "Holm p=0.02" in rows[0][1] and "10/2" in rows[0][1]


def test_sections_skip_when_no_runs(tmp_path):
    assert R.section_ablations(str(tmp_path)) is None
    assert R.section_ext(str(tmp_path), str(tmp_path)) is None
    assert R.section_scaling(str(tmp_path), str(tmp_path)) is None


def test_ablation_section_end_to_end_on_tiny_runs(tmp_path):
    """Real tiny training runs -> report section with split labels, McNemar vs base and mean +/- std."""
    df = make_synthetic_df(40, seed=0)
    p = tmp_path / "d.parquet"
    df.to_parquet(p)
    runs = tmp_path / "runs"
    base = dict(model="cnn", variant="output", data=str(p), max_epochs=1, emb=8, nf=4, eval_test=True,
                cache_dir=str(tmp_path / "c"), runs_dir=str(runs), verbose=False)
    for s in (0, 1):
        run({**base, "seed": s, "tag": "base"})
        run({**base, "seed": s, "tag": "abl_lc", "lowercase": True})
    sec = R.section_ablations(str(runs))
    assert sec.startswith("## Ablations") and "[VAL split]" in sec and "[TEST split: test]" in sec
    assert "(`base`)" in sec and "(`abl_lc`)" in sec and "±" in sec and "Holm p=" in sec
    out = tmp_path / "results.md"
    out.write_text("TABLE\n\n## Keep me\nx\n")
    for _ in range(2):  # idempotent
        R.write_all(str(runs), str(tmp_path / "t"), str(tmp_path / "b"), str(out), str(tmp_path / "an.md"),
                    fig_dir=str(tmp_path / "f"))
    txt = out.read_text()
    assert txt.count("## Ablations") == 1 and "## Keep me" in txt and txt.startswith("TABLE")
