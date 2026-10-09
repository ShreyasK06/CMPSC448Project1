"""Tests for src.scaling (nested subsampling, id drawing, expanded-set invariants) and the format normaliser."""
import numpy as np
import pandas as pd
import pytest

from src.analysis.t6_ablate import format_normalise as t6_format_normalise
from src.preprocess import NL, TextDataset, build_vocab_from_df, format_normalise, tokenize
from src.scaling import draw_new_ids, subsample_ids, verify_scaled
from tests.synthetic import make_synthetic_df


# ---------------------------------------------------------------- normaliser
def test_format_normalise_strips_markdown_and_newlines():
    t = "# Title\n\n* item one\n- item two\n1. first\n2) second\n> quoted\n**bold** and _it_ `code`\r\nend"
    out = format_normalise(t)
    assert out == "Title item one item two first second quoted bold and it code end"
    assert NL not in tokenize(out) and "\n" not in out


def test_format_normalise_handles_non_strings_and_is_idempotent():
    assert format_normalise(None) == "" and format_normalise(float("nan")) == ""
    t = "## H\n- a\n- b\n\ntext  here"
    assert format_normalise(format_normalise(t)) == format_normalise(t)


def test_t6_reuses_shared_normaliser():
    assert t6_format_normalise is format_normalise


def test_dataset_normalise_flag_removes_nl_and_changes_vocab():
    df = make_synthetic_df(40, seed=0)
    df["generation"] = df["generation"].map(lambda s: "# H\n- " + s + "\nend")
    tr = df[df.split == "train"]
    v_plain, v_norm = build_vocab_from_df(tr), build_vocab_from_df(tr, normalise=True)
    d_plain = TextDataset(tr, v_plain, "output")
    d_norm = TextDataset(tr, v_norm, "output", normalise=True)
    assert "#" in v_plain.stoi and "#" not in v_norm.stoi  # markdown marker gone from the normalised vocab
    assert 3 in {t for ids in d_plain.items for t in ids}  # <NL> id present without normalisation
    assert 3 not in {t for ids in d_norm.items for t in ids}


def test_max_output_cap_is_configurable():
    df = make_synthetic_df(20, seed=1)
    v = build_vocab_from_df(df[df.split == "train"])
    short = TextDataset(df, v, "output", max_output=8)
    full = TextDataset(df, v, "output")
    assert max(short.lengths) <= 8 and max(full.lengths) > 8


# ---------------------------------------------------------------- subsampling
IDS = [f"id{i:04d}" for i in range(2240)]


def test_subsample_sizes_and_determinism():
    assert [len(subsample_ids(IDS, f)) for f in (0.25, 0.5, 1.0)] == [560, 1120, 2240]
    assert subsample_ids(IDS, 0.25) == subsample_ids(list(reversed(IDS)), 0.25)  # input order irrelevant
    assert subsample_ids(IDS, 0.25, seed=1) != subsample_ids(IDS, 0.25, seed=2)


def test_subsample_nested_property():
    a, b, c = (set(subsample_ids(IDS, f)) for f in (0.25, 0.5, 1.0))
    assert a < b < c and c == set(IDS)


def test_subsample_rejects_bad_fraction():
    for f in (0, -0.1, 1.5):
        with pytest.raises(ValueError):
            subsample_ids(IDS, f)


# ---------------------------------------------------------------- id drawing
def _cands():
    return {"a": [f"a{i}" for i in range(30)], "b": [f"b{i}" for i in range(12)]}


def test_draw_new_ids_excludes_used_and_hits_cap():
    used = {f"a{i}" for i in range(5)} | {"b0", "b1"}
    picks = draw_new_ids(_cands(), used, {"a": 5, "b": 2}, cap=10, rng=np.random.default_rng(42), domains=["a", "b"])
    assert len(picks["a"]) == 5 and len(picks["b"]) == 8
    assert not (set(picks["a"]) | set(picks["b"])) & used
    assert len(set(picks["a"]) | set(picks["b"])) == 13  # no id drawn twice


def test_draw_new_ids_deterministic_and_errors():
    kw = dict(used_ids=set(), existing_by_domain={"a": 0, "b": 0}, cap=5, domains=["a", "b"])
    assert draw_new_ids(_cands(), rng=np.random.default_rng(42), **kw) == draw_new_ids(
        _cands(), rng=np.random.default_rng(42), **kw)
    with pytest.raises(ValueError):  # pool too small
        draw_new_ids(_cands(), {f"b{i}" for i in range(10)}, {"b": 0}, 5, np.random.default_rng(0), domains=["b"])


# ---------------------------------------------------------------- expanded-set invariants
def _expanded():
    main = make_synthetic_df(40, seed=0)
    new_rows = []
    base = main[main.split == "train"].copy()
    for k, sid in enumerate(sorted(base.source_id.unique())[:6]):
        g = base[base.source_id == sid].copy()
        g["source_id"] = f"new{k}"
        g["id"] = [f"new{k}_{j}" for j in range(len(g))]
        new_rows.append(g)
    scaled = pd.concat([main] + new_rows, ignore_index=True)
    return main, scaled


def test_verify_scaled_passes_on_valid_expansion():
    main, scaled = _expanded()
    res = verify_scaled(main, scaled)
    assert all(res.values()), {k: v for k, v in res.items() if not v}


def test_verify_scaled_detects_overlap_with_test_and_unpaired():
    main, scaled = _expanded()
    leak = scaled.copy()
    test_sid = main[main.split == "test"].source_id.iloc[0]
    idx = leak.index[leak.source_id == "new0"][0]
    leak.loc[idx, "source_id"] = test_sid  # a new train row now shares a test source_id
    res = verify_scaled(main, leak)
    assert not res["no new source_id in val/test"] and not res["each source_id in exactly one split"]
    unpaired = scaled.drop(scaled.index[scaled.source_id == "new1"][0])
    r2 = verify_scaled(main, unpaired)
    assert not r2["paired: 5 rows per source_id"] and not r2["classes exactly balanced in every split"]


def test_verify_scaled_detects_new_val_rows():
    main, scaled = _expanded()
    bad = scaled.copy()
    bad.loc[bad.source_id == "new2", "split"] = "val"
    res = verify_scaled(main, bad)
    assert not res["new rows are train only"] and not res["val rows identical to raid_main"]


def test_real_scaled_parquet_if_present():
    from pathlib import Path
    p, m = Path("data/processed/raid_scaled.parquet"), Path("data/processed/raid_main.parquet")
    if not (p.exists() and m.exists()):
        pytest.skip("raid_scaled.parquet not built")
    main, scaled = pd.read_parquet(m), pd.read_parquet(p)
    assert all(verify_scaled(main, scaled).values())
    assert list(scaled.columns) == list(main.columns)
    code = pd.read_parquet("data/processed/raid_rq3.parquet", columns=["source_id", "split_rq3"])
    assert set(code[code.split_rq3 == "test_code"].source_id).isdisjoint(set(scaled.source_id))
    assert (scaled[scaled.split == "train"].groupby("domain").source_id.nunique() > 700).all()
