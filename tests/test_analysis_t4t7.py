"""Tests for pure helpers in src.analysis.{common,t4_errors,t5_length,t6_ablate,t7_rq3}."""
import numpy as np

from src.analysis.common import length_bins, length_matched_indices, write_section
from src.analysis.t4_errors import confusion, row_normalise, tag_ngram, top_confused_pairs
from src.analysis.t5_length import LEN_FEATS, STYLO_FEATS, text_features
from src.analysis.t6_ablate import format_normalise, make_text_fn, mask_self_id
from src.analysis.t7_rq3 import oov_rate, predicted_distribution


def test_row_normalise_rows_sum_to_one_and_handles_empty_row():
    c = np.array([[2, 2], [0, 0]])
    r = row_normalise(c)
    assert np.allclose(r[0], [0.5, 0.5]) and np.allclose(r[1], [0, 0])


def test_top_confused_pairs_ranks_by_mutual_errors():
    conf = confusion(np.array([0, 0, 1, 2, 2, 2]), np.array([1, 1, 0, 0, 2, 1]), k=3)
    pairs = top_confused_pairs(conf, 2)
    assert pairs[0][:2] == (0, 1) and pairs[0][2] + pairs[0][3] == 3


def test_tag_ngram_heuristic():
    assert tag_ngram("<NL> <NL>") == "formatting"
    assert tag_ngram("Sure !") == "formatting"
    assert tag_ngram(", and") == "punctuation"
    assert tag_ngram(". .") == "punctuation"
    assert tag_ngram("recipe") == "content"


def test_length_bins_and_matched_sampling_equalises_histograms():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(3), 200)
    vals = np.concatenate([rng.normal(100 + 40 * c, 30, 200) for c in range(3)])
    bins = length_bins(vals, 5)
    assert bins.min() == 0 and bins.max() == 4
    idx = length_matched_indices(labels, bins, seed=1)
    h = [np.bincount(bins[idx][labels[idx] == c], minlength=5) for c in range(3)]
    assert all(np.array_equal(h[0], x) for x in h)
    assert np.array_equal(idx, length_matched_indices(labels, bins, seed=1))  # deterministic
    assert len(set(idx)) == len(idx)


def test_text_features_basic():
    f = text_features("Sure! Here is a list:\n- one\n- two\n\nDone.", ntok=300)
    assert set(STYLO_FEATS) <= set(f)
    assert f["starts_boilerplate"] == 1.0 and f["hit_cap"] == 1.0
    assert f["bullet_count"] == 2 and f["newline_count"] == 4 and f["blank_line_count"] == 1
    assert f["ends_terminal_punct"] == 1.0 and f["line_count"] == 4
    empty = text_features("", ntok=0)
    assert empty["word_count"] == 0 and all(np.isfinite(list(empty.values())))
    assert len(LEN_FEATS) == 4


def test_mask_self_id_and_format_normalise():
    assert "cohere" not in mask_self_id("Written by Cohere, as an AI language model.").lower()
    assert "Sure" not in mask_self_id("Sure! Here is a poem\nroses")
    t = format_normalise("# Title\n\n- item **bold**\n1. step\nplain")
    assert "\n" not in t and "#" not in t and "*" not in t and "step" in t and "plain" in t
    assert "<NL>" not in make_text_fn("format_norm")("a\nb\n\nc")
    assert "<NL>" in make_text_fn("K32")("a\nb")


def test_truncation_fn_respects_k():
    toks = " ".join(["w"] * 100)
    assert len(make_text_fn("K32")(toks).split()) == 32
    assert len(make_text_fn("Kfull")(toks).split()) == 100
    assert make_text_fn("lowercase")("ABC Def") == "abc def"


def test_predicted_distribution_and_oov():
    d = predicted_distribution(np.array([0, 0, 1, 4]))
    assert np.isclose(d.sum(), 100) and d[0] == 50
    assert np.isclose(oov_rate([["a", "b"], ["c"]], {"a"}), 2 / 3)
    assert oov_rate([], {"a"}) == 0


def test_write_section_is_idempotent(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("# T\n\n## Keep\n\nx\n", encoding="utf-8")
    for _ in range(2):
        write_section("## New", "## New\n\nbody", analysis_md=str(p))
    s = p.read_text(encoding="utf-8")
    assert s.count("## New") == 1 and "## Keep" in s
