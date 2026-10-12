"""Tests for the format normaliser and the output-length cap."""

from src.analysis.t6_ablate import format_normalise as t6_format_normalise
from src.preprocess import NL, TextDataset, build_vocab_from_df, format_normalise, tokenize
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
