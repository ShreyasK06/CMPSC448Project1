import pytest
import torch

from src.preprocess import (NL, PAD, SEP, UNK, TextDataset, Vocab, build_vocab_from_df, collate_fn,
                            encode_variant, pad_batch, run_checks, tokenize)
from tests.synthetic import make_synthetic_df


@pytest.fixture(scope="module")
def df():
    return make_synthetic_df(n_per_class=40, seed=0)  # 200 rows


@pytest.fixture(scope="module")
def vocab(df):
    return build_vocab_from_df(df[df.split == "train"])


# ---- tokenizer
def test_punct_split_and_contraction():
    assert tokenize("Hello, world!") == ["Hello", ",", "world", "!"]
    assert tokenize("I don't know.") == ["I", "don't", "know", "."]
    assert tokenize("a_b 12.5") == ["a_b", "12", ".", "5"]
    assert tokenize("'hi'") == ["'", "hi", "'"]


def test_newlines():
    assert tokenize("a\nb") == ["a", NL, "b"]
    assert tokenize("a\r\nb") == ["a", NL, "b"]
    assert tokenize("a\n\nb") == ["a", NL, NL, "b"]
    assert tokenize("a \t b") == ["a", "b"]
    assert tokenize("") == []


def test_casing():
    assert tokenize("Hello WORLD") == ["Hello", "WORLD"]
    assert tokenize("Hello WORLD", lowercase=True) == ["hello", "world"]
    assert tokenize("A\nB", lowercase=True) == ["a", NL, "b"]  # <NL> not lowercased


# ---- vocab
def test_specials_fixed_ids(vocab):
    assert [vocab.stoi[t] for t in (PAD, UNK, SEP, NL)] == [0, 1, 2, 3]
    assert vocab.itos[:4] == [PAD, UNK, SEP, NL]
    assert vocab.encode(["zzzunseen"]) == [1]


def test_min_freq_and_order():
    v = Vocab.build([["a", "a", "a", "b", "b", "c", "d", "d"]], min_freq=2)
    assert v.itos == [PAD, UNK, SEP, NL, "a", "b", "d"]  # c dropped; freq desc then alpha
    v2 = Vocab.build([["b", "a", "b", "a"]], min_freq=1)
    assert v2.itos[4:] == ["a", "b"]  # tie -> token asc


def test_max_size_cap():
    v = Vocab.build([[f"t{i}" for i in range(100)] * 2], min_freq=1, max_size=10)
    assert len(v) == 10 and v.itos[4] == "t0"
    assert len(Vocab.build([["a"] * 3], max_size=2)) == 4  # specials always kept


def test_nl_in_vocab_even_if_absent():
    assert Vocab.build([["a", "a"]]).stoi[NL] == 3


def test_train_only_vocab(df):
    train = df[df.split == "train"].copy()
    test = df[df.split == "test"].copy()
    test["generation"] = "ZQXUNIQUE hello"
    v = build_vocab_from_df(train)
    assert "ZQXUNIQUE" not in v.stoi
    ds = TextDataset(test, v, "output")
    assert 1 in ds.items[0]
    with pytest.raises(AssertionError):
        build_vocab_from_df(df)  # contains val/test


def test_save_load(vocab, tmp_path):
    vocab.save(tmp_path / "v.json")
    v2 = Vocab.load(tmp_path / "v.json")
    assert v2.itos == vocab.itos and v2.stoi == vocab.stoi


def test_pad_id_never_real(vocab):
    assert all(i != 0 for t, i in vocab.stoi.items() if t != PAD)
    assert vocab.itos[0] == PAD and len(set(vocab.itos)) == len(vocab.itos)


# ---- encode_variant
def test_truncation_and_sep():
    p, o = list(range(10, 210)), list(range(300, 800))
    ids, n = encode_variant(p, o, "prompt")
    assert ids == p[:64] and n == 64
    ids, n = encode_variant(p, o, "output")
    assert ids == o[:256] and n == 256
    ids, n = encode_variant(p, o, "both")
    assert len(ids) == 64 + 1 + 256 == n
    assert ids[64] == 2 and ids[:64] == p[:64] and ids[65:] == o[:256]


def test_short_padding_true_len():
    ids, n = encode_variant([], [7, 8], "output")
    assert ids == [7, 8, 0, 0, 0] and n == 2
    ids, n = encode_variant([], [7], "both")  # [SEP, 7] -> padded
    assert ids == [2, 7, 0, 0, 0] and n == 2


def test_never_empty():
    for v in ("prompt", "output", "both"):
        ids, n = encode_variant([], [], v)
        assert ids == [1, 0, 0, 0, 0] and n == 1
    ids, n = encode_variant([], [3, 4, 5, 6, 7], "prompt")  # empty prompt variant
    assert n == 1 and ids[0] == 1 and len(ids) == 5


def test_dataset_no_empty_all_variants(df, vocab):
    for variant in ("prompt", "output", "both"):
        ds = TextDataset(df, vocab, variant)
        assert len(ds) == len(df)
        assert min(ds.lengths) >= 1 and all(len(x) >= 5 for x in ds.items)
        assert max(ds.lengths) <= (64 if variant == "prompt" else 256 if variant == "output" else 321)


# ---- batching
def test_pad_batch_and_collate(df, vocab):
    x, l = pad_batch([[5, 6], [1, 2, 3, 4, 5, 6, 7]])
    assert x.shape == (2, 7) and l.tolist() == [2, 7] and x[0, 2:].sum() == 0
    x, l = pad_batch([[5]])
    assert x.shape == (1, 5) and l.tolist() == [1]
    ds = TextDataset(df.head(8), vocab, "both")
    xb, lb, yb = collate_fn([ds[i] for i in range(8)])
    assert xb.shape[0] == lb.shape[0] == yb.shape[0] == 8 and xb.shape[1] >= 5
    assert xb.dtype == lb.dtype == yb.dtype == torch.long
    assert ((xb != 0).sum(1) == lb).all()
    dl = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=collate_fn)
    assert next(iter(dl))[0].shape[0] == 4


# ---- run_checks
def _dss(df, vocab):
    return {s: TextDataset(df[df.split == s], vocab, "both") for s in ("train", "val", "test")}


def test_run_checks_clean(df, vocab):
    r = run_checks(df, _dss(df, vocab), verbose=False)
    assert r["empty_ok"] and r["disjoint_ok"]
    assert not any(r["imbalance_flags"].values())
    assert all(sum(c.values()) > 0 for c in r["class_counts"].values())


def test_run_checks_detects_overlap(df, vocab, capsys):
    bad = df.copy()
    i = bad.index[bad.split == "test"][0]
    bad.loc[i, "source_id"] = bad[bad.split == "train"].source_id.iloc[0]
    r = run_checks(bad, _dss(bad, vocab))
    assert not r["disjoint_ok"] and r["overlaps"]["test/train"] >= 1
    assert "FAIL" in capsys.readouterr().out


def test_run_checks_flags_imbalance(df, vocab):
    skew = df[~((df.split == "train") & (df.label == 0) & (df.index % 2 == 0))]
    r = run_checks(skew, {"train": TextDataset(skew[skew.split == "train"], vocab, "both")}, verbose=False)
    assert r["imbalance_flags"]["train"]
