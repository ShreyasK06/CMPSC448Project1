"""Pure helpers of the RQ4 follow-ups: length-robust diversity features (t16) and token-deletion interventions (t17)."""
import pytest

from src.analysis.t16_feature_groups import ALL_NAMES, DIV, GROUPS, diversity_features, mattr
from src.analysis.t17_occlusion import make_variants
from src.analysis.t5_length import STYLO_FEATS
from src.preprocess import NL_ID, SPECIALS, Vocab


def test_mattr_extremes_and_short_text():
    assert mattr([]) == 0.0
    assert mattr(["a"] * 100) == pytest.approx(1 / 50)        # one distinct word in every 50-word window
    assert mattr([str(i) for i in range(100)]) == pytest.approx(1.0)  # all distinct
    assert mattr(["a", "b", "a", "b"]) == 0.5                 # shorter than the window: plain TTR


def test_mattr_is_length_robust_unlike_ttr():
    # a text repeating a 10-word pool: plain TTR keeps falling with length, MATTR stays put
    pool = [f"w{i}" for i in range(10)]
    short, long_ = pool * 10, pool * 40
    assert len(set(short)) / len(short) > 3 * len(set(long_)) / len(long_)
    assert abs(mattr(short) - mattr(long_)) < 1e-9


def test_diversity_features_shape_and_values():
    m, t100, h100 = diversity_features("one two two three")
    assert m == 0.75 and t100 == 0.75 and h100 == 0.5        # 3 types / 4 words; 'one' and 'three' occur once
    assert diversity_features("") == [0.0, 0.0, 0.0]
    assert diversity_features(None) == [0.0, 0.0, 0.0]
    assert len(DIV) == 3


def test_feature_groups_partition_the_22_stylometric_features():
    flat = [f for g in GROUPS.values() for f in g]
    assert sorted(flat) == sorted(STYLO_FEATS) and len(flat) == len(set(flat)) == 22
    assert ALL_NAMES[-1].startswith("diversity")


def _vocab():
    return Vocab(SPECIALS + ["Sure", ":", "the", "cat", "*", "sat", ".", "x"])


def test_occlusion_variants_delete_the_right_tokens():
    v = _vocab()

    def enc(s):
        return [NL_ID if t == "<NL>" else v.stoi[t] for t in s.split()]

    fn = dict(make_variants(v))
    ids = enc("Sure : the <NL> cat * sat .")
    assert fn["delete newline tokens (<NL>)"](ids) == enc("Sure : the cat * sat .")
    assert fn["delete markdown marker tokens (* # ` _ ~ >)"](ids) == enc("Sure : the <NL> cat sat .")
    assert fn["delete all formatting (newlines + markdown markers)"](ids) == enc("Sure : the cat sat .")
    opener = "delete boilerplate opener (Sure/Here/Certainly... up to first : . ! ?)"
    assert fn[opener](ids) == enc("the <NL> cat * sat .")
    assert fn[opener](enc("the cat sat .")) == enc("the cat sat .")       # no opener -> unchanged


def test_occlusion_position_controls():
    v = _vocab()
    fn = dict(make_variants(v))
    ids = [v.stoi["the"], v.stoi["cat"]] * 30                   # 60 tokens
    assert len(fn["delete the first 32 tokens"](ids)) == 28
    assert len(fn["control: delete the last 32 tokens"](ids)) == 28
    assert len(fn["control: delete 32 random tokens"](ids)) == 28
    short = ids[:20]
    assert fn["delete the first 32 tokens"](short) == short     # inputs <= 32 tokens are left alone
    assert fn["delete all formatting (newlines + markdown markers)"]([NL_ID]) == [1]  # never returns an empty sequence
