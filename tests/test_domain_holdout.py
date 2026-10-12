"""Domain-holdout split rules (RQ3): no leakage of the held-out domain into train / val / vocabulary."""
import pandas as pd
import pytest

from src.domain_holdout import HOLDOUT, holdout_splits
from src.phase2_spec import is_phase2_tag
from src.train import DEFAULTS, _cache_path, _test_names, load_splits


def _frame():
    rows = []
    for d in ("a", "b", "c", "d"):
        for i in range(10):
            sp = "train" if i < 6 else ("val" if i < 8 else "test")
            for lab in range(5):
                rows.append(dict(domain=d, split=sp, source_id=f"{d}{i}", label=lab, model=str(lab),
                                 prompt="p", generation=f"{d} text {lab}"))
    return pd.DataFrame(rows)


def test_default_seen_domains_and_sizes():
    s = holdout_splits(_frame(), "c")
    assert set(s) == {"train", "val", HOLDOUT}
    assert set(s["train"]["domain"]) == {"a", "b", "d"} and set(s["val"]["domain"]) == {"a", "b", "d"}
    assert len(s["train"]) == 3 * 6 * 5 and len(s["val"]) == 3 * 2 * 5
    assert set(s[HOLDOUT]["domain"]) == {"c"} and len(s[HOLDOUT]) == 10 * 5  # every split of the held-out domain
    assert set(s["train"]["split"]) == {"train"}


def test_no_source_id_overlap_and_original_splits_kept():
    s = holdout_splits(_frame(), "a")
    assert not set(s["train"]["source_id"]) & set(s[HOLDOUT]["source_id"])
    assert not set(s["val"]["source_id"]) & set(s[HOLDOUT]["source_id"])
    assert set(s[HOLDOUT]["split"]) == {"train", "val", "test"}  # original split column preserved for paired controls


def test_grouped_train_domains():
    s = holdout_splits(_frame(), "d", ["a", "b"])
    assert set(s["train"]["domain"]) == {"a", "b"} and set(s["val"]["domain"]) == {"a", "b"}


def test_bad_arguments():
    f = _frame()
    with pytest.raises(ValueError):
        holdout_splits(f, "zzz")
    with pytest.raises(ValueError):
        holdout_splits(f, "a", ["a", "b"])
    with pytest.raises(ValueError):
        holdout_splits(f, "a", ["b", "nope"])


def test_train_load_splits_and_cache_keys(tmp_path):
    p = tmp_path / "x.parquet"
    _frame().to_parquet(p)
    cfg = {**DEFAULTS, "data": str(p), "holdout_domain": "b"}
    s = load_splits(cfg)
    assert _test_names(cfg) == [HOLDOUT] and set(s) == {"train", "val", HOLDOUT}
    assert set(s["train"]["domain"]) == {"a", "c", "d"}
    k1 = _cache_path(cfg)
    k2 = _cache_path({**cfg, "holdout_domain": "c"})
    k3 = _cache_path({**cfg, "train_domains": ["a", "c"]})
    k0 = _cache_path({**DEFAULTS, "data": str(p)})
    assert len({k0, k1, k2, k3}) == 4


def test_holdout_tags_excluded_from_main_table():
    assert is_phase2_tag("ho_news") and is_phase2_tag("hog_poetry") and not is_phase2_tag("base")
