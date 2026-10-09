import json

import pytest

from src import baselines
from tests.synthetic import make_synthetic_df

VARS = ["prompt", "output", "both"]


@pytest.fixture(scope="module")
def df():
    d = make_synthetic_df(n_per_class=40, seed=0)
    # token present ONLY in val/test generations
    d.loc[d.split == "val", "generation"] += " zzvalonly zzvalonly"
    d.loc[d.split == "test", "generation"] += " zztestonly zztestonly"
    return d


@pytest.fixture(scope="module")
def parquet(df, tmp_path_factory):
    p = tmp_path_factory.mktemp("d") / "main.parquet"
    df.to_parquet(p)
    return str(p)


def test_runs_all_variants_no_test(parquet, tmp_path):
    res = baselines.main(["--data", parquet, "--variants", *VARS, "--out-dir", str(tmp_path)])
    assert set(res) == set(VARS)
    for tag, r in res.items():
        assert abs(r["majority"]["val"]["accuracy"] - 0.2) < 1e-9  # balanced -> 0.2
        assert "test" not in r["tfidf_lr"] and "test" not in r["majority"]
        assert set(r["val_curve"]) == {"0.01", "0.1", "1", "10", "100", "1000"}
        assert len(r["tfidf_lr"]["val"]["confusion_matrix"]) == 5
        assert r["tfidf_lr"]["val"]["per_domain_accuracy"]
        assert len(r["top_features"]) == 5
        data = json.loads((tmp_path / f"{tag}.json").read_text())
        assert data["best_C"] == r["best_C"]
        assert "test" not in data["tfidf_lr"]


def test_eval_test_adds_test(parquet, tmp_path):
    res = baselines.main(["--data", parquet, "--variants", "output", "--eval-test",
                          "--char", "--out-dir", str(tmp_path)])
    r = res["output_char"]
    assert "accuracy" in r["tfidf_lr"]["test"]["test"]
    assert (tmp_path / "output_char.json").exists()


def test_vectorizer_fit_on_train_only(df):
    splits = {s: df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test")}
    _, art = baselines.run_variant(splits, "output", eval_test=True, n_jobs=1)
    vocab = art["vectorizer"].vocabulary_
    assert "zzvalonly" not in vocab and "zztestonly" not in vocab
    assert "<NL>" in vocab and "," in vocab  # newline + punctuation tokens kept


def test_truncation():
    import pandas as pd
    d = pd.DataFrame({"prompt": ["w " * 100], "generation": ["g " * 300]})
    assert len(baselines.build_texts(d, "prompt")[0].split()) == 64
    assert len(baselines.build_texts(d, "output")[0].split()) == 256
    both = baselines.build_texts(d, "both")[0].split()
    assert len(both) == 64 + 1 + 256 and both[64] == "<SEP>"
    assert len(baselines.build_texts(d, "output", truncate=False)[0].split()) == 300
