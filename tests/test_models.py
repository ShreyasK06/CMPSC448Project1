import pickle

import pytest
import torch
from torch import nn

from src.models import BiLSTM, BiLSTMPacked, TextCNN, copy_packed_to_fast
from src.preprocess import TextDataset, build_vocab_from_df, collate_fn, pad_batch
from src.train import run
from tests.synthetic import make_synthetic_df

V, C = 50, 5


def _batch():
    # mixed lengths, including a length-1 sequence that pad_batch pads up to T>=5
    return pad_batch([[5, 6, 7, 8, 9, 10, 11, 12], [4], [3, 4, 5, 6, 7], [9] * 11])


@pytest.mark.parametrize("cls", [TextCNN, BiLSTM])
def test_forward_shapes_mixed_lengths(cls):
    x, l = _batch()
    assert l.tolist() == [8, 1, 5, 11]
    assert (x != 0).sum(1).tolist() == l.tolist()
    assert cls(V, C)(x, l).shape == (4, C)


def test_cnn_T5():
    x = torch.randint(1, V, (3, 5))
    assert TextCNN(V, C)(x, torch.full((3,), 5)).shape == (3, C)


def test_bilstm_padding_invariance():
    torch.manual_seed(0)
    m = BiLSTM(V, C).eval()
    seq = torch.randint(1, V, (7,))
    outs = []
    for T in (10, 50):
        x = torch.zeros(1, T, dtype=torch.long)
        x[0, :7] = seq
        with torch.no_grad():
            outs.append(m(x, torch.tensor([7])))
    assert torch.allclose(outs[0], outs[1], atol=1e-5)


def _mixed_batch():
    g = torch.Generator().manual_seed(3)
    seqs = [torch.randint(1, V, (n,), generator=g).tolist() for n in (1, 5, 17, 40, 23, 40)]
    return pad_batch(seqs)  # includes length 1, 5 and full T=40


def _pair():
    torch.manual_seed(0)
    packed = BiLSTMPacked(V, C, emb=16, hid=16).eval()
    fast = copy_packed_to_fast(packed, BiLSTM(V, C, emb=16, hid=16)).eval()
    return packed, fast


def test_fast_bilstm_matches_packed_logits():
    packed, fast = _pair()
    x, l = _mixed_batch()
    assert l.tolist()[:2] == [1, 5] and int(l.max()) == x.size(1)
    with torch.no_grad():
        assert torch.allclose(packed(x, l), fast(x, l), atol=1e-5)
    # short batch: min padding to 5
    x2, l2 = pad_batch([[3, 4], [7]])
    assert x2.size(1) == 5
    with torch.no_grad():
        assert torch.allclose(packed(x2, l2), fast(x2, l2), atol=1e-5)


def test_fast_bilstm_matches_packed_grads():
    packed, fast = _pair()
    x, l = _mixed_batch()
    y = torch.tensor([0, 1, 2, 3, 4, 0])
    lf = nn.CrossEntropyLoss()
    lf(packed(x, l), y).backward()
    lf(fast(x, l), y).backward()
    assert torch.allclose(packed.emb.weight.grad, fast.emb.weight.grad, atol=1e-5)
    assert torch.allclose(packed.rnn.weight_hh_l0_reverse.grad, fast.rnn_b.weight_hh_l0.grad, atol=1e-5)
    assert torch.allclose(packed.rnn.weight_ih_l0.grad, fast.rnn_f.weight_ih_l0.grad, atol=1e-5)


def test_mask_sum_equals_lengths_on_collated_batches():
    df = make_synthetic_df(20, seed=1)
    df["split"] = "train"
    vocab = build_vocab_from_df(df)
    for variant in ("prompt", "output", "both"):
        ds = TextDataset(df, vocab, variant)
        for i in range(0, len(ds), 16):
            x, l, y = collate_fn([ds[j] for j in range(i, min(i + 16, len(ds)))])
            assert torch.equal((x != 0).sum(1), l), variant


@pytest.mark.parametrize("model", ["cnn", "bilstm"])
@pytest.mark.parametrize("variant", ["prompt", "output", "both"])
def test_train_run_smoke(tmp_path, model, variant):
    df = make_synthetic_df(40, seed=0)
    # marker token (twice, so min_freq=2 would keep it) that appears only in val/test rows
    mask = df["split"] != "train"
    df.loc[mask, "generation"] = df.loc[mask, "generation"] + " zzvalonlytok zzvalonlytok"
    p = tmp_path / "d.parquet"
    df.to_parquet(p)
    cfg = dict(model=model, variant=variant, data=str(p), max_epochs=2, emb=16, hid=16, nf=8,
               cache_dir=str(tmp_path / "cache"), runs_dir=str(tmp_path / "runs"), verbose=False)
    r = run(cfg)
    assert not any(k.startswith("test") for k in r), "test metrics present without eval_test"
    assert not list((tmp_path / "runs").glob("*.npz"))
    assert len(r["train_loss"]) == 2 and 1 <= r["best_epoch"] <= 2
    assert 0 <= r["val_macro_f1"] <= 1 and r["n_params"] > 0
    c = pickle.loads(next((tmp_path / "cache").glob("*.pkl")).read_bytes())
    assert "zzvalonlytok" not in c["itos"]
    ref = build_vocab_from_df(df[df["split"] == "train"])
    assert c["itos"] == ref.itos and r["vocab_size"] == len(ref)
    if variant == "output":  # with eval_test, test metrics appear
        r2 = run({**cfg, "eval_test": True})
        assert {"test_acc", "test_macro_f1", "test_confusion", "test_domain_acc"} <= set(r2)
        assert (tmp_path / "runs" / f"{model}_output_base_s0.npz").exists()


@pytest.mark.parametrize("cls", [TextCNN, BiLSTM])
def test_overfit_tiny(cls):
    torch.manual_seed(0)
    x = torch.randint(1, V, (32, 12))
    l = torch.full((32,), 12)
    y = torch.randint(0, C, (32,))
    m = cls(V, C, p=0.0)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    lf = nn.CrossEntropyLoss()
    losses = []
    for _ in range(30):
        opt.zero_grad()
        loss = lf(m(x, l), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]


def test_build_model_selects_bilstm_path_by_device():
    from src.models import build_model
    assert isinstance(build_model("bilstm", V, C), BiLSTM)  # CPU default: fast two-LSTM path
    assert isinstance(build_model("bilstm", V, C, packed=True), BiLSTMPacked)  # CUDA path
    assert isinstance(build_model("cnn", V, C, packed=True), TextCNN)


def test_packed_path_trains_end_to_end_and_device_resolution(tmp_path):
    from src.train import resolve_device
    assert resolve_device("auto").type in ("cpu", "cuda")
    assert resolve_device("cpu").type == "cpu"
    df = make_synthetic_df(40, seed=0)
    p = tmp_path / "d.parquet"
    df.to_parquet(p)
    r = run(dict(model="bilstm", variant="output", data=str(p), max_epochs=1, emb=16, hid=16, device="cpu",
                 cache_dir=str(tmp_path / "c"), runs_dir=None, verbose=False))
    assert r["config"]["device"] == "cpu"
    # packed model on CPU still runs through the same training code path
    from src.train import evaluate
    m = BiLSTMPacked(V, C, emb=16, hid=16)
    x, l = _mixed_batch()
    assert m(x, l).shape == (x.size(0), C)
