"""Phase 4 training loop. Usage:
python -m src.train --model cnn --variant output --seeds 0 1 2 [--eval-test] [--rq3] [--lowercase] ...
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader

from src.models import build_model
from src.preprocess import TextDataset, Vocab, build_vocab_from_df, collate_fn
from src.scaling import subsample_ids

N_CLASSES = 5
DEFAULTS = dict(
    model="cnn", variant="output", seed=0, emb=128, hid=128, nf=100, dropout=0.5,
    lowercase=False, max_epochs=20, patience=3, data="data/processed/raid_main.parquet",
    rq3=False, eval_test=False, tag="base", batch_size=64, lr=1e-3, weight_decay=1e-5,
    clip=1.0, cache_dir="data/processed/cache", runs_dir="runs", ckpt_dir=None,
    save_preds=True, verbose=True, threads=8, device="auto",
    emb_dropout=0.0,      # dropout on the embedding output (0 = off, bit-identical to older runs)
    max_out_len=256,      # output-token cap (prompt cap stays 64)
    format_norm=False,    # format-normalise the generation (strip markdown, drop <NL>) before tokenising
    data_fraction=1.0,    # nested seeded fraction of TRAIN source_ids (val/test never touched)
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name="auto"):
    """'auto' -> cuda if available else cpu; otherwise torch.device(name)."""
    if name in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.device(name)
    if d.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"--device {name} requested but CUDA is not available")
    return d


def _test_names(rq3):
    return ["test_indomain", "test_code"] if rq3 else ["test"]


def load_splits(cfg):
    """-> dict name -> dataframe for train, val and all test sets."""
    df = pd.read_parquet(cfg["data"])
    col = "split_rq3" if cfg["rq3"] else "split"
    out = {}
    for n in ["train", "val"] + _test_names(cfg["rq3"]):
        g = df[df[col] == n].reset_index(drop=True)
        if n == "train":
            g = g.assign(split="train")  # vocab guard in build_vocab_from_df checks split == 'train'
            if cfg.get("data_fraction", 1.0) < 1.0:
                keep = set(subsample_ids(g["source_id"].unique(), cfg["data_fraction"]))
                g = g[g["source_id"].astype(str).isin(keep)].reset_index(drop=True)
        out[n] = g
    return out


def _cache_path(cfg):
    p = Path(cfg["data"])
    st = p.stat()
    key = (f"{p.stem}_{'rq3' if cfg['rq3'] else 'main'}_{cfg['variant']}_"
           f"{'lc' if cfg['lowercase'] else 'cs'}_{st.st_size}_{int(st.st_mtime)}")
    # non-default options extend the key (defaults keep the historical key, so old caches stay valid)
    if cfg.get("max_out_len", 256) != 256:
        key += f"_mo{cfg['max_out_len']}"
    if cfg.get("format_norm"):
        key += "_fn"
    if cfg.get("data_fraction", 1.0) < 1.0:
        key += f"_fr{cfg['data_fraction']:g}"
    return Path(cfg["cache_dir"]) / f"{key}.pkl"


def get_data(cfg):
    """Vocab (TRAIN split only) + encoded datasets per split, cached per (variant, lowercase)."""
    cp = _cache_path(cfg)
    if cp.exists():
        try:
            c = pickle.loads(cp.read_bytes())
            return Vocab(c["itos"]), c["ds"], c["domains"]
        except Exception:
            pass
    splits = load_splits(cfg)  # all splits are encoded so the cache is reusable; test is only *used* if eval_test
    fn = bool(cfg.get("format_norm"))
    vocab = build_vocab_from_df(splits["train"], cfg["lowercase"], normalise=fn)
    ds = {n: TextDataset(g, vocab, cfg["variant"], cfg["lowercase"], max_output=int(cfg.get("max_out_len", 256)),
                         normalise=fn) for n, g in splits.items()}
    domains = {n: g["domain"].tolist() for n, g in splits.items()}
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_bytes(pickle.dumps(dict(itos=vocab.itos, ds=ds, domains=domains)))
    return vocab, ds, domains


@torch.no_grad()
def evaluate(model, loader, loss_fn=None, device=None):
    device = device or torch.device("cpu")
    model.eval()
    ys, ps, probs, tot, n = [], [], [], 0.0, 0
    for x, l, y in loader:
        y_d = y.to(device, non_blocking=True)
        lo = model(x.to(device, non_blocking=True), l)  # lengths stay on CPU
        if loss_fn is not None:
            tot += loss_fn(lo, y_d).item() * len(y)
        n += len(y)
        probs.append(torch.softmax(lo, 1).cpu().numpy())
        ps.append(lo.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(probs), tot / max(n, 1)


def metrics(y, p, domains):
    d = np.asarray(domains)
    dom = {str(k): float((p[d == k] == y[d == k]).mean()) for k in sorted(set(d))}
    labels = list(range(N_CLASSES))
    return dict(acc=float((p == y).mean()),
                macro_f1=float(f1_score(y, p, average="macro", labels=labels, zero_division=0)),
                confusion=confusion_matrix(y, p, labels=labels).tolist(), domain_acc=dom)


def run(config):
    cfg = {**DEFAULTS, **config}
    device = resolve_device(cfg["device"])
    if device.type == "cpu":
        torch.set_num_threads(cfg["threads"])
    pin = device.type == "cuda"
    set_seed(cfg["seed"])
    vocab, ds, domains = get_data(cfg)
    g = torch.Generator()
    g.manual_seed(cfg["seed"])
    train_dl = DataLoader(ds["train"], cfg["batch_size"], shuffle=True, collate_fn=collate_fn, generator=g,
                          num_workers=0, pin_memory=pin)

    def mk(d):
        return DataLoader(d, 256, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=pin)

    val_dl = mk(ds["val"])
    model = build_model(cfg["model"], len(vocab), N_CLASSES, cfg["emb"], cfg["hid"], cfg["nf"], cfg["dropout"],
                        packed=device.type == "cuda", emb_dropout=cfg["emb_dropout"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    hist = dict(train_loss=[], val_loss=[], val_acc_per_epoch=[], val_macro_f1_per_epoch=[])
    best_f1, best_ep, best_state, bad = -1.0, -1, None, 0
    t0 = time.time()
    for ep in range(cfg["max_epochs"]):
        model.train()
        tot, n = 0.0, 0
        for x, l, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(model(x, l), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        y, p, _, vl = evaluate(model, val_dl, loss_fn, device)
        m = metrics(y, p, domains["val"])
        hist["train_loss"].append(tot / n)
        hist["val_loss"].append(vl)
        hist["val_acc_per_epoch"].append(m["acc"])
        hist["val_macro_f1_per_epoch"].append(m["macro_f1"])
        if cfg["verbose"]:
            print(f"[{cfg['model']}/{cfg['variant']}/s{cfg['seed']}] ep{ep + 1} loss={tot / n:.4f} "
                  f"val_acc={m['acc']:.4f} val_f1={m['macro_f1']:.4f}", flush=True)
        if m["macro_f1"] > best_f1:
            best_f1, best_ep, bad = m["macro_f1"], ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    train_time = time.time() - t0
    model.load_state_dict(best_state)
    y, p, val_prob, _ = evaluate(model, val_dl, device=device)
    vm = metrics(y, p, domains["val"])
    res = dict(config={**cfg, "device": str(device)}, vocab_size=len(vocab), n_params=n_params, train_time_s=train_time,
               epochs_run=len(hist["train_loss"]), **hist, best_epoch=best_ep + 1,
               val_acc=vm["acc"], val_macro_f1=vm["macro_f1"], val_confusion=vm["confusion"],
               val_domain_acc=vm["domain_acc"])
    # VAL predictions at the best-val checkpoint are always saved (lets a VAL-chosen ensemble weight be fit later)
    preds = dict(val_y=y, val_pred=p, val_prob=val_prob, val_source_id=np.array(ds["val"].source_ids))
    if cfg["eval_test"]:
        res["test"] = {}
        for name in _test_names(cfg["rq3"]):
            y, p, pr, _ = evaluate(model, mk(ds[name]), device=device)
            res["test"][name] = metrics(y, p, domains[name])
            preds[f"{name}_y"], preds[f"{name}_pred"], preds[f"{name}_prob"] = y, p, pr
            preds[f"{name}_source_id"] = np.array(ds[name].source_ids)
        if not cfg["rq3"]:
            t = res["test"]["test"]
            res.update(test_acc=t["acc"], test_macro_f1=t["macro_f1"], test_confusion=t["confusion"],
                       test_domain_acc=t["domain_acc"])
    if cfg["runs_dir"]:
        rd = Path(cfg["runs_dir"])
        rd.mkdir(parents=True, exist_ok=True)
        stem = f"{cfg['model']}_{cfg['variant']}_{cfg['tag']}_s{cfg['seed']}"
        (rd / f"{stem}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        if preds and cfg["save_preds"]:
            np.savez_compressed(rd / f"{stem}.npz", **preds)
        if cfg["ckpt_dir"]:
            Path(cfg["ckpt_dir"]).mkdir(parents=True, exist_ok=True)
            torch.save(best_state, Path(cfg["ckpt_dir"]) / f"{stem}.pt")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "bilstm"], required=True)
    ap.add_argument("--variant", choices=["prompt", "output", "both"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--eval-test", action="store_true")
    ap.add_argument("--rq3", action="store_true")
    ap.add_argument("--lowercase", action="store_true")
    ap.add_argument("--emb", type=int, default=128)
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--nf", type=int, default=100)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    ap.add_argument("--emb-dropout", type=float, default=0.0, help="dropout on embeddings (default 0 = off)")
    ap.add_argument("--max-out-len", type=int, default=256, help="output-token cap (default 256; cache key includes it)")
    ap.add_argument("--format-norm", action="store_true",
                    help="format-normalise the generation (strip markdown/bullets, collapse whitespace, no <NL>)")
    ap.add_argument("--data-fraction", type=float, default=1.0,
                    help="nested seeded fraction of TRAIN source_ids (val/test unchanged)")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--data", default=None)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--save-ckpt", action="store_true")
    ap.add_argument("--threads", type=int, default=8, help="CPU threads (ignored on CUDA)")
    ap.add_argument("--device", default="auto", help="auto (cuda if available else cpu), cpu, cuda, cuda:1")
    a = ap.parse_args(argv)
    data = a.data or ("data/processed/raid_rq3.parquet" if a.rq3 else "data/processed/raid_main.parquet")
    for s in a.seeds:
        cfg = dict(model=a.model, variant=a.variant, seed=s, emb=a.emb, hid=a.hid, nf=a.nf, dropout=a.dropout,
                   lowercase=a.lowercase, max_epochs=a.max_epochs, patience=a.patience, data=data, rq3=a.rq3,
                   eval_test=a.eval_test, tag=a.tag, runs_dir=a.runs_dir,
                   ckpt_dir="checkpoints" if a.save_ckpt else None, threads=a.threads, device=a.device,
                   weight_decay=a.weight_decay, emb_dropout=a.emb_dropout, max_out_len=a.max_out_len,
                   format_norm=a.format_norm, data_fraction=a.data_fraction)
        r = run(cfg)
        msg = f"seed {s}: best_ep={r['best_epoch']} val_acc={r['val_acc']:.4f} val_f1={r['val_macro_f1']:.4f}"
        for n, t in r.get("test", {}).items():
            msg += f" | {n} acc={t['acc']:.4f} f1={t['macro_f1']:.4f}"
        print(msg, flush=True)


if __name__ == "__main__":
    main()
