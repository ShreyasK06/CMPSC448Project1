"""Phase 2 text preprocessing: tokenizer, vocabulary, RQ2 input variants, datasets, checks."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset

PAD, UNK, SEP, NL = "<PAD>", "<UNK>", "<SEP>", "<NL>"
PAD_ID, UNK_ID, SEP_ID, NL_ID = 0, 1, 2, 3
SPECIALS = [PAD, UNK, SEP, NL]
MIN_LEN = 5  # CNN max kernel size

# Order matters: CRLF / LF newline -> one token; word (letters/digits/underscore, with
# internal apostrophes so "don't" / "it's" stay single tokens); else any single
# non-whitespace char (punctuation/symbol) is its own token. Lone '\r' and other whitespace is dropped.
_TOKEN_RE = re.compile(r"\r\n|\n|\w+(?:['’]\w+)*|[^\s\w]")


# Format normaliser (shared by the neural format-norm ablation and src/analysis/t6_ablate.py).
_MD_LINE = re.compile(r"^[ \t]*(?:#{1,6}[ \t]+|>[ \t]?|[-*+•][ \t]+|\d+[.)][ \t]+)", re.M)
_MD_INLINE = re.compile(r"[*_`~]+")


def format_normalise(text) -> str:
    """Strip markdown (heading/quote/bullet/numbered-list markers, emphasis chars), turn every newline into a space and
    collapse whitespace, so no <NL> token and no list/markdown structure survives."""
    t = text if isinstance(text, str) else ""
    t = _MD_LINE.sub("", t)
    t = _MD_INLINE.sub("", t)
    return " ".join(t.split())


def tokenize(text: str, lowercase: bool = False) -> list[str]:
    """Regex tokenizer. Newlines -> '<NL>' ('\\r\\n' counts as one); words are \\w+ runs with
    internal apostrophes kept ("don't" is one token; a leading/trailing quote is split off);
    every other non-space char is its own token; whitespace dropped; casing kept unless lowercase."""
    if not isinstance(text, str):
        return []
    if lowercase:
        text = text.lower()
    return [NL if m in ("\n", "\r\n") else m for m in _TOKEN_RE.findall(text)]


class Vocab:
    """Token vocabulary. Specials: <PAD>=0, <UNK>=1, <SEP>=2, <NL>=3.

    IMPORTANT: must be built on the TRAIN split only (never val/test) to avoid leakage;
    val/test tokens unseen in train map to <UNK>.
    """

    def __init__(self, itos: list[str] | None = None):
        self.itos: list[str] = list(itos) if itos is not None else list(SPECIALS)
        assert self.itos[: len(SPECIALS)] == SPECIALS, "specials must be at fixed ids"
        self.stoi: dict[str, int] = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def build(cls, token_lists, min_freq: int = 2, max_size: int = 30000) -> "Vocab":
        """Keep tokens with freq >= min_freq, ordered by (freq desc, token asc), capped so that
        len(vocab) <= max_size including the 4 specials (specials are always kept)."""
        counts: Counter = Counter()
        for toks in token_lists:
            counts.update(toks)
        for s in SPECIALS:
            counts.pop(s, None)
        kept = sorted((t for t, c in counts.items() if c >= min_freq), key=lambda t: (-counts[t], t))
        kept = kept[: max(0, max_size - len(SPECIALS))]
        return cls(SPECIALS + kept)

    def encode(self, tokens) -> list[int]:
        g = self.stoi.get
        return [g(t, UNK_ID) for t in tokens]

    def __len__(self) -> int:
        return len(self.itos)

    def save(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"itos": self.itos}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Vocab":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["itos"])


def encode_variant(prompt_ids, output_ids, variant, max_prompt=64, max_output=256, min_len=MIN_LEN):
    """RQ2 input variants. 'prompt': prompt[:max_prompt]; 'output': output[:max_output];
    'both': prompt[:max_prompt] + [SEP] + output[:max_output]. Returns (ids, true_len).
    Empty content -> [UNK]; sequences shorter than min_len are right-padded with PAD, but
    true_len = max(1, number of non-PAD tokens)."""
    p, o = list(prompt_ids)[:max_prompt], list(output_ids)[:max_output]
    if variant == "prompt":
        ids = p
    elif variant == "output":
        ids = o
    elif variant == "both":
        ids = p + [SEP_ID] + o if (p or o) else []
    else:
        raise ValueError(f"unknown variant {variant!r}")
    if not ids:
        ids = [UNK_ID]
    true_len = max(1, sum(1 for i in ids if i != PAD_ID))
    if len(ids) < min_len:
        ids = ids + [PAD_ID] * (min_len - len(ids))
    return ids, true_len


def pad_batch(seqs, pad_to=None):
    """Right-pad with 0. T = max(longest, 5, pad_to). Returns (LongTensor [B,T], LongTensor lengths)
    where lengths = max(1, non-pad count)."""
    T = max([len(s) for s in seqs] + [MIN_LEN, pad_to or 0])
    x = torch.zeros(len(seqs), T, dtype=torch.long)
    lengths = torch.empty(len(seqs), dtype=torch.long)
    for i, s in enumerate(seqs):
        if len(s):
            x[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        lengths[i] = max(1, sum(1 for t in s if t != PAD_ID))
    return x, lengths


def _text(v) -> str:
    return v if isinstance(v, str) else ""


class TextDataset(Dataset):
    """Pre-tokenizes + encodes once. __getitem__ -> (ids list, length, label)."""

    def __init__(self, df, vocab: Vocab, variant="both", lowercase=False,
                 max_prompt=64, max_output=256, min_len=MIN_LEN, normalise=False):
        self.variant, self.lowercase = variant, lowercase
        self.normalise = normalise  # format-normalise the generation (not the prompt) before tokenising
        self.items, self.lengths, self.labels, self.source_ids = [], [], [], []
        self.raw_prompt_len, self.raw_output_len = [], []
        for pr, gen, lab, sid in zip(df["prompt"], df["generation"], df["label"], df["source_id"]):
            gen = format_normalise(gen) if normalise else gen
            pt, gt = tokenize(_text(pr), lowercase), tokenize(_text(gen), lowercase)
            ids, n = encode_variant(vocab.encode(pt), vocab.encode(gt), variant,
                                    max_prompt, max_output, min_len)
            self.items.append(ids)
            self.lengths.append(n)
            self.labels.append(int(lab))
            self.source_ids.append(sid)
            self.raw_prompt_len.append(len(pt))
            self.raw_output_len.append(len(gt))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i], self.lengths[i], self.labels[i]


def collate_fn(batch):
    """-> x [B,T] (PAD=0), lengths [B], y [B]."""
    x, lengths = pad_batch([b[0] for b in batch])
    y = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return x, lengths, y


def build_vocab_from_df(df_train, lowercase=False, min_freq=2, max_size=30000,
                        fields=("prompt", "generation"), normalise=False) -> Vocab:
    """Build vocab from the train split's prompt + generation tokens."""
    if "split" in df_train.columns:
        assert (df_train["split"] == "train").all(), "vocab must be built on split == 'train' only"

    def gen():
        for f in fields:
            for t in df_train[f]:
                if normalise and f == "generation":
                    t = format_normalise(t)
                yield tokenize(_text(t), lowercase)

    return Vocab.build(gen(), min_freq=min_freq, max_size=max_size)


def run_checks(df, datasets_by_split, ratio_limit=1.1, verbose=True) -> dict:
    """(1) no empty sequences, (2) source_id sets disjoint across splits, (3) per-split class
    counts, flagged when max/min > ratio_limit. Returns a dict and prints a summary."""
    res = {"min_length": {}, "empty_ok": True, "overlaps": {}, "disjoint_ok": True,
           "class_counts": {}, "class_ratio": {}, "imbalance_flags": {}}
    for s, ds in datasets_by_split.items():
        m = min(ds.lengths) if len(ds) else 1
        res["min_length"][s] = m
        if m < 1 or any(len(x) == 0 for x in ds.items):
            res["empty_ok"] = False
    sids = {s: set(g["source_id"]) for s, g in df.groupby("split")}
    names = sorted(sids)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ov = sids[a] & sids[b]
            if ov:
                res["overlaps"][f"{a}/{b}"] = len(ov)
                res["disjoint_ok"] = False
    for s, ds in datasets_by_split.items():
        cnt = Counter(ds.labels)
        counts = {c: cnt.get(c, 0) for c in range(5)}
        res["class_counts"][s] = counts
        lo, hi = min(counts.values()), max(counts.values())
        ratio = float("inf") if lo == 0 else hi / lo
        res["class_ratio"][s] = ratio
        res["imbalance_flags"][s] = ratio > ratio_limit
    if verbose:
        print(f"  [1] no empty sequences: {'OK' if res['empty_ok'] else 'FAIL'} (min len {res['min_length']})")
        print(f"  [2] source_id disjoint across splits: "
              f"{'OK' if res['disjoint_ok'] else 'FAIL ' + str(res['overlaps'])}")
        for s in res["class_counts"]:
            flag = "FLAG imbalance" if res["imbalance_flags"][s] else "ok"
            print(f"  [3] {s:5s} class counts {list(res['class_counts'][s].values())} "
                  f"max/min={res['class_ratio'][s]:.2f} -> {flag}")
    return res


def _main(argv=None):
    import numpy as np
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/raid_main.parquet")
    ap.add_argument("--out", default="data/processed/vocab.json")
    ap.add_argument("--lowercase", action="store_true")
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--max-size", type=int, default=30000)
    a = ap.parse_args(argv)
    df = pd.read_parquet(a.data)
    vocab = build_vocab_from_df(df[df["split"] == "train"], a.lowercase, a.min_freq, a.max_size)
    vocab.save(a.out)
    print(f"vocab size: {len(vocab)} (lowercase={a.lowercase}) -> {a.out}")
    splits = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    for s, g in splits.items():  # OOV over all tokens (incl. <NL>, which is always in vocab)
        tot = unk = 0
        for f in ("prompt", "generation"):
            for t in g[f]:
                ids = vocab.encode(tokenize(_text(t), a.lowercase))
                tot += len(ids)
                unk += sum(1 for i in ids if i == UNK_ID)
        print(f"OOV rate [{s}]: {unk / max(tot, 1):.4%} ({unk}/{tot} tokens)")
    for variant in ("prompt", "output", "both"):
        print(f"\n=== variant: {variant} ===")
        dss = {s: TextDataset(g, vocab, variant, a.lowercase) for s, g in splits.items()}
        for s, ds in dss.items():
            raw_o, raw_p = np.array(ds.raw_output_len), np.array(ds.raw_prompt_len)
            L = np.array(ds.lengths)
            print(f"  {s}: n={len(ds)} final len mean={L.mean():.1f} med={np.median(L):.0f} max={L.max()} | "
                  f"raw output tok mean={raw_o.mean():.1f} med={np.median(raw_o):.0f} | "
                  f"raw prompt tok mean={raw_p.mean():.1f} | "
                  f"truncated: output>256 {np.mean(raw_o > 256):.1%}, prompt>64 {np.mean(raw_p > 64):.1%}")
        run_checks(df, dss)


if __name__ == "__main__":
    _main()
