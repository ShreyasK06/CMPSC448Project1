"""Synthetic dataframe matching the processed RAID schema, for smoke tests."""
import numpy as np
import pandas as pd

MODELS = ["chatgpt", "cohere-chat", "llama-chat", "mistral-chat", "mpt-chat"]
_WORDS = ("the a of and to in is that it for was on are with as his they at be this from have or by one "
          "had not but what all were when we there can an your which their said if do will each about "
          "how up out them then she many some so these would other into has more her two like him see "
          "time could no make than first been its who now people my made over did down only way find "
          "use may water long little very after words called just where most know").split()
_PUNCT = [",", ".", "!", "?", ";", ":", "(", ")", '"', "-"]
DOMAINS = ["news", "books", "wiki", "reddit", "reviews"]
DECODING = ["greedy", "sampling"]


def _text(rng, n_words, bias, newlines=True):
    out = []
    for _ in range(n_words):
        if rng.random() > 0.15:
            w = _WORDS[rng.integers(len(_WORDS))]
        else:
            w = _WORDS[(bias * 7 + rng.integers(5)) % len(_WORDS)]
        if rng.random() < 0.05:
            w = w.capitalize()
        out.append(w)
        if rng.random() < 0.1:
            out.append(_PUNCT[rng.integers(len(_PUNCT))])
        if newlines and rng.random() < 0.03:
            out.append("\n" if rng.random() < 0.7 else "\r\n")
    return " ".join(out)


def make_synthetic_df(n_per_class=40, seed=0):
    """Balanced synthetic df with the processed schema (label order chatgpt=0, cohere-chat=1,
    llama-chat=2, mistral-chat=3, mpt-chat=4). n_per_class source_ids, each with one generation per
    class, split ~70/15/15 BY source_id (so no source_id crosses splits and per-split class counts
    are exactly equal). ~25% empty prompts, ~5% generations are 'Hi.'."""
    rng = np.random.default_rng(seed)
    n_train, n_val = int(round(n_per_class * 0.7)), int(round(n_per_class * 0.15))
    rows = []
    for s in range(n_per_class):
        sid = f"src{s:05d}"
        split = "train" if s < n_train else "val" if s < n_train + n_val else "test"
        prompt = "" if rng.random() < 0.25 else _text(rng, int(rng.integers(3, 120)), 0)
        title = _text(rng, 4, 0, newlines=False)
        dom = DOMAINS[int(rng.integers(len(DOMAINS)))]
        for lab, model in enumerate(MODELS):
            gen = "Hi." if rng.random() < 0.05 else _text(rng, int(rng.integers(5, 500)), lab + 1)
            rows.append(dict(
                id=f"{sid}_{model}", source_id=sid, model=model, label=lab, domain=dom,
                decoding=DECODING[int(rng.integers(2))],
                repetition_penalty=float(rng.choice([1.0, 1.2])), title=title, prompt=prompt,
                generation=gen, char_len=len(gen), word_len=len(gen.split()),
                prompt_char_len=len(prompt), prompt_word_len=len(prompt.split()), split=split))
    return pd.DataFrame(rows)
