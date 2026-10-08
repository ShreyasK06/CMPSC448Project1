"""Phase 1b: sampling, splits, coverage tables, data checks. CLI: python -m src.data

Reads data/interim/*.parquet|csv (produced by src/raid_stream.py) and writes
data/processed/raid_main.parquet, raid_rq3.parquet, reports/coverage/*, reports/data_checks.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
COVERAGE = ROOT / "reports" / "coverage"
REPORTS = ROOT / "reports"

MODELS = ["chatgpt", "cohere-chat", "llama-chat", "mistral-chat", "mpt-chat"]
LABELS = {m: i for i, m in enumerate(MODELS)}
DOMAINS = ["abstracts", "books", "news", "poetry", "recipes", "reddit", "reviews", "wiki"]
DECODING, REP = "sampling", "no"
N_PER_DOMAIN = 400
SEED = 42
COLS = ["id", "source_id", "model", "label", "domain", "decoding", "repetition_penalty", "title",
        "prompt", "generation", "char_len", "word_len", "prompt_char_len", "prompt_word_len", "split"]


def md_table(df: pd.DataFrame, index=True) -> str:
    d = df.reset_index() if index else df
    d = d.copy()
    d.columns = [" / ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in d.columns]
    rows = [[str(v) for v in r] for r in d.itertuples(index=False)]
    out = ["| " + " | ".join(d.columns) + " |", "|" + "---|" * len(d.columns)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def coverage_table(counts_path: Path) -> pd.DataFrame:
    c = pd.read_csv(counts_path, keep_default_na=False)
    c = c[c.attack == "none"]
    t = c.pivot_table(index=["model", "decoding", "repetition_penalty"], columns="domain",
                      values="n_rows", aggfunc="sum", fill_value=0)
    t["total"] = t.sum(axis=1)
    return t


def sample_paired(df: pd.DataFrame, domains: list, rng) -> pd.DataFrame:
    parts = []
    for dom in domains:
        d = df[(df.domain == dom) & (df.decoding == DECODING) & (df.repetition_penalty == REP)
               & df.model.isin(MODELS)]
        dup = d.duplicated(["source_id", "model"], keep=False)
        if dup.any():
            print(f"[warn] {dom}: {dup.sum()} duplicate (source_id, model) rows in "
                  f"{d[dup].groupby(['source_id', 'model']).ngroups} groups")
            raise AssertionError(f"duplicates in {dom}")
        k = d.groupby("source_id").model.nunique()
        cand = np.array(sorted(k[k == len(MODELS)].index))
        print(f"{dom}: {len(cand)} source_ids with all 5 models; no duplicate (source_id, model)")
        assert len(cand) >= N_PER_DOMAIN, dom
        pick = set(rng.choice(cand, size=N_PER_DOMAIN, replace=False))
        parts.append(d[d.source_id.isin(pick)])
    out = pd.concat(parts, ignore_index=True)
    assert not out.duplicated(["source_id", "model"]).any()
    assert (out.groupby("source_id").size() == len(MODELS)).all()
    return out


def add_stats(df: pd.DataFrame) -> pd.DataFrame:
    df["label"] = df.model.map(LABELS).astype(int)
    df["char_len"] = df.generation.str.len()
    df["word_len"] = df.generation.str.split().str.len()
    df["prompt_char_len"] = df.prompt.str.len()
    df["prompt_word_len"] = df.prompt.str.split().str.len()
    return df


def template(prompt: str, title: str) -> str:
    return prompt.replace(title, "<TITLE>") if title else prompt


def prompt_analysis(df: pd.DataFrame) -> str:
    g = df.groupby("source_id").prompt.nunique()
    frac = (g == 1).mean()
    lines = [f"- source_ids where all 5 prompts are identical: {(g == 1).sum()} / {len(g)} = {frac:.4f}",
             f"- distinct prompts per source_id (value counts): {g.value_counts().to_dict()}", "",
             "Distinct prompt templates per model (title replaced by `<TITLE>`):", ""]
    df = df.assign(tmpl=[template(p, t) for p, t in zip(df.prompt, df.title)])
    rows = []
    for m in MODELS:
        s = df[df.model == m]
        found = np.mean([t in p for t, p in zip(s.title, s.prompt) if t]) if (s.title != "").any() else float("nan")
        rows.append({"model": m, "n_templates": s.tmpl.nunique(), "n_raw_prompts": s.prompt.nunique(),
                     "title_found_in_prompt": round(float(found), 3)})
    lines += [md_table(pd.DataFrame(rows), index=False), "",
              "Template counts by (model, domain):", "",
              md_table(df.groupby(["model", "domain"]).tmpl.nunique().unstack()), "",
              "Two example prompts per model (truncated to 300 chars):", ""]
    for m in MODELS:
        for _, r in df[df.model == m].head(2).iterrows():
            lines.append(f"- {m} [{r.domain}]: `{r.prompt[:300]!r}`")
    return "\n".join(lines)


def split_by_source(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    ids = np.array(sorted(df.source_id.unique()))
    rng.shuffle(ids)
    n = len(ids)
    train_ids, val_ids = set(ids[:int(.7 * n)]), set(ids[int(.7 * n):int(.85 * n)])
    df["split"] = np.where(df.source_id.isin(train_ids), "train",
                           np.where(df.source_id.isin(val_ids), "val", "test"))
    return df


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    COVERAGE.mkdir(parents=True, exist_ok=True)
    md = ["# Coverage tables (attack == none; row counts over ALL rows of the full streamed CSVs)\n"]

    for name, f, title in [("train", "train_counts.csv", "(a) train split"),
                           ("extra", "extra_counts.csv", "(b) extra split")]:
        t = coverage_table(INTERIM / f)
        t.to_csv(COVERAGE / f"coverage_{name}.csv")
        md += [f"## {title}: model x (decoding, repetition_penalty) x domain\n", md_table(t), ""]

    cols = ["id", "source_id", "model", "decoding", "repetition_penalty", "domain", "title", "prompt", "generation"]
    tr = pd.read_parquet(INTERIM / "train_chat_none.parquet", columns=cols)
    tr = tr[tr.model.isin(MODELS) & (tr.generation.str.len() > 0)]
    ex = pd.read_parquet(INTERIM / "extra_code_chat_none.parquet", columns=cols)
    ex = ex[(ex.domain == "code") & ex.model.isin(MODELS) & (ex.generation.str.len() > 0)]

    sel = tr[(tr.decoding == DECODING) & (tr.repetition_penalty == REP)]
    ct = pd.crosstab(sel.model, sel.domain, margins=True)
    ct.to_csv(COVERAGE / "coverage_chosen.csv")
    k = sel.groupby(["domain", "source_id"]).model.nunique()
    full = (k == len(MODELS)).groupby("domain").sum().rename("source_ids_with_all_5_models").to_frame()
    full["source_ids_any"] = k.groupby("domain").size()
    full.to_csv(COVERAGE / "coverage_chosen_full_ids.csv")
    sel_ex = ex[(ex.decoding == DECODING) & (ex.repetition_penalty == REP)]
    kx = sel_ex.groupby("source_id").model.nunique()
    code_full = int((kx == len(MODELS)).sum())
    c_txt = [f"## (c) Chosen chat models x domain under ({DECODING}, rep_penalty={REP}) (rows in interim parquet)\n",
             md_table(ct), "", "Source_ids having all 5 models (text domains):", "", md_table(full), "",
             f"Code (extra split): {code_full} source_ids with all 5 models (of {kx.shape[0]} with any).", ""]
    md += c_txt
    (COVERAGE / "coverage.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(c_txt))
    del sel, sel_ex

    text = sample_paired(tr, DOMAINS, np.random.default_rng(SEED))
    code = sample_paired(ex, ["code"], np.random.default_rng(SEED))
    del tr, ex
    text, code = add_stats(text), add_stats(code)
    pa = prompt_analysis(text)
    print("\n# Prompt analysis (text)\n" + pa)
    pa_code = prompt_analysis(code)
    (REPORTS / "prompt_analysis.md").write_text("# Prompt analysis (text domains)\n\n" + pa +
                                                "\n\n# Prompt analysis (code)\n\n" + pa_code + "\n", encoding="utf-8")

    text = split_by_source(text)
    code["split"] = "test_code"
    assert set(code.source_id).isdisjoint(set(text.source_id)), "code/text source_id overlap"
    text[COLS].to_parquet(PROCESSED / "raid_main.parquet", index=False)
    rq3_text = text.copy()
    rq3_text["split_rq3"] = rq3_text.split.map({"train": "train", "val": "val", "test": "test_indomain"})
    code["split_rq3"] = "test_code"
    rq3 = pd.concat([rq3_text, code], ignore_index=True)
    rq3[COLS + ["split_rq3"]].to_parquet(PROCESSED / "raid_rq3.parquet", index=False)

    from src.preprocess import tokenize
    L = ["# Data checks\n"]
    ok = text.groupby("source_id").split.nunique().max() == 1
    L.append(f"- no source_id in >1 split: {'PASS' if ok else 'FAIL'}")
    L.append("- code source_ids disjoint from text source_ids: PASS (asserted)")
    L.append("- (source_id, model) unique and 5 rows per source_id: PASS (asserted)")
    empty = int((rq3.generation.str.len() == 0).sum())
    L.append(f"- empty generations: {empty} ({'PASS' if empty == 0 else 'FAIL'})")
    L.append(f"- total rows: text {len(text)}, code {len(code)}, rq3 {len(rq3)}")
    ms = pd.crosstab(text.model, text.split)
    L += ["", "## model x split (text)", "", md_table(ms)]
    bal = all(ms[c].nunique() == 1 for c in ms.columns)
    L.append(f"\nBalanced per split (all models equal count): {'PASS' if bal else 'FAIL'}")
    L += ["", "## domain x split (text)", "", md_table(pd.crosstab(text.domain, text.split)),
          "", "## model x domain (text)", "", md_table(pd.crosstab(text.model, text.domain)),
          "", "## model x split_rq3 (incl. code)", "", md_table(pd.crosstab(rq3.model, rq3.split_rq3)),
          "", "## source_ids per split (text)",
          "", md_table(text.groupby("split").source_id.nunique().to_frame("n_source_ids"))]
    ls = text.groupby("model")[["char_len", "word_len"]].agg(["mean", "median"]).round(1).loc[MODELS]
    L += ["", "## Length stats per model (text, raw, before truncation)", "", md_table(ls)]
    lc = code.groupby("model")[["char_len", "word_len"]].agg(["mean", "median"]).round(1).loc[MODELS]
    L += ["", "## Length stats per model (code)", "", md_table(lc)]
    samp = rq3.sample(2000, random_state=SEED)
    ntok = samp.generation.map(lambda s: len(tokenize(s)))
    samp = samp.assign(ntok=ntok, gt256=ntok > 256)
    L += ["", "## Tokens > 256 (src.preprocess.tokenize, random sample of 2000 from rq3 incl. code)", "",
          f"- overall: {samp.gt256.mean() * 100:.1f}% ; median tokens {ntok.median():.0f}",
          "", md_table((samp.groupby("model").gt256.mean() * 100).round(1).loc[MODELS].to_frame("pct_gt_256")),
          "", md_table((samp.groupby("domain").gt256.mean() * 100).round(1).to_frame("pct_gt_256"))]
    (REPORTS / "data_checks.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print("\nWrote:", PROCESSED / "raid_main.parquet", PROCESSED / "raid_rq3.parquet")


if __name__ == "__main__":
    main()
