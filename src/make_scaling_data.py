"""Build data/processed/raid_scaled.parquet (SECONDARY, 'beyond spec' data-scaling study; see DATA_CARD addendum).

Same schema as raid_main.parquet. Val and test rows are copied unchanged (identical source_ids and rows), all original
train rows are kept, and NEW source_ids are drawn per domain from the unused pool of data/interim/train_chat_none.parquet
(seed 42, one shared rng, domains in the data.py order), added to TRAIN ONLY, until each domain has `--cap` ids in total
(default 900 = 400 original + 500 new). All 5 models are kept for every drawn id (paired). Every id already used (the 3,200
text ids and the 400 code ids of RQ3) is excluded from the pool.

    python -m src.make_scaling_data            # needs data/interim/ (not tracked in git); the output IS tracked
Memory: the interim parquet is streamed one row group at a time (62 groups) with only the needed columns.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.data import COLS, DECODING, DOMAINS, INTERIM, MODELS, PROCESSED, REP, SEED, add_stats
from src.scaling import draw_new_ids, verify_scaled

POOL_COLS = ["source_id", "model", "domain", "decoding", "repetition_penalty", "generation"]
ROW_COLS = ["id", "source_id", "model", "domain", "decoding", "repetition_penalty", "title", "prompt", "generation"]


def _filter_group(t: pa.Table) -> pa.Table:
    """Same row filter as src.data (chosen models, sampling, rep penalty 'no', non-empty generation)."""
    m = pc.and_(pc.is_in(t["model"], value_set=pa.array(MODELS)),
                pc.and_(pc.equal(t["decoding"], DECODING), pc.equal(t["repetition_penalty"], REP)))
    m = pc.and_(m, pc.greater(pc.utf8_length(t["generation"]), 0))
    return t.filter(m)


def candidate_ids(path) -> dict:
    """-> {domain: sorted list of source_ids having exactly one (sampling, no) row for each of the 5 models}."""
    pf = pq.ParquetFile(path)
    parts = []
    for i in range(pf.num_row_groups):
        t = _filter_group(pf.read_row_group(i, columns=POOL_COLS)).drop_columns(["generation"])
        parts.append(t.to_pandas())
    d = pd.concat(parts, ignore_index=True)
    assert not d.duplicated(["source_id", "model"]).any(), "duplicate (source_id, model) rows"
    k = d.groupby(["domain", "source_id"])["model"].nunique()
    full = k[k == len(MODELS)].reset_index()
    return {dom: sorted(full.loc[full["domain"] == dom, "source_id"]) for dom in DOMAINS}


def read_rows(path, ids) -> pd.DataFrame:
    """Rows (all needed columns) of the chosen ids, streamed per row group."""
    pf = pq.ParquetFile(path)
    ids_arr = pa.array(sorted(ids))
    parts = []
    for i in range(pf.num_row_groups):
        t = _filter_group(pf.read_row_group(i, columns=ROW_COLS))
        t = t.filter(pc.is_in(t["source_id"], value_set=ids_arr))
        if t.num_rows:
            parts.append(t.to_pandas())
    return pd.concat(parts, ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--interim", default=str(INTERIM / "train_chat_none.parquet"))
    ap.add_argument("--main", default=str(PROCESSED / "raid_main.parquet"))
    ap.add_argument("--rq3", default=str(PROCESSED / "raid_rq3.parquet"))
    ap.add_argument("--out", default=str(PROCESSED / "raid_scaled.parquet"))
    ap.add_argument("--cap", type=int, default=900, help="total source_ids per domain (original + new)")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)

    main_df = pd.read_parquet(a.main)
    rq3 = pd.read_parquet(a.rq3, columns=["source_id", "split_rq3"])
    code_ids = set(rq3.loc[rq3["split_rq3"] == "test_code", "source_id"])
    used = set(main_df["source_id"]) | code_ids
    existing = main_df.groupby("domain")["source_id"].nunique().to_dict()
    print(f"used ids: {len(used)} (text {main_df['source_id'].nunique()}, code {len(code_ids)}); per domain {existing}")

    cands = candidate_ids(a.interim)
    print("candidate ids with all 5 models per domain:", {d: len(v) for d, v in cands.items()})
    picks = draw_new_ids(cands, used, existing, a.cap, np.random.default_rng(a.seed), domains=DOMAINS)
    new_ids = {i for v in picks.values() for i in v}
    print("new ids per domain:", {d: len(v) for d, v in picks.items()}, "total", len(new_ids))

    new = add_stats(read_rows(a.interim, new_ids))
    new["split"] = "train"
    scaled = pd.concat([main_df[COLS], new[COLS]], ignore_index=True)
    checks = verify_scaled(main_df, scaled)
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    assert all(checks.values()), "verification failed; not writing"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    scaled.to_parquet(a.out, index=False)
    print(f"wrote {a.out}: {len(scaled)} rows, {Path(a.out).stat().st_size / 1e6:.1f} MB\n"
          f"{pd.crosstab(scaled['model'], scaled['split'])}")
    print("train ids per domain:", scaled[scaled.split == "train"].groupby("domain")["source_id"].nunique().to_dict())


if __name__ == "__main__":
    main()
