"""Pure helpers for the secondary data-scaling study (T10, 'beyond spec').

* subsample_ids: nested, seeded subsampling of train source_ids (25% subset of 50% subset of 100%).
* draw_new_ids: draw NEW source_ids per domain from an unused pool, excluding every id already used.
* verify_scaled: invariants of the expanded dataset (val/test untouched, no overlap, paired, balanced).
"""
from __future__ import annotations

import math

import numpy as np

FRACTION_SEED = 42


def subsample_ids(ids, fraction, seed=FRACTION_SEED):
    """Return a sorted list of ceil(fraction * n) ids. The ids are put in one fixed seeded random order (independent of
    `fraction`) and the first k are taken, so subsets are NESTED: subsample_ids(ids, .25) is contained in
    subsample_ids(ids, .5), which is contained in subsample_ids(ids, 1.0) (= all ids)."""
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    ids = sorted(set(map(str, ids)))
    order = np.random.default_rng(seed).permutation(len(ids))
    k = min(len(ids), max(1, math.ceil(round(fraction * len(ids), 9))))
    return sorted(ids[i] for i in order[:k])


def draw_new_ids(candidates_by_domain, used_ids, existing_by_domain, cap, rng, domains=None):
    """For each domain (in `domains` order, one shared rng) draw `cap - existing_by_domain[domain]` ids from the sorted
    candidates that are not in `used_ids`. Raises if a pool is too small. Returns {domain: sorted list of new ids}."""
    used = set(map(str, used_ids))
    out, taken = {}, set()
    for dom in (domains or sorted(candidates_by_domain)):
        need = cap - existing_by_domain.get(dom, 0)
        pool = np.array(sorted(c for c in map(str, candidates_by_domain[dom]) if c not in used and c not in taken))
        if need < 0:
            raise ValueError(f"{dom}: already has more ids than cap")
        if len(pool) < need:
            raise ValueError(f"{dom}: need {need} new ids, only {len(pool)} unused candidates")
        pick = sorted(rng.choice(pool, size=need, replace=False).tolist()) if need else []
        out[dom] = pick
        taken.update(pick)
    return out


def verify_scaled(main, scaled, n_models=5):
    """Check the expanded frame against the original raid_main frame. Returns {check: bool}; all must be True."""
    new = scaled[~scaled["id"].isin(main["id"])]
    res = {}
    for s in ("val", "test"):
        a = main[main["split"] == s].sort_values("id")["id"].to_numpy()
        b = scaled[scaled["split"] == s].sort_values("id")["id"].to_numpy()
        res[f"{s} rows identical to raid_main"] = bool(np.array_equal(a, b))
    res["old train rows kept"] = set(main.loc[main["split"] == "train", "id"]) <= set(
        scaled.loc[scaled["split"] == "train", "id"])
    res["new rows are train only"] = bool((new["split"] == "train").all())
    held = set(main.loc[main["split"].isin(["val", "test"]), "source_id"])
    res["no new source_id in val/test"] = set(new["source_id"]).isdisjoint(held)
    res["new source_ids disjoint from old train ids"] = set(new["source_id"]).isdisjoint(
        set(main.loc[main["split"] == "train", "source_id"]))
    res["each source_id in exactly one split"] = bool(scaled.groupby("source_id")["split"].nunique().max() == 1)
    res[f"paired: {n_models} rows per source_id"] = bool((scaled.groupby("source_id").size() == n_models).all())
    res["(source_id, model) unique"] = not scaled.duplicated(["source_id", "model"]).any()
    counts = scaled.groupby("split")["label"].apply(lambda s: s.value_counts().nunique() == 1)
    res["classes exactly balanced in every split"] = bool(counts.all())
    return res
