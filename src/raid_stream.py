"""Stream a RAID CSV over HTTP, filter on the fly, and write parquet + counts.

Usage:
  python -m src.raid_stream --url <csv url> --out data/interim/x.parquet --counts data/interim/c.csv
"""
import argparse
import io
import json
import os
import sys
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

KEEP_COLS = ["id", "source_id", "model", "decoding", "repetition_penalty",
             "attack", "domain", "title", "prompt", "generation"]
COUNT_KEYS = ["model", "domain", "decoding", "repetition_penalty", "attack"]
DEFAULT_MODELS = ["chatgpt", "llama-chat", "mistral-chat", "cohere-chat", "mpt-chat"]


class ResumableHTTPReader(io.RawIOBase):
    """Raw binary reader over HTTP that reconnects with Range on errors/short EOF."""

    def __init__(self, url, session=None, max_retries=20, backoff_base=1.0,
                 backoff_max=60.0, timeout=60, sleep=time.sleep):
        super().__init__()
        self.url = url
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.timeout = timeout
        self._sleep = sleep
        self.pos = 0
        self.total = None
        self.reconnects = 0
        self._resp = None
        self._connect()

    def _connect(self):
        """Open a connection at self.pos, with retries."""
        attempt = 0
        while True:
            try:
                headers = {"Range": f"bytes={self.pos}-"} if self.pos > 0 else {}
                resp = self.session.get(self.url, headers=headers, stream=True,
                                        timeout=self.timeout)
                resp.raise_for_status()
                if self.pos > 0 and resp.status_code != 206:
                    resp.close()
                    raise IOError(f"server ignored Range (status {resp.status_code})")
                cl = resp.headers.get("Content-Length")
                if self.total is None and cl is not None:
                    self.total = self.pos + int(cl)
                self._resp = resp
                return
            except Exception as e:  # noqa: BLE001
                attempt += 1
                if attempt > self.max_retries:
                    raise
                self._backoff(attempt, e)

    def _backoff(self, attempt, err):
        delay = min(self.backoff_base * 2 ** (attempt - 1), self.backoff_max)
        print(f"[reader] {type(err).__name__}: {err}; retry {attempt}/{self.max_retries} "
              f"in {delay:.1f}s at byte {self.pos}", file=sys.stderr, flush=True)
        self._sleep(delay)

    def _close_resp(self):
        if self._resp is not None:
            try:
                self._resp.close()
            except Exception:  # noqa: BLE001
                pass
            self._resp = None

    def readable(self):
        return True

    def readinto(self, b):
        n = len(b)
        if n == 0:
            return 0
        attempt = 0
        while True:
            try:
                if self._resp is None:
                    self._connect()
                data = self._resp.raw.read(n)
                if data:
                    b[:len(data)] = data
                    self.pos += len(data)
                    return len(data)
                # EOF
                if self.total is None or self.pos >= self.total:
                    return 0
                raise IOError(f"premature EOF at {self.pos}/{self.total}")
            except Exception as e:  # noqa: BLE001
                attempt += 1
                if attempt > self.max_retries:
                    raise
                self.reconnects += 1
                self._close_resp()
                self._backoff(attempt, e)

    def close(self):
        self._close_resp()
        super().close()


def _fmt_elapsed(s):
    return f"{int(s // 3600):d}:{int(s % 3600 // 60):02d}:{int(s % 60):02d}"


def _write_counts(counts, path):
    if counts is None:
        return
    tmp = path + ".tmp"
    counts.reset_index().to_csv(tmp, index=False)
    os.replace(tmp, path)


def run(url, out, counts_path, models=DEFAULT_MODELS, domains=None, max_bytes=None,
        chunksize=50_000, checkpoint_every=20, session=None, **reader_kw):
    t0 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(counts_path)), exist_ok=True)
    raw = ResumableHTTPReader(url, session=session, **reader_kw)
    text = io.TextIOWrapper(io.BufferedReader(raw, 1 << 20), encoding="utf-8", newline="")
    schema = pa.schema([(c, pa.string()) for c in KEEP_COLS])
    writer = pq.ParquetWriter(out, schema)
    models = set(models)
    domains = set(domains) if domains else None
    counts = None
    rows_seen = rows_kept = 0
    completed = True
    try:
        reader = pd.read_csv(text, dtype=str, keep_default_na=False,
                             chunksize=chunksize, engine="c")
        for i, chunk in enumerate(reader, 1):
            gen_ok = chunk["generation"].str.strip() != ""
            tmp = chunk[COUNT_KEYS].copy()
            tmp["n_rows"] = 1
            tmp["n_nonempty"] = gen_ok.astype("int64")
            agg = tmp.groupby(COUNT_KEYS, sort=False)[["n_rows", "n_nonempty"]].sum()
            counts = agg if counts is None else \
                pd.concat([counts, agg]).groupby(level=COUNT_KEYS, sort=False).sum()

            mask = (chunk["attack"] == "none") & chunk["model"].isin(models) & gen_ok
            if domains is not None:
                mask &= chunk["domain"].isin(domains)
            kept = chunk.loc[mask, KEEP_COLS]
            if len(kept):
                writer.write_table(pa.Table.from_pandas(kept, schema=schema,
                                                        preserve_index=False))
            rows_seen += len(chunk)
            rows_kept += len(kept)
            tot = raw.total or 0
            print(f"[chunk {i}] bytes {raw.pos:,}/{tot:,} ({100 * raw.pos / tot if tot else 0:.1f}%) "
                  f"seen {rows_seen:,} kept {rows_kept:,} elapsed {_fmt_elapsed(time.time() - t0)}",
                  flush=True)
            if i % checkpoint_every == 0:
                _write_counts(counts, counts_path)
            if max_bytes is not None and raw.pos > max_bytes:
                completed = False
                break
    finally:
        writer.close()
        _write_counts(counts, counts_path)
        bytes_read = raw.pos
        raw.close()
    elapsed = time.time() - t0
    stats = {"rows_seen": rows_seen, "rows_kept": rows_kept, "bytes": bytes_read,
             "elapsed": elapsed, "completed": completed}
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"DONE rows_seen={rows_seen:,} rows_kept={rows_kept:,} bytes={bytes_read:,} "
          f"elapsed={_fmt_elapsed(elapsed)} completed={completed}", flush=True)
    return stats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--counts", required=True)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--domains", nargs="+", default=None)
    p.add_argument("--max-bytes", type=int, default=None)
    p.add_argument("--chunksize", type=int, default=50_000)
    a = p.parse_args(argv)
    run(a.url, a.out, a.counts, a.models, a.domains, a.max_bytes, a.chunksize)


if __name__ == "__main__":
    main()
