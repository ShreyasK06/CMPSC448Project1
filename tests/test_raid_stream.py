import io

import pandas as pd

from src import raid_stream as rs

HEADER = "id,adv_source_id,source_id,model,decoding,repetition_penalty,attack,domain,title,prompt,generation\n"


def make_csv(n=300):
    rows = []
    for i in range(n):
        model = ["human", "chatgpt", "llama-chat", "gpt4"][i % 4]
        attack = "none" if i % 5 else "homoglyph"
        dec = "" if model == "human" else "greedy"
        rp = "" if model == "human" else "no"
        gen = "" if i % 17 == 0 else f'line one\nline, "two" {i}'
        gen = gen.replace('"', '""')
        dom = "code" if i % 2 else "books"
        rows.append(f'{i},{i},{i},{model},{dec},{rp},{attack},{dom},"t {i}","p\n{i}","{gen}"\n')
    return (HEADER + "".join(rows)).encode()


class FakeResp:
    def __init__(self, data, start, fail_after=None, status=200):
        self.status_code = status
        self.headers = {"Content-Length": str(len(data) - start)}
        self._buf = io.BytesIO(data[start:])
        self._fail_after = fail_after
        self._served = 0
        self.raw = self

    def raise_for_status(self):
        pass

    def read(self, n):
        if self._fail_after is not None and self._served >= self._fail_after:
            raise ConnectionError("boom")
        if self._fail_after is not None:
            n = min(n, self._fail_after - self._served)
        d = self._buf.read(n)
        self._served += len(d)
        return d

    def close(self):
        pass


class FakeSession:
    """Plan entries: int = raise after that many bytes; 'eof' = silently end early."""

    def __init__(self, data, plan):
        self.data, self.plan, self.calls = data, list(plan), []

    def get(self, url, headers=None, stream=True, timeout=None):
        rng = (headers or {}).get("Range")
        start = int(rng.split("=")[1].rstrip("-")) if rng else 0
        self.calls.append(start)
        fail = self.plan.pop(0) if self.plan else None
        status = 206 if rng else 200
        if fail == "eof":
            r = FakeResp(self.data[: start + 1000], start, status=status)
            r.headers["Content-Length"] = str(len(self.data) - start)  # premature EOF
            return r
        return FakeResp(self.data, start, fail_after=fail, status=status)


def test_reconnect_reads_all_bytes():
    data = make_csv()
    sess = FakeSession(data, [5000, "eof", 3000])
    r = rs.ResumableHTTPReader("x", session=sess, sleep=lambda s: None)
    out = io.BufferedReader(r, 4096).read()
    assert out == data
    assert r.reconnects >= 3
    assert sess.calls[0] == 0 and all(c > 0 for c in sess.calls[1:])


def test_run_filters_and_counts(tmp_path):
    data = make_csv()
    sess = FakeSession(data, [4000, "eof"])
    out, cnt = str(tmp_path / "o.parquet"), str(tmp_path / "c.csv")
    stats = rs.run("x", out, cnt, domains=["code"], chunksize=50, session=sess,
                   sleep=lambda s: None)
    full = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
    assert stats["completed"] and stats["rows_seen"] == len(full)
    exp = full[(full.attack == "none") & full.model.isin(rs.DEFAULT_MODELS)
               & (full.generation.str.strip() != "") & (full.domain == "code")]
    got = pd.read_parquet(out)
    assert list(got.id) == list(exp.id) and len(got) == stats["rows_kept"] > 0
    assert list(got.columns) == rs.KEEP_COLS
    assert "\n" in got.generation.iloc[0] and '"two"' in got.generation.iloc[0]
    c = pd.read_csv(cnt, dtype=str, keep_default_na=False)
    assert c.n_rows.astype(int).sum() == len(full)
    assert c.n_nonempty.astype(int).sum() == (full.generation.str.strip() != "").sum()


def test_max_bytes_stops_early(tmp_path):
    data = make_csv(2000)
    sess = FakeSession(data, [])
    stats = rs.run("x", str(tmp_path / "o.parquet"), str(tmp_path / "c.csv"),
                   chunksize=100, max_bytes=5000, session=sess)
    assert not stats["completed"] and stats["rows_seen"] < 2000
