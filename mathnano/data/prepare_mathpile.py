r"""Convert MathPile into the parquet text-shards nanochat pretrains on (Track A).

nanochat reads ``shard_NNNNN.parquet`` files (single ``text`` column) from
``$NANOCHAT_BASE_DIR/base_data_climbmix/`` and tokenises on the fly; the **last** shard is the
validation split. There is no ``.bin`` step. So our job: stream MathPile documents and write them
out as text parquet shards, capped to a Chinchilla-aware token budget, with the final shard held
out for validation.

MathPile (``GAIR/MathPile``) is gated + non-commercial. Download it yourself first::

    huggingface-cli login
    huggingface-cli download GAIR/MathPile --repo-type dataset --local-dir data/raw/mathpile

then point this script at that dir. Files are ``*.jsonl.gz`` with fields ``text`` and ``SubSet``.

Budget: depth=16 (~200M scaling params) is compute-optimal around 2.4B tokens (nanochat's default
12 tokens/param) to ~4B (Chinchilla 20). We size by characters using ~4 chars/token.

Usage:
    python -m mathnano.data.prepare_mathpile --input data/raw/mathpile --target-tokens 4e9
    python -m mathnano.data.prepare_mathpile --self-test     # no data needed; verifies mechanics
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
from typing import Iterator, Optional

CONFIG = {
    "chars_per_token": 4.0,        # rough English/LaTeX estimate for budgeting
    "docs_per_shard": 20_000,      # ~ matches nanochat shard granularity
    "min_doc_chars": 200,          # drop near-empty docs (arXiv preambles etc.)
    "max_doc_chars": 100_000,      # truncate pathological giant docs
    # Optional per-SubSet weighting (None = keep all). e.g. downweight raw web text.
    "subset_filter": None,         # e.g. {"arXiv","Textbooks","StackExchange","ProofWiki","Wikipedia"}
}


def _base_data_dir() -> str:
    base = os.environ.get("NANOCHAT_BASE_DIR")
    if not base:
        raise SystemExit("Set NANOCHAT_BASE_DIR (where nanochat keeps its data/checkpoints).")
    return os.path.join(base, "base_data_climbmix")


def iter_mathpile_docs(input_dir: str) -> Iterator[str]:
    """Yield cleaned `text` strings from all MathPile *.jsonl.gz under input_dir."""
    files = sorted(glob.glob(os.path.join(input_dir, "**", "*.jsonl.gz"), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(input_dir, "**", "*.jsonl"), recursive=True))
    if not files:
        raise SystemExit(f"No .jsonl(.gz) files found under {input_dir}")
    keep = CONFIG["subset_filter"]
    for fp in files:
        opener = gzip.open if fp.endswith(".gz") else open
        with opener(fp, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if keep is not None and obj.get("SubSet") not in keep:
                    continue
                text = (obj.get("text") or "").strip()
                if len(text) < CONFIG["min_doc_chars"]:
                    continue
                yield text[: CONFIG["max_doc_chars"]]


def write_shards(docs: Iterator[str], out_dir: str, target_tokens: float) -> dict:
    """Write `docs` to shard_NNNNN.parquet (text column) until the token budget is hit.

    Returns stats. The caller is responsible for ensuring ≥2 shards exist so nanochat has a
    train split (all-but-last) and a val split (last shard).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(out_dir, exist_ok=True)
    target_chars = target_tokens * CONFIG["chars_per_token"]
    buf: list[str] = []
    shard_idx = total_chars = total_docs = 0

    def flush(idx: int, rows: list[str]) -> None:
        path = os.path.join(out_dir, f"shard_{idx:05d}.parquet")
        table = pa.table({"text": rows})
        pq.write_table(table, path)
        print(f"  wrote {path}  ({len(rows):,} docs)")

    for text in docs:
        buf.append(text)
        total_chars += len(text)
        total_docs += 1
        if len(buf) >= CONFIG["docs_per_shard"]:
            flush(shard_idx, buf)
            shard_idx += 1
            buf = []
        if total_chars >= target_chars:
            break
    if buf:
        flush(shard_idx, buf)
        shard_idx += 1

    return {
        "shards": shard_idx,
        "docs": total_docs,
        "chars": total_chars,
        "approx_tokens": int(total_chars / CONFIG["chars_per_token"]),
        "out_dir": out_dir,
    }


def _self_test() -> None:
    """Verify the sharding round-trips, with synthetic data (no MathPile needed)."""
    import tempfile
    import pyarrow.parquet as pq

    with tempfile.TemporaryDirectory() as tmp:
        # fake input: one jsonl.gz with a few docs across SubSets
        raw = os.path.join(tmp, "raw")
        os.makedirs(raw)
        with gzip.open(os.path.join(raw, "part.jsonl.gz"), "wt", encoding="utf-8") as f:
            for i in range(50):
                f.write(json.dumps({"text": f"Theorem {i}. " + "x" * 500,
                                    "SubSet": "arXiv"}) + "\n")
            f.write(json.dumps({"text": "too short", "SubSet": "arXiv"}) + "\n")  # dropped

        out = os.path.join(tmp, "shards")
        CONFIG["docs_per_shard"] = 20  # force multiple shards
        stats = write_shards(iter_mathpile_docs(raw), out, target_tokens=1e9)

        files = sorted(glob.glob(os.path.join(out, "*.parquet")))
        assert len(files) == stats["shards"] >= 2, "need >=2 shards (train + val)"
        # round-trip: every shard has a single 'text' column, no empties
        ndocs = 0
        for fp in files:
            t = pq.read_table(fp)
            assert t.column_names == ["text"], t.column_names
            ndocs += t.num_rows
        assert ndocs == 50, f"expected 50 kept docs, got {ndocs}"  # the short one dropped
        print(f"  self-test OK: {len(files)} shards, {ndocs} docs, "
              f"~{stats['approx_tokens']:,} tokens, cols=['text']")


def main() -> None:
    ap = argparse.ArgumentParser(description="MathPile -> nanochat text parquet shards")
    ap.add_argument("--input", type=str, help="dir with MathPile *.jsonl.gz")
    ap.add_argument("--target-tokens", type=float, default=4e9, help="approx token budget")
    ap.add_argument("--out", type=str, default=None,
                    help="output dir (default: $NANOCHAT_BASE_DIR/base_data_climbmix)")
    ap.add_argument("--self-test", action="store_true", help="run synthetic round-trip test")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return
    if not args.input:
        raise SystemExit("--input is required (or use --self-test)")

    out_dir = args.out or _base_data_dir()
    print(f"Sharding MathPile from {args.input} -> {out_dir} "
          f"(target ~{args.target_tokens:.2g} tokens)")
    stats = write_shards(iter_mathpile_docs(args.input), out_dir, args.target_tokens)
    print(f"\nDone: {stats['shards']} shards, {stats['docs']:,} docs, "
          f"~{stats['approx_tokens']:,} tokens in {stats['out_dir']}")
    if stats["shards"] < 2:
        print("WARNING: <2 shards — nanochat needs >=2 (train=all-but-last, val=last).")


if __name__ == "__main__":
    main()
