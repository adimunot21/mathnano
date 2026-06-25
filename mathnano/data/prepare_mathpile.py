r"""Convert MathPile into the parquet text-shards nanochat pretrains on (Track A).

nanochat reads ``shard_NNNNN.parquet`` files (single ``text`` column) from
``$NANOCHAT_BASE_DIR/base_data_climbmix/`` and tokenises on the fly; the **last** shard is the
validation split. There is no ``.bin`` step. So our job: stream MathPile documents and write them
out as text parquet shards, capped to a Chinchilla-aware token budget, with the final shard held
out for validation.

Two quality decisions:
  * **Source-balanced sampling.** MathPile is ~85% arXiv. Reading files in order would make the
    token budget almost all arXiv. We round-robin across sources (arXiv, textbooks, proofwiki,
    stackexchange, wikipedia, commoncrawl) so a 200M model sees a balanced diet of notation
    fluency (arXiv) and clean step-by-step reasoning (textbooks/stackexchange/proofwiki).
  * **Real held-out val.** MathPile ships a ``validation/`` split; we make that nanochat's last
    shard (its val) instead of slicing val out of train.

MathPile (``GAIR/MathPile``) is gated + non-commercial. Download it first::

    huggingface-cli download GAIR/MathPile --repo-type dataset --local-dir data/raw/mathpile

Files are ``train/<source>/*.jsonl.gz`` and ``validation/<source>/*.jsonl.gz`` with fields
``text`` and ``subset`` (lowercase). Budget by characters using ~4 chars/token; depth=16 is
compute-optimal around 2.4B (nanochat default 12 tok/param) to ~4B (Chinchilla 20).

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
    "val_tokens": 20_000_000,      # ~20M-token validation shard (the final shard)
    # Optional per-subset filter (None = keep all). MathPile `subset` values (verified):
    # {"arXiv","CommonCrawl","ProofWiki","StackExchange","Textbooks","Wikipedia"}.
    "subset_filter": None,
}


def _base_data_dir() -> str:
    base = os.environ.get("NANOCHAT_BASE_DIR")
    if not base:
        raise SystemExit("Set NANOCHAT_BASE_DIR (where nanochat keeps its data/checkpoints).")
    return os.path.join(base, "base_data_climbmix")


def _discover(input_dir: str) -> tuple[list[str], list[str]]:
    """Return (train_files, val_files). MathPile uses train/ and validation/ subdirs."""
    files = sorted(glob.glob(os.path.join(input_dir, "**", "*.jsonl.gz"), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(input_dir, "**", "*.jsonl"), recursive=True))
    if not files:
        raise SystemExit(f"No .jsonl(.gz) files found under {input_dir}")
    train = [f for f in files if os.sep + "validation" + os.sep not in f]
    val = [f for f in files if os.sep + "validation" + os.sep in f]
    return train, val


def _docs_from_file(fp: str) -> Iterator[str]:
    """Yield cleaned `text` from one jsonl(.gz) file, applying filters."""
    keep = CONFIG["subset_filter"]
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
            if keep is not None and obj.get("subset") not in keep:
                continue
            text = (obj.get("text") or "").strip()
            if len(text) < CONFIG["min_doc_chars"]:
                continue
            yield text[: CONFIG["max_doc_chars"]]


def _source_of(fp: str) -> str:
    """Source = the directory name holding the file (arXiv, textbooks, ...)."""
    return os.path.basename(os.path.dirname(fp))


def iter_balanced(files: list[str]) -> Iterator[str]:
    """Round-robin documents across sources so no single source dominates the budget."""
    by_source: dict[str, list[str]] = {}
    for fp in files:
        by_source.setdefault(_source_of(fp), []).append(fp)
    # one chained generator per source
    gens = {src: _chain(fps) for src, fps in by_source.items()}
    active = list(gens)
    while active:
        for src in list(active):
            try:
                yield next(gens[src])
            except StopIteration:
                active.remove(src)


def _chain(files: list[str]) -> Iterator[str]:
    for fp in files:
        yield from _docs_from_file(fp)


def _write_one_shard(rows: list[str], out_dir: str, idx: int) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = os.path.join(out_dir, f"shard_{idx:05d}.parquet")
    pq.write_table(pa.table({"text": rows}), path)
    print(f"  wrote {path}  ({len(rows):,} docs)")


def write_shards(docs: Iterator[str], out_dir: str, target_tokens: float,
                 start_index: int = 0) -> dict:
    """Write `docs` to shard_{i}.parquet (text column) from `start_index` until the budget hits."""
    os.makedirs(out_dir, exist_ok=True)
    target_chars = target_tokens * CONFIG["chars_per_token"]
    buf: list[str] = []
    idx = start_index
    total_chars = total_docs = 0
    for text in docs:
        buf.append(text)
        total_chars += len(text)
        total_docs += 1
        if len(buf) >= CONFIG["docs_per_shard"]:
            _write_one_shard(buf, out_dir, idx); idx += 1; buf = []
        if total_chars >= target_chars:
            break
    if buf:
        _write_one_shard(buf, out_dir, idx); idx += 1
    return {"next_index": idx, "shards": idx - start_index, "docs": total_docs,
            "chars": total_chars, "approx_tokens": int(total_chars / CONFIG["chars_per_token"])}


def _write_val_shard(docs: Iterator[str], out_dir: str, idx: int, target_tokens: float) -> int:
    """Write a SINGLE final shard for validation (nanochat uses only the last shard as val)."""
    target_chars = target_tokens * CONFIG["chars_per_token"]
    rows, chars = [], 0
    for text in docs:
        rows.append(text); chars += len(text)
        if chars >= target_chars:
            break
    if rows:
        _write_one_shard(rows, out_dir, idx)
        print(f"  (val shard: {len(rows):,} docs, ~{int(chars/CONFIG['chars_per_token']):,} tokens)")
        return idx + 1
    return idx


def _self_test() -> None:
    """Verify sharding + balancing round-trips with synthetic data (no MathPile needed)."""
    import tempfile
    import pyarrow.parquet as pq

    with tempfile.TemporaryDirectory() as tmp:
        # synthetic: two sources under train/, one under validation/
        for src, n in [("arXiv", 60), ("textbooks", 30)]:
            d = os.path.join(tmp, "raw", "train", src); os.makedirs(d)
            with gzip.open(os.path.join(d, "p.jsonl.gz"), "wt", encoding="utf-8") as f:
                for i in range(n):
                    f.write(json.dumps({"text": f"{src} doc {i} " + "x" * 400,
                                        "subset": src}) + "\n")
        vd = os.path.join(tmp, "raw", "validation", "arXiv"); os.makedirs(vd)
        with gzip.open(os.path.join(vd, "v.jsonl.gz"), "wt", encoding="utf-8") as f:
            for i in range(10):
                f.write(json.dumps({"text": f"val doc {i} " + "x" * 400, "subset": "arXiv"}) + "\n")

        out = os.path.join(tmp, "shards")
        CONFIG["docs_per_shard"] = 25
        train_files, val_files = _discover(os.path.join(tmp, "raw"))
        assert len(train_files) == 2 and len(val_files) == 1

        # balanced order: first ~ docs should alternate arXiv/textbooks
        order = [t.split()[0] for t in iter_balanced(train_files)]
        assert order[:4] == ["arXiv", "textbooks", "arXiv", "textbooks"], order[:4]

        st = write_shards(iter_balanced(train_files), out, target_tokens=1e9)
        nxt = _write_val_shard(iter_balanced(val_files), out, st["next_index"], 1e9)
        files = sorted(glob.glob(os.path.join(out, "*.parquet")))
        assert len(files) == nxt >= 2, files
        # last shard is the val shard
        last = pq.read_table(files[-1])
        assert last.column_names == ["text"] and last.num_rows == 10
        # train shards hold the 90 train docs
        train_docs = sum(pq.read_table(f).num_rows for f in files[:-1])
        assert train_docs == 90, train_docs
        print(f"  self-test OK: {len(files)} shards (last=val, {last.num_rows} docs), "
              f"{train_docs} train docs, balanced order verified")


def main() -> None:
    ap = argparse.ArgumentParser(description="MathPile -> nanochat text parquet shards")
    ap.add_argument("--input", type=str, help="dir with MathPile train/ + validation/")
    ap.add_argument("--target-tokens", type=float, default=4e9, help="approx TRAIN token budget")
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
    train_files, val_files = _discover(args.input)
    print(f"Sharding MathPile from {args.input} -> {out_dir}")
    print(f"  sources: {sorted({_source_of(f) for f in train_files})}")
    print(f"  train files: {len(train_files)}  val files: {len(val_files)}  "
          f"(target ~{args.target_tokens:.2g} train tokens)")

    st = write_shards(iter_balanced(train_files), out_dir, args.target_tokens)
    next_idx = st["next_index"]
    if val_files:
        next_idx = _write_val_shard(iter_balanced(val_files), out_dir, next_idx,
                                    CONFIG["val_tokens"])
    else:
        print("  WARNING: no validation/ files — nanochat will use the last TRAIN shard as val.")

    print(f"\nDone: {next_idx} shards total, ~{st['approx_tokens']:,} train tokens in {out_dir}")
    if next_idx < 2:
        print("WARNING: <2 shards — nanochat needs >=2 (train=all-but-last, val=last).")


if __name__ == "__main__":
    main()
