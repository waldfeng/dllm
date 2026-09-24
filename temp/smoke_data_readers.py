"""Smoke test for the extended dataset readers (parquet, JSON-array zst,
split gzip parts, zst-JSONL) plus a download-manifest dry run.

All fixtures are synthesized locally; no network is required.
"""

from __future__ import annotations

import gzip
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import zstandard

from pretrain_data import iter_text_records, jsonl_paths

import pyarrow as pa
import pyarrow.parquet as pq


def make_fixtures(root: Path) -> None:
    # 1. parquet with 'text' column (FineWeb-Edu / Falcon style)
    parquet_dir = root / "fineweb" / "data"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"text": ["fineweb doc one", "fineweb doc two"], "id": [1, 2]}),
        parquet_dir / "train-00000.parquet",
    )
    # 2. parquet with 'content' column (Falcon style)
    falcon_dir = root / "falcon" / "data"
    falcon_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"content": ["falcon doc one", "falcon doc two"], "url": ["a", "b"]}),
        falcon_dir / "train-00000.parquet",
    )
    # 3. JSON-array zst (WuDao style)
    records = [
        {"title": "词条标题", "content": "词条正文内容。"},
        {"content": "只有正文的词条"},
    ]
    path = root / "wudao" / "data" / "part-0000.json.zst"
    path.parent.mkdir(parents=True, exist_ok=True)
    with zstandard.open(path, "wb") as stream:
        stream.write(json.dumps(records, ensure_ascii=False).encode("utf-8"))
    # 4. split gzip parts (MAP-CC style): one stream, three .part files
    lines = "".join(
        json.dumps({"text": f"map-cc record {index}"}, ensure_ascii=False) + "\n"
        for index in range(30)
    )
    blob = gzip.compress(lines.encode("utf-8"))
    split_dir = root / "map-cc"
    split_dir.mkdir(parents=True, exist_ok=True)
    chunk = (len(blob) + 2) // 3
    for index in range(3):
        part = blob[index * chunk : (index + 1) * chunk]
        (split_dir / f"zh_test.jsonl.gz.part{index + 1:02d}").write_bytes(part)
    # 5. plain jsonl.gz (MathPile style)
    math_lines = "".join(
        json.dumps({"text": f"math record {index}"}) + "\n" for index in range(3)
    )
    math_dir = root / "mathpile" / "train" / "arXiv"
    math_dir.mkdir(parents=True, exist_ok=True)
    math_path = math_dir / "chunk_1.jsonl.gz"
    math_path.write_bytes(gzip.compress(math_lines.encode()))
    # 6. zst-JSONL (peS2o / dclm style)
    dclm_lines = "".join(
        json.dumps({"text": f"dclm record {index}"}) + "\n" for index in range(5)
    )
    compressor = zstandard.ZstdCompressor()
    dclm_dir = root / "dclm"
    dclm_dir.mkdir(parents=True, exist_ok=True)
    (dclm_dir / "shard_0.jsonl.zst").write_bytes(
        compressor.compress(dclm_lines.encode())
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        make_fixtures(root)
        paths = jsonl_paths(root)
        names = [str(path.relative_to(root)) for path in paths]
        assert "fineweb/data/train-00000.parquet" in names, names
        assert "map-cc/zh_test.jsonl.gz" not in names, names  # representative only
        assert sum(1 for n in names if n.startswith("map-cc/zh_test")) == 1, names
        records = list(iter_text_records(root))
        by_prefix = {}
        for path, text in records:
            key = str(path.relative_to(root)).split("/")[0]
            by_prefix.setdefault(key, []).append(text)
        assert by_prefix["fineweb"] == ["fineweb doc one", "fineweb doc two"], by_prefix
        assert by_prefix["falcon"] == ["falcon doc one", "falcon doc two"], by_prefix
        assert by_prefix["wudao"] == ["词条标题\n词条正文内容。", "只有正文的词条"], by_prefix
        assert by_prefix["map-cc"] == [f"map-cc record {i}" for i in range(30)], by_prefix["map-cc"][:3]
        assert by_prefix["mathpile"] == [f"math record {i}" for i in range(3)]
        assert by_prefix["dclm"] == [f"dclm record {i}" for i in range(5)]
        print(f"smoke_data_readers OK: {len(records)} records across {len(by_prefix)} sources")


if __name__ == "__main__":
    main()
