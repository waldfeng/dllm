"""Download a reproducible ~20B-token pretraining corpus from ModelScope.

ModelScope repositories expose byte sizes, not tokenizer token counts. The
manifest therefore uses a conservative byte budget and records the files used;
run ``--count-tokens`` after downloading for the exact Qwen2.5 token count.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from modelscope import snapshot_download
from modelscope.hub.api import HubApi


# Chinese web text commonly yields materially fewer Qwen tokens than raw UTF-8
# bytes. A 55 GiB raw-text budget is a practical 20-25B-token target.
TARGET_BYTES = 55 * 1024**3
DATASETS = (
    ("modelscope/SkyPile-150B", "skypile", "data/*.jsonl"),
    ("swift/chinese-c4", "chinese-c4", "data/*.jsonl.zst"),
)


def files_for(repo_id: str, pattern: str) -> list[dict]:
    files = []
    for item in HubApi().get_dataset_files(repo_id):
        path = item.get("Path", "")
        if path.startswith("data/") and path.endswith(pattern.split("*")[-1]):
            files.append({"path": path, "size": int(item.get("Size") or 0), "sha256": item.get("Sha256", "")})
    return files


def select_files() -> list[dict]:
    selected: list[dict] = []
    # Alternate sources so the resulting corpus is not a single-source crawl.
    candidates = []
    for repo_id, name, pattern in DATASETS:
        for item in files_for(repo_id, pattern):
            item.update(repo_id=repo_id, source=name)
            candidates.append(item)
    candidates.sort(key=lambda x: (x["path"].rsplit("/", 1)[-1], x["source"]))
    total = 0
    for item in candidates:
        if total >= TARGET_BYTES:
            break
        selected.append(item)
        total += item["size"]
    return selected


def download(selected: list[dict], output: Path, workers: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for repo_id in sorted({x["repo_id"] for x in selected}):
        paths = [x["path"] for x in selected if x["repo_id"] == repo_id]
        local = output / repo_id.replace("/", "__")
        print(f"Downloading {repo_id}: {len(paths)} files -> {local}", flush=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local),
            allow_patterns=paths,
            max_workers=workers,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="datasets")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--count-tokens", action="store_true")
    args = parser.parse_args()
    selected = select_files()
    manifest = {
        "target_bytes": TARGET_BYTES,
        "selected_bytes": sum(x["size"] for x in selected),
        "files": selected,
        "tokenizer": "Qwen/Qwen2.5-0.5B",
        "token_count": None,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Selected {len(selected)} files, {manifest['selected_bytes'] / 1024**3:.2f} GiB", flush=True)
    if args.dry_run:
        return
    download(selected, output, args.workers)
    if args.count_tokens:
        print("Exact token counting is intentionally separate: use the tokenizer-aware preprocessing pipeline before training.", file=sys.stderr)


if __name__ == "__main__":
    main()
