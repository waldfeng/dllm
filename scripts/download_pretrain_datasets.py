"""Download a reproducible ~20B-token pretraining corpus from ModelScope.

ModelScope repositories expose byte sizes, not tokenizer token counts. The
manifest therefore uses a conservative byte budget and records the files used;
run ``--count-tokens`` after downloading for the exact Qwen2.5 token count.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from modelscope import snapshot_download
from modelscope.hub.api import HubApi

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Chinese web text commonly yields materially fewer Qwen tokens than raw UTF-8
# bytes. A 55 GiB raw-text budget is a practical 20-25B-token target.
TARGET_BYTES = 55 * 1024**3
DATASETS = (
    ("modelscope/SkyPile-150B", "skypile", "data/*.jsonl"),
    ("swift/chinese-c4", "chinese-c4", "data/*.jsonl.zst"),
)


def files_for(repo_id: str, pattern: str) -> list[dict]:
    files = []
    api = HubApi()
    page_number = 1
    suffix = pattern.split("*")[-1]
    while True:
        page = api.get_dataset_files(
            repo_id, page_number=page_number, page_size=100
        )
        if not page:
            break
        for item in page:
            path = item.get("Path", "")
            if path.startswith("data/") and path.endswith(suffix):
                files.append(
                    {
                        "path": path,
                        "size": int(item.get("Size") or 0),
                        "sha256": item.get("Sha256", ""),
                    }
                )
        if len(page) < 100:
            break
        page_number += 1
    return files


def select_files(
    target_bytes: int = TARGET_BYTES,
    excluded: set[tuple[str, str]] | None = None,
    source_filter: set[str] | None = None,
) -> list[dict]:
    excluded = excluded or set()
    selected: list[dict] = []
    candidates: dict[str, list[dict]] = {}
    for repo_id, name, pattern in DATASETS:
        if source_filter is not None and name not in source_filter:
            continue
        for item in files_for(repo_id, pattern):
            if (repo_id, item["path"]) in excluded:
                continue
            item.update(repo_id=repo_id, source=name)
            candidates.setdefault(name, []).append(item)
    for source in candidates:
        candidates[source].sort(key=lambda x: x["path"])
    total = 0
    source_names = sorted(candidates)
    positions = {source: 0 for source in source_names}
    while total < target_bytes and source_names:
        made_progress = False
        for source in source_names:
            position = positions[source]
            if position >= len(candidates[source]):
                continue
            item = candidates[source][position]
            positions[source] += 1
            selected.append(item)
            total += item["size"]
            made_progress = True
            if total >= target_bytes:
                break
        if not made_progress:
            break
    return selected


def write_manifest(output: Path, manifest: dict) -> None:
    temporary = output / ".manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(output / "manifest.json")


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
    parser.add_argument("--tokenizer", default="temp/qwen-tokenizer")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--target-bytes", type=int, default=TARGET_BYTES)
    parser.add_argument(
        "--supplement-bytes",
        type=int,
        help="add this many bytes not already listed in output/manifest.json",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=[name for _, name, _ in DATASETS],
        help="restrict selection to named sources, useful for uncompressed estimates",
    )
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    existing_manifest_path = output / "manifest.json"
    if args.supplement_bytes is not None:
        if args.supplement_bytes <= 0:
            raise ValueError("--supplement-bytes must be positive")
        if not existing_manifest_path.exists():
            raise FileNotFoundError(
                f"--supplement-bytes requires an existing {existing_manifest_path}"
            )
        manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        existing_files = manifest.get("files", [])
        excluded = {(item["repo_id"], item["path"]) for item in existing_files}
        source_filter = set(args.sources) if args.sources else None
        selected = select_files(args.supplement_bytes, excluded, source_filter)
        if sum(item["size"] for item in selected) < args.supplement_bytes:
            raise ValueError("available candidate files cannot satisfy supplement target")
        manifest["supplement_bytes"] = args.supplement_bytes
        manifest["files"] = existing_files + selected
        manifest["selected_bytes"] = sum(item["size"] for item in manifest["files"])
    else:
        source_filter = set(args.sources) if args.sources else None
        selected = select_files(args.target_bytes, source_filter=source_filter)
        manifest = {
            "target_bytes": args.target_bytes,
            "selected_bytes": sum(x["size"] for x in selected),
            "files": selected,
            "tokenizer": "Qwen/Qwen2.5-0.5B",
            "token_count": None,
        }
    write_manifest(output, manifest)
    print(f"Selected {len(selected)} files, {manifest['selected_bytes'] / 1024**3:.2f} GiB", flush=True)
    if args.dry_run:
        return
    download(selected, output, args.workers)
    if args.count_tokens:
        from pretrain_data import count_tokens, load_tokenizer

        tokenizer = load_tokenizer(args.tokenizer)
        token_count, document_count = count_tokens(output, tokenizer, args.max_tokens)
        manifest["token_count"] = token_count
        manifest["document_count"] = document_count
        write_manifest(output, manifest)
        print(f"Counted documents={document_count} tokens={token_count}", flush=True)


if __name__ == "__main__":
    main()
