"""Download a reproducible ~12B-token classic pretraining corpus from ModelScope.

Sources are classic, high-quality pretraining corpora ranked by training-set
quality (FineWeb-Edu > DCLM-baseline > RefinedWeb > The Stack > peS2o > MAP-CC
> WuDaoCorpus > MathPile) and are disjoint from the previously prepared data
(English Wikipedia-derived WIT text plus SkyPile-150B and chinese-c4).

ModelScope repositories expose byte sizes, not tokenizer token counts. Each
source therefore carries a byte budget; run ``--count-tokens`` or
``scripts/estimate_token_count.py`` after downloading for the exact Qwen2.5
token count.
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

GIB = 1024**3

# (repo_id, source_name, path_prefixes, name_contains, path_suffix, byte_budget)
DATASETS = (
    (
        "AI-ModelScope/fineweb-edu",
        "fineweb-edu",
        ("data/CC-MAIN-",),
        "train-",
        ".parquet",
        7 * GIB,
    ),
    (
        "AI-ModelScope/dclm-baseline-1.0",
        "dclm-baseline",
        ("global-shard_",),
        "",
        ".jsonl.zst",
        4 * GIB,
    ),
    (
        "tiiuae/falcon-refinedweb",
        "refinedweb",
        ("data/",),
        "train-",
        ".parquet",
        4 * GIB,
    ),
    (
        "AI-ModelScope/starcoderdata",
        "starcoder",
        (
            "python/",
            "javascript/",
            "java/",
            "cpp/",
            "c/",
            "typescript/",
            "go/",
            "rust/",
            "php/",
            "c-sharp/",
            "markdown/",
            "github-issues-filtered-structured/",
            "git-commits-cleaned/",
            "jupyter-structured-clean-dedup/",
        ),
        "",
        ".parquet",
        7 * GIB,
    ),
    (
        "allenai/peS2o",
        "pes2o",
        ("data/v3/",),
        "",
        ".zst",
        4 * GIB,
    ),
    (
        "m-a-p/MAP-CC",
        "map-cc",
        ("",),
        "zh_",
        ".gz.part",
        4 * GIB,
    ),
    (
        "whynlp/WuDaoCorpus-200G-shuffled",
        "wudao",
        ("data/",),
        "",
        ".json.zst",
        3 * GIB,
    ),
    (
        "GAIR/MathPile",
        "mathpile",
        ("train/",),
        "",
        ".jsonl.gz",
        2 * GIB,
    ),
)


def _page_files(repo_id: str) -> list[dict]:
    files: list[dict] = []
    api = HubApi()
    page_number = 1
    while True:
        page = api.get_dataset_files(repo_id, page_number=page_number, page_size=100)
        if not page:
            break
        for item in page:
            path = item.get("Path", "")
            if path:
                files.append(item)
        if len(page) < 100:
            break
        page_number += 1
    return files


def _matches(path: str, prefixes: tuple[str, ...], contains: str, suffix: str) -> bool:
    if not any(path.startswith(prefix) for prefix in prefixes):
        return False
    if suffix.endswith(".part"):
        _, _, tail = path.rpartition(".part")
        return tail.isdigit() and contains in path
    if not path.endswith(suffix):
        return False
    return contains in Path(path).name


def _group_split_archives(items: list[dict]) -> list[dict]:
    """Merge `x.jsonl.gz.partNN` listings into single logical files."""
    groups: dict[str, list[dict]] = {}
    plain: list[dict] = []
    for raw in items:
        path = raw.get("path") or raw.get("Path") or ""
        item = {
            "path": path,
            "size": int(raw.get("Size") or 0),
            "sha256": raw.get("Sha256", ""),
        }
        head, _, tail = path.rpartition(".part")
        if head and tail.isdigit():
            groups.setdefault(head, []).append(item)
        else:
            plain.append(item)
    merged: list[dict] = []
    for base, members in sorted(groups.items()):
        merged.append(
            {
                "path": base,
                "size": sum(int(m["size"]) for m in members),
                "parts": [m["path"] for m in sorted(members, key=lambda m: m["path"])],
                "sha256": "",
            }
        )
    plain.extend(merged)
    return plain


def files_for(repo_id: str, prefixes: tuple[str, ...], contains: str, suffix: str) -> list[dict]:
    if suffix.endswith(".part"):
        items = _group_split_archives(
            [
                item
                for item in _page_files(repo_id)
                if _matches(item.get("Path", ""), prefixes, contains, suffix)
            ]
        )
    else:
        items = []
        for item in _page_files(repo_id):
            path = item.get("Path", "")
            if _matches(path, prefixes, contains, suffix):
                items.append(
                    {
                        "path": path,
                        "size": int(item.get("Size") or 0),
                        "sha256": item.get("Sha256", ""),
                    }
                )
    return items


def select_files(
    target_bytes: int | None = None,
    excluded: set[tuple[str, str]] | None = None,
    source_filter: set[str] | None = None,
) -> list[dict]:
    """Round-robin one file per source until every source budget is filled.

    Files that would overshoot a source budget by more than 5% are skipped so
    a single oversized archive cannot blow up the download total.
    """
    excluded = excluded or set()
    selected: list[dict] = []
    candidates: dict[str, list[dict]] = {}
    budgets: dict[str, int] = {}
    for repo_id, name, prefixes, contains, suffix, budget in DATASETS:
        if source_filter is not None and name not in source_filter:
            continue
        items = []
        for item in files_for(repo_id, prefixes, contains, suffix):
            if (repo_id, item["path"]) in excluded:
                continue
            item.update(repo_id=repo_id, source=name)
            items.append(item)
        items.sort(key=lambda x: x["path"])
        candidates[name] = items
        budgets[name] = budget
    spent: dict[str, int] = {name: 0 for name in candidates}
    positions = {name: 0 for name in candidates}
    source_names = sorted(candidates)
    while True:
        progressed_any = False
        for source in list(source_names):
            items = candidates[source]
            progressed_source = False
            while positions[source] < len(items):
                item = items[positions[source]]
                positions[source] += 1
                size = int(item["size"])
                if spent[source] + size > budgets[source] * 1.05:
                    continue
                if target_bytes is not None and sum(
                    x["size"] for x in selected
                ) + size > target_bytes:
                    source_names.remove(source)
                    break
                selected.append(item)
                spent[source] += size
                progressed_source = True
                if spent[source] >= budgets[source]:
                    break
            if positions[source] >= len(items) or spent[source] >= budgets[source]:
                source_names.remove(source)
            progressed_any = progressed_any or progressed_source
        if not progressed_any:
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
        patterns = []
        for item in selected:
            if item["repo_id"] != repo_id:
                continue
            patterns.extend(item.get("parts") or [item["path"]])
        local = output / repo_id.replace("/", "__")
        print(f"Downloading {repo_id}: {len(patterns)} files -> {local}", flush=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local),
            allow_patterns=patterns,
            max_workers=workers,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="datasets")
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="parallel file workers passed to ModelScope snapshot_download",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--count-tokens", action="store_true")
    parser.add_argument("--tokenizer", default="tokenizer/qwen")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument(
        "--target-bytes",
        type=int,
        help="optional global cap on the selected byte total",
    )
    parser.add_argument(
        "--supplement-bytes",
        type=int,
        help="add this many bytes not already listed in output/manifest.json",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=[name for _, name, _, _, _, _ in DATASETS],
        help="restrict selection to named sources",
    )
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
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
            "source_budgets": {name: budget for _, name, _, _, _, budget in DATASETS},
            "files": selected,
            "tokenizer": "Qwen/Qwen2.5-0.5B",
            "token_count": None,
        }
    write_manifest(output, manifest)
    per_source: dict[str, int] = {}
    for item in selected:
        per_source[item["source"]] = per_source.get(item["source"], 0) + item["size"]
    for source, size in sorted(per_source.items()):
        print(f"  {source:16s} {size / GIB:8.2f} GiB", flush=True)
    print(f"Selected {len(selected)} files, {manifest['selected_bytes'] / GIB:.2f} GiB", flush=True)
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
