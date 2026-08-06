# Pretraining Data

The raw corpus is intentionally ignored by Git. The downloader records selected
ModelScope files, sizes, and SHA-256 metadata in `manifest.json`; it does not
commit the downloaded data.

Install the downloader dependency through the required Tsinghua mirror:

```bash
uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --python .venv -e ".[dev]"
```

Download/resume the selected source shards:

```bash
.venv/bin/python scripts/download_pretrain_datasets.py --output datasets
```

The selection alternates available sources and paginates the ModelScope file
listing. `--count-tokens` performs an exact tokenizer-aware count, but the
training-ready token manifest is produced by the preprocessing command:

```bash
.venv/bin/python scripts/prepare_pretrain_data.py \
  --input-dir datasets \
  --output-dir datasets/tokens \
  --tokenizer temp/qwen-tokenizer \
  --local-files-only \
  --min-tokens 20000000000
```

The preprocessor reads all `.jsonl`, `.jsonl.gz`, and `.jsonl.zst` files under
the input directory, appends one EOS token per document, writes uint32 mmap
shards, and fails if fewer than 20B tokens were prepared. Use
`--min-tokens 0` only for a smoke dataset.
