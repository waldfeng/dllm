"""End-to-end smoke: readout training -> selection checkpoint -> benchmark on a tiny model."""

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from memory_transformer import MemoryTransformerConfig, MemoryTransformerForCausalLM

WORK = Path("/tmp/opencode/exit_smoke")


def build_store() -> Path:
    store = WORK / "tokens"
    store.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(0)
    tokens = generator.integers(0, 32, size=4000).astype(np.uint32)
    (store / "shard0.bin").write_bytes(tokens.tobytes())
    manifest = {
        "shards": [{"path": "shard0.bin", "tokens": int(tokens.size)}],
        "vocab_size": 32,
        "eos_token_id": 9,
        "total_tokens": int(tokens.size),
    }
    (store / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return store


def build_base_checkpoint(store: Path) -> Path:
    config = MemoryTransformerConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_memory_blocks=4,
        top_k=2,
        max_position_embeddings=32,
        early_exit_min_depth=2,
    )
    model = MemoryTransformerForCausalLM(config)
    path = WORK / "base.pt"
    torch.save({"model": model.state_dict(), "config": asdict(config), "step": 0}, path)
    return path


def build_tokenizer() -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders

    directory = WORK / "tiny-tokenizer"
    directory.mkdir(parents=True, exist_ok=True)
    vocab = {"[UNK]": 0, "abc": 1, "def": 2, "the": 3, "meaning": 4, "of": 5, "life": 6, "is": 7}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.WordPiece(prefix="")
    tokenizer.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]"}),
        encoding="utf-8",
    )
    return directory


def run(command: list[str]) -> None:
    print("$", " ".join(str(part) for part in command), flush=True)
    result = subprocess.run([str(part) for part in command], capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit(f"command failed: {result.returncode}")


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    store = build_store()
    base = build_base_checkpoint(store)
    trained = WORK / "exit-readout.pt"
    tokenizer_dir = build_tokenizer()
    python = str(Path(".venv/bin/python").resolve())
    run([
        python, "train.py",
        "--train-exit-readout",
        "--resume", base,
        "--checkpoint", trained,
        "--dataset-dir", store,
        "--tokenizer", "nonexistent-tokenizer",
        "--num-layers", "4", "--hidden-size", "32", "--intermediate-size", "64",
        "--num-attention-heads", "4", "--num-key-value-heads", "2",
        "--num-memory-blocks", "4", "--top-k", "2", "--max-position-embeddings", "32",
        "--steps", "6", "--batch-size", "2", "--sequence-length", "8",
        "--log-steps", "1", "--device", "cpu", "--precision", "fp32",
        "--exit-readout-lr", "3e-2", "--exit-tau-confidence", "0.3",
        "--exit-tau-margin", "0.05", "--exit-tau-delta", "0.05", "--exit-min-depth", "2",
    ])
    payload = torch.load(trained, map_location="cpu", weights_only=False)
    assert any(name.startswith("exit_readout.") for name in payload["model"])
    assert "exit_loss_weights" in payload and "exit_temperature" in payload
    print(
        "readout checkpoint keys ok weights=",
        [round(float(v), 4) for v in payload["exit_loss_weights"].tolist()],
        "temperature=",
        round(float(payload["exit_temperature"]), 4),
        flush=True,
    )
    run([
        python, "test.py",
        "--checkpoint", trained,
        "--tokenizer", tokenizer_dir,
        "--prompt", "abc def",
        "--max-chars", "8",
        "--benchmark-tokens", "5",
        "--benchmark-prompt", "abc def",
    ])
    print("SMOKE OK", flush=True)


if __name__ == "__main__":
    main()
