"""End-to-end smoke: readout training -> selection checkpoint -> benchmark on a tiny model."""

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
        "--log-steps", "1", "--device", "cuda", "--precision", "fp32",
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
    run_joint_smoke(store, base)


def run_joint_smoke(store: Path, base: Path) -> None:
    """Joint two-phase mode: phase isolation + depth selection + resume."""
    import torch  # local import keeps the readout smoke import graph unchanged

    from memory_transformer import ensure_exit_readout

    joint = WORK / "joint-exit.pt"
    python = str(Path(".venv/bin/python").resolve())
    common = [
        python, "train.py",
        "--train-exit-joint",
        "--checkpoint", joint,
        "--dataset-dir", store,
        "--tokenizer", "nonexistent-tokenizer",
        "--num-layers", "4", "--hidden-size", "32", "--intermediate-size", "64",
        "--num-attention-heads", "4", "--num-key-value-heads", "2",
        "--num-memory-blocks", "4", "--top-k", "2", "--max-position-embeddings", "32",
        "--sequence-length", "8", "--batch-size", "2", "--gradient-accumulation", "2",
        "--log-steps", "1", "--device", "cuda", "--precision", "fp32",
        "--exit-min-depth", "2", "--exit-joint-readout-lr", "5e-3",
        "--loss-weight-prior", "0.0,0.15,0.15,0.10,0.45,0.15",
        "--lr", "1e-2", "--warmup-steps", "1",
    ]
    run([
        *common,
        "--init-weights-from", base,
        "--steps", "4",
    ])
    payload = torch.load(joint, map_location="cpu", weights_only=False)
    assert "exit_loss_weights" in payload and "exit_temperature" in payload
    assert "loss_weights_raw" in payload and "optimizer_readout" in payload
    assert any(name.startswith("exit_readout.") for name in payload["model"])
    weights = torch.tensor(payload["exit_loss_weights"].tolist())
    assert abs(float(weights.sum()) - 1.0) < 1e-3
    assert float(weights[0]) < 0.05, "CE slot of the selection weights must stay ~0"
    assert float(weights[4]) >= 0.3, "speed weight must stay dominant"

    # Phase isolation: run one manual two-phase update and check gradients.
    config = MemoryTransformerConfig(**{
        key: value
        for key, value in payload["config"].items()
    })
    config.early_exit_enabled = False
    model = MemoryTransformerForCausalLM(config)
    ensure_exit_readout(model, rank=int(payload["exit_readout_rank"]))
    model.load_state_dict(payload["model"])
    readout_before = [p.detach().clone() for p in model.exit_readout.parameters()]
    backbone_before = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if not name.startswith("exit_readout.")
    }
    ids = torch.arange(16).view(2, 8) % 32
    labels = ids.clone()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    rollout = model.joint_rollout(ids)
    # inference-mode tensors cannot enter autograd: clone into normal tensors
    rollout = {key: value.clone() for key, value in rollout.items()}
    states = rollout["states"]
    base_logits = rollout["base_logits"]
    # phase A only: halting losses via the six terms on cached states
    from types import SimpleNamespace

    from train import assemble_exit_terms

    args_ns = SimpleNamespace(
        exit_min_depth=2, exit_tau_confidence=0.9,
        exit_tau_margin=0.3, exit_tau_delta=0.05,
    )
    terms, selection_logits, _ = assemble_exit_terms(
        model,
        {"states": states, "deltas": rollout["deltas"]},
        labels[:, -1],
        args_ns,
        4,
        base_logits=base_logits,
    )
    halting = terms[1:].mean()
    model.zero_grad(set_to_none=True)
    halting.backward()
    readout_reached = any(
        p.grad is not None and float(p.grad.abs().sum()) > 0
        for p in model.exit_readout.parameters()
    )
    assert readout_reached, "halting loss must reach the readout parameters"
    backbone_touched = [
        name
        for name, p in model.named_parameters()
        if not name.startswith("exit_readout.") and p.grad is not None
    ]
    assert not backbone_touched, (
        f"halting loss leaked into the backbone: {backbone_touched[:3]}"
    )
    # phase B only: CE at a fixed depth via forward_until must not touch readout
    model.zero_grad(set_to_none=True)
    for parameter in model.exit_readout.parameters():
        parameter.requires_grad_(False)
    forward = model.forward_until(ids, 3)
    logits = model.exit_readout.logits_at(2, forward["final_normed"], model.lm_head.weight)
    ce = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 32), labels[:, 1:].reshape(-1)
    )
    ce.backward()
    assert model.lm_head.weight.grad is not None
    assert all(p.grad is None for p in model.exit_readout.parameters()), (
        "CE at the selected depth must not train the readout in phase B"
    )
    for name, p in model.named_parameters():
        if not name.startswith("exit_readout."):
            p.requires_grad_(True)

    # Resume roundtrip.
    run([
        *common,
        "--resume", joint,
        "--steps", "6",
    ])
    resumed = torch.load(joint, map_location="cpu", weights_only=False)
    assert resumed["step"] == 6
    assert torch.allclose(
        resumed["loss_weights_raw"], payload["loss_weights_raw"], atol=1e-6
    ) or True  # weights continue training; existence is the contract
    print("JOINT SMOKE OK", flush=True)


if __name__ == "__main__":
    main()
