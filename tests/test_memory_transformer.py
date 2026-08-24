import json
from pathlib import Path

import numpy as np
import pytest
import torch

from memory_transformer import (
    MemoryTransformerConfig,
    MemoryTransformerForCausalLM,
    SharedMemory,
)
from pretrain_data import TokenStore, prepare_token_shards, token_batches


class DummyTokenizer:
    eos_token_id = 9
    name_or_path = "dummy"

    def __len__(self):
        return 10

    def encode(self, text, add_special_tokens=False):
        return [ord(char) % 9 for char in text]


def tiny_model() -> MemoryTransformerForCausalLM:
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
    )
    return MemoryTransformerForCausalLM(config)


def test_labels_equal_to_input_ids_is_next_token():
    model = tiny_model().eval()
    ids = torch.arange(16).view(2, 8) % 32
    shifted = torch.cat((ids[:, 1:], torch.zeros(2, 1, dtype=ids.dtype)), dim=1)
    with torch.no_grad():
        next_token = model(ids, labels=ids)["loss"]
        plus_two = model(ids, labels=shifted)["loss"]
    assert torch.isfinite(next_token)
    assert next_token != plus_two


def test_forward_loss_router_gradients_and_stats():
    model = tiny_model()
    ids = torch.randint(32, (2, 8))
    output = model(ids, labels=ids, return_router_stats=True)
    assert output["logits"].shape == (2, 8, 32)
    assert torch.isfinite(output["loss"])
    assert torch.isfinite(output["router_loss"])
    (output["loss"] + 0.01 * output["router_loss"]).backward()
    assert model.memory.ffn_gate.grad is not None
    assert model.layers[0].router.weight.grad is not None
    assert len(output["route_indices"]) == 4
    assert output["route_indices"][0].shape == (2, 8, 4, 2)


def test_default_topology_is_24_cycles_and_24_memory_blocks():
    config = MemoryTransformerConfig()
    assert config.num_layers == 24
    assert config.num_memory_blocks == 24
    assert config.top_k == 3
    assert config.parameter_count() == 1_092_383_232


def test_parameter_count_matches_actual_tiny_model():
    model = tiny_model()
    assert sum(parameter.numel() for parameter in model.parameters()) == model.config.parameter_count()


def test_memory_initializes_each_matrix_at_linear_scale():
    config = tiny_model().config
    model = tiny_model()
    q_expected = (2.0 / (config.hidden_size + config.hidden_size)) ** 0.5
    gate_expected = (
        2.0 / (config.hidden_size + config.intermediate_size)
    ) ** 0.5
    down_expected = gate_expected
    assert 0.5 * q_expected < model.memory.q_proj.std() < 1.5 * q_expected
    assert 0.5 * gate_expected < model.memory.ffn_gate.std() < 1.5 * gate_expected
    assert 0.5 * down_expected < model.memory.ffn_down.std() < 1.5 * down_expected


def test_router_balance_uses_block_ids():
    probabilities = torch.full((1, 4, 1, 4), 0.05)
    probabilities[..., 0] = 0.85
    probabilities /= probabilities.sum(dim=-1, keepdim=True)
    all_zero = torch.zeros((1, 4, 1, 2), dtype=torch.long)
    balanced = torch.tensor([[[[0, 1]], [[1, 2]], [[2, 3]], [[3, 0]]]])
    concentrated = MemoryTransformerForCausalLM.load_balance_loss(
        all_zero, probabilities
    )
    spread = MemoryTransformerForCausalLM.load_balance_loss(balanced, probabilities)
    assert concentrated > spread


def test_attention_mask_and_invalid_head_dimension():
    model = tiny_model().eval()
    ids = torch.randint(32, (2, 8))
    mask = torch.tensor(
        [[True, True, True, True, False, False, False, False], [True] * 8]
    )
    with torch.no_grad():
        output = model(ids, labels=ids, attention_mask=mask)
    assert torch.isfinite(output["loss"])
    with pytest.raises(ValueError, match="head dimension"):
        MemoryTransformerConfig(
            vocab_size=32,
            hidden_size=10,
            intermediate_size=20,
            num_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_memory_blocks=1,
            top_k=1,
        )
    with pytest.raises(ValueError, match="num_layers must equal"):
        MemoryTransformerConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_memory_blocks=2,
            top_k=1,
        )


def test_token_shards_and_cross_shard_batches(tmp_path: Path):
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "tokens"
    input_dir.mkdir()
    (input_dir / "part.jsonl").write_text(
        "\n".join(json.dumps({"text": text}) for text in ("abc", "defgh", "ij")),
        encoding="utf-8",
    )
    manifest = prepare_token_shards(
        input_dir,
        output_dir,
        DummyTokenizer(),
        shard_tokens=5,
        min_tokens=0,
    )
    assert manifest["total_tokens"] == 13
    store = TokenStore(output_dir)
    assert store.read(3, 5).tolist() == [9, 1, 2, 3, 4]
    batches = list(token_batches(store, sequence_length=3, batch_size=1, steps=2))
    assert len(batches) == 2
    assert batches[0][0].shape == (1, 3)
    assert batches[0][0].dtype == np.int64
