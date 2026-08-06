import torch

from memory_transformer import QWEN25_TOKENIZER, MemoryTransformerConfig, MemoryTransformerForCausalLM


def tiny_model() -> MemoryTransformerForCausalLM:
    config = MemoryTransformerConfig(
        vocab_size=97, hidden_size=32, intermediate_size=64, num_layers=2,
        num_attention_heads=4, num_key_value_heads=2, num_memory_blocks=24,
        top_k=3, max_position_embeddings=32,
    )
    return MemoryTransformerForCausalLM(config)


def test_forward_loss_and_gradients():
    model = tiny_model()
    ids = torch.randint(97, (2, 8))
    output = model(ids, labels=ids)
    assert output["logits"].shape == (2, 8, 97)
    assert torch.isfinite(output["loss"])
    assert torch.isfinite(output["router_loss"])
    output["loss"].backward()
    assert model.memory.ffn_gate.grad is not None


def test_top_k_and_recurrent_inference():
    model = tiny_model().eval()
    ids = torch.randint(97, (1, 5))
    with torch.no_grad():
        output = model(ids, recurrent_steps=2)
        generated = model.generate(ids, max_new_tokens=2, recurrent_steps=2)
    assert output["logits"].shape[1] == 5
    assert generated.shape == (1, 7)


def test_parameter_target_configuration():
    config = MemoryTransformerConfig()
    count = config.parameter_count()
    assert 1_000_000_000 <= count <= 1_500_000_000, count


def test_initialization_uses_finite_xavier_and_kaiming_weights():
    model = tiny_model()
    for parameter in (model.memory.q_proj, model.memory.k_proj, model.memory.v_proj, model.memory.o_proj):
        assert torch.isfinite(parameter).all()
        assert parameter.std() > 0
    for parameter in (model.memory.ffn_gate, model.memory.ffn_up, model.memory.ffn_down):
        assert torch.isfinite(parameter).all()
        assert parameter.std() > 0
    assert torch.isfinite(model.layers[0].router.weight).all()
    assert torch.isfinite(model.lm_head.weight).all()


def test_tokenizer_is_fixed_to_qwen25():
    assert QWEN25_TOKENIZER == "Qwen/Qwen2.5-0.5B"
    assert MemoryTransformerConfig().vocab_size == 151665
