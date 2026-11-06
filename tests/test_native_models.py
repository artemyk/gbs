"""Additional native model families and grouped-query cache layouts."""
import pytest
import torch
from transformers import (LlamaConfig, LlamaForCausalLM, GPTNeoXConfig, GPTNeoXForCausalLM,
                          GPTBigCodeConfig, GPTBigCodeForCausalLM)

from gbs import TransformersAdapter


@pytest.mark.parametrize("family", ["llama", "gpt_neox", "gpt_bigcode"])
def test_native_family_predictions_cache_gradients_and_query_features(family):
    torch.manual_seed(31)
    common = dict(vocab_size=5, bos_token_id=0, eos_token_id=None)
    if family == "llama":
        model = LlamaForCausalLM(LlamaConfig(**common, hidden_size=16, intermediate_size=24,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=4, max_position_embeddings=32))
    elif family == "gpt_neox":
        model = GPTNeoXForCausalLM(GPTNeoXConfig(**common, hidden_size=16, intermediate_size=24,
            num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=32,
            rope_parameters={"rope_type": "default", "rope_theta": 10000., "partial_rotary_factor": 1.}))
    else:
        model = GPTBigCodeForCausalLM(GPTBigCodeConfig(**common, n_embd=16, n_layer=2,
            n_head=4, n_inner=24, n_positions=32))
    adapter = TransformersAdapter(model)
    histories = torch.randint(5, (2, 4))
    state = adapter.history_state(histories).requires_grad_(True)
    inputs = torch.cat([torch.zeros(2, 1, dtype=torch.long), histories], 1)
    expected = model(inputs, use_cache=False).logits[:, -1].log_softmax(-1)
    actual = adapter.predict(state)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    actual.square().mean().backward()
    assert state.grad.norm() > 0
    queries = torch.randn(adapter.n_layers, adapter.n_heads, 2, adapter.head_dim, requires_grad=True)
    features = adapter.features(state.detach(), queries)
    features.square().mean().backward()
    assert queries.grad.norm() > 0
    assert features.shape[1] == adapter.token_features+queries.numel()
    expected_heads = model.config.num_attention_heads if family == "gpt_neox" else model.config.num_key_value_heads
    assert adapter.n_heads == expected_heads
