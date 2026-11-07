"""Shared learned-query fitting, prefix coverage, and saved abstractions."""
import numpy as np
import pytest
import torch

from gbs import AffineEncoder, BeliefStateAdapter, DecoderConfig, GBSAbstraction, fit_decoder
from gbs.adapters import words
from reference_transformer import Config, make_model


@pytest.mark.parametrize('rope', [False, True])
def test_learned_prefix_queries_update_and_saved_abstraction_preserves_predictions(rope, tmp_path):
    adapter = BeliefStateAdapter(make_model(Config(rope=rope, bos_visible=True)).double())
    parents = words(3, 5)
    train, valid = parents[:100], parents[100:120]
    config = DecoderConfig(dimension=2, queries_per_head=3, hidden=8, decoder_steps=20,
        validate_every=5, decoder_validation_samples=10, seed=4)
    torch.manual_seed(config.seed)
    initial_queries = torch.randn(adapter.n_layers, adapter.n_heads, config.queries_per_head,
                                  adapter.head_dim, dtype=adapter.dtype)
    coordinates = lambda h: torch.stack([h.sum(1), (h == 1).sum(1)], 1).double()
    result = fit_decoder(adapter, train, valid, coordinates, config=config, progress=None)
    assert set(result.validation.groups) == set(range(6))
    assert result.diagnostics['training_rows'] == len(train)
    assert result.diagnostics['validation_rows'] == 60
    assert 'queries' in dict(result.decoder.named_parameters())
    assert not torch.equal(initial_queries, result.decoder.queries)
    roots = adapter.history_state(train[:, :4])
    code = result.decoder(adapter, roots).detach()
    encoder = AffineEncoder.initialize(adapter, roots, code)
    model = GBSAbstraction(adapter, result.decoder, encoder, result.architecture)
    written = model.encode(code)
    model.predict(code).square().mean().backward()
    assert encoder.linear.weight.grad.norm() > 0
    path = tmp_path/'abstraction.pt'
    model.save(path)
    restored = GBSAbstraction.load(path, adapter)
    torch.testing.assert_close(restored.decode(written), model.decode(written), atol=0, rtol=0)
    torch.testing.assert_close(restored.predict(code), model.predict(code), atol=0, rtol=0)


def test_learned_prefix_preparation_uses_bounded_batches():
    class CountingAdapter(BeliefStateAdapter):
        def __init__(self, model):
            super().__init__(model)
            self.batches = []
        def history_state(self, histories):
            self.batches.append(len(histories))
            return super().history_state(histories)
    adapter = CountingAdapter(make_model(Config(vocab=2)))
    parents = words(2, 4)
    result = fit_decoder(adapter, parents[:8], parents[8:], lambda h: h.float().sum(1, keepdim=True),
        config=DecoderConfig(dimension=1, queries_per_head=2, hidden=4, decoder_steps=0,
            decoder_samples=600, decoder_min_length=3, decoder_max_length=3), progress=None)
    assert sum(adapter.batches) == 608
    assert max(adapter.batches) <= 256
    assert result.diagnostics['training_rows'] == 600


def test_prefix_domain_and_parent_split_validation():
    adapter = BeliefStateAdapter(make_model(Config()))
    parents = words(3, 3)
    target = lambda h: h.float().sum(1, keepdim=True)
    with pytest.raises(ValueError, match='overlap'):
        fit_decoder(adapter, parents, parents[:2], target, progress=None)
    with pytest.raises(ValueError, match='supplied parent'):
        fit_decoder(adapter, parents[:10], parents[10:], target,
                    config=DecoderConfig(decoder_max_length=4), progress=None)
    # Root-only is an explicit control; the default spans all supplied prefixes.
    result = fit_decoder(adapter, parents[:10], parents[10:], target,
        config=DecoderConfig(dimension=1, queries_per_head=1, hidden=4, decoder_steps=0,
                             decoder_min_length=3, decoder_max_length=3), progress=None)
    assert set(result.training.groups) == set(result.validation.groups) == {3}


def test_full_fit_anchors_prefixes_with_learned_queries():
    from gbs import FitConfig, fit_abstraction
    torch.manual_seed(5)
    adapter = BeliefStateAdapter(make_model(Config(bos_visible=True)).double())
    roots = words(3, 3)
    result = fit_abstraction(adapter, roots[:15], roots[15:],
        initializer=lambda h: h.double().sum(1, keepdim=True), config=FitConfig(
            dimension=1, queries_per_head=2, hidden=8,
            horizon=2, decoder_steps=0, joint_steps=2,
            validate_every=2), progress=None)
    joint = [r for r in result.trace if r['stage'] == 'joint']
    assert len(joint) == 2
    assert set(joint[-1]['validation_coordinate_by_length']) == {str(n) for n in range(6)}
    assert np.isfinite(joint[-1]['validation_loss'])
    assert result.abstraction.decoder.queries.requires_grad
    assert result.abstraction.encoder.mean[-1] == 3


@pytest.mark.parametrize('field,value', [('decoder_lr', float('nan')),
    ('decoder_lr', float('inf')), ('decoder_steps', 1.5), ('decoder_validation_samples', True),
    ('decoder_max_length', float('inf')), ('decoder_samples', float('nan')),
    ('hidden', None), ('queries_per_head', 1.5)])
def test_invalid_decoder_settings_fail_before_teacher_preparation(field, value):
    with pytest.raises(ValueError, match=field):
        config = DecoderConfig(**{field: value})
        fit_decoder(None, None, None, None, config=config, progress=None)
