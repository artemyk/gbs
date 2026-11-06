import copy

import pytest
import torch

from gbs import (AffineEncoder, Architecture, AttentionDecoder, BeliefStateAdapter,
                 FitConfig, GBSAbstraction, GHMM, RefineConfig, fit_abstraction, refine_encoder)
from gbs.adapters import words
from gbs.fit import conditional_kl
from reference_transformer import Config, Transformer


@pytest.mark.parametrize("steps", [0, 2])
def test_refinement_keeps_a_fallback_when_exact_prediction_kl_rounds_negative(steps):
    torch.manual_seed(12)
    biases = torch.randn(1000, 2)
    logp = biases.log_softmax(-1)
    bias = biases[int(conditional_kl(logp.exp(), logp).argmin())]
    teacher = Transformer(Config(vocab=2, bos_visible=True))
    with torch.no_grad():
        teacher.unembed.weight.zero_()
        teacher.unembed.bias.copy_(bias)
    adapter = BeliefStateAdapter(teacher)
    logp = teacher(torch.zeros(1, 4, dtype=torch.long)).log_softmax(-1)
    assert conditional_kl(logp.exp(), logp).mean() < 0
    histories = words(2, 4)
    architecture = Architecture(1, 1, 4)
    decoder = AttentionDecoder(adapter, architecture)
    with torch.no_grad():
        states = adapter.history_state(histories[::2])
        codes = decoder(adapter, states)
    encoder = AffineEncoder.initialize(adapter, states, codes)
    model = GBSAbstraction(adapter, decoder, encoder, architecture)
    initial = copy.deepcopy(encoder.state_dict())
    result = refine_encoder(model, histories[::2], histories[1::2], config=RefineConfig(
        steps=steps, horizon=2, batch_size=4, validate_every=1), progress=None)
    assert result.trace[0]["feasible"]
    assert result.trace[0]["metrics"]["single_kl"]["mean"] < 0
    assert result.initialization["validation_limits"]["single_kl"] == 0
    assert all(torch.isfinite(value).all() for value in encoder.state_dict().values())
    if steps == 0:
        assert all(torch.equal(value, encoder.state_dict()[key]) for key, value in initial.items())


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8])
def test_default_ghmm_fitting_accepts_all_supported_integer_history_types(dtype):
    torch.manual_seed(5)
    adapter = BeliefStateAdapter(Transformer(Config(bos_visible=True)))
    histories = words(3, 3).to(dtype)
    result = fit_abstraction(adapter, histories[:12], histories[12:20], config=FitConfig(encoder_refine_steps=0, joint_refine_steps=0,
        dimension=1, queries_per_head=1, hidden=4, horizon=2, future_length=1,
        spectral_samples=16, decoder_steps=0, encoder_steps=0, joint_steps=0,
        batch_size=2, validate_every=1, seed=10), progress=None)
    assert torch.isfinite(result.abstraction.encode(torch.zeros(2, 1))).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS hardware required")
def test_cpu_double_ghmm_initializes_an_mps_teacher():
    torch.manual_seed(5)
    adapter = BeliefStateAdapter(Transformer(Config(vocab=2, bos_visible=True))).to("mps")
    initializer = GHMM(torch.tensor([[[.2, 0.], [.5, .7]], [[.8, 0.], [-.5, .3]]], dtype=torch.float64),
                       torch.tensor([1., .5], dtype=torch.float64))
    histories = words(2, 4)
    result = fit_abstraction(adapter, histories[::2], histories[1::2], initializer=initializer,
        config=FitConfig(encoder_refine_steps=0, joint_refine_steps=0, dimension=1, queries_per_head=1, hidden=4, horizon=2,
                         decoder_steps=0, encoder_steps=0, joint_steps=0), progress=None)
    assert initializer.initial.device.type == "cpu" and initializer.initial.dtype == torch.float64
    assert result.abstraction.encoder.mean.device.type == "mps"


def test_byte_tokens_are_valid_for_a_256_symbol_vocabulary():
    from gbs.data import prepare_histories
    tokens = torch.tensor([[0, 128, 255]], dtype=torch.uint8)
    torch.testing.assert_close(prepare_histories(tokens, 256), tokens.long())


def test_fractional_histories_are_rejected_before_token_conversion():
    adapter = BeliefStateAdapter(Transformer(Config()))
    with pytest.raises(ValueError, match="integer"):
        adapter.history_state(torch.tensor([[.1, 1.9]]))
