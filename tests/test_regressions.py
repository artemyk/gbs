import pytest
import torch

from gbs import (BeliefStateAdapter,
                 FitConfig, GHMM, fit_abstraction)
from gbs.adapters import words
from reference_transformer import Config, make_model




@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8])
def test_default_ghmm_fitting_accepts_all_supported_integer_history_types(dtype):
    torch.manual_seed(5)
    adapter = BeliefStateAdapter(make_model(Config()))
    histories = words(3, 3).to(dtype)
    result = fit_abstraction(adapter, histories[:12], histories[12:20], config=FitConfig(
        dimension=1, queries_per_head=1, hidden=4, horizon=2, future_length=1,
        spectral_samples=16, decoder_steps=0, joint_steps=0,
        batch_size=2, validate_every=1, seed=10), progress=None)
    assert torch.isfinite(result.abstraction.encode(torch.zeros(2, 1))).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS hardware required")
def test_cpu_double_ghmm_initializes_an_mps_teacher():
    torch.manual_seed(5)
    adapter = BeliefStateAdapter(make_model(Config(vocab=2))).to("mps")
    initializer = GHMM(torch.tensor([[[.2, 0.], [.5, .7]], [[.8, 0.], [-.5, .3]]], dtype=torch.float64),
                       torch.tensor([1., .5], dtype=torch.float64))
    histories = words(2, 4)
    result = fit_abstraction(adapter, histories[::2], histories[1::2], initializer=initializer,
        config=FitConfig(dimension=1, queries_per_head=1, hidden=4, horizon=2,
                         decoder_steps=0, joint_steps=0), progress=None)
    assert initializer.initial.device.type == "cpu" and initializer.initial.dtype == torch.float64
    assert result.abstraction.encoder.mean.device.type == "mps"


def test_byte_tokens_are_valid_for_a_256_symbol_vocabulary():
    from gbs.data import prepare_histories
    tokens = torch.tensor([[0, 128, 255]], dtype=torch.uint8)
    torch.testing.assert_close(prepare_histories(tokens, 256), tokens.long())


def test_fractional_histories_are_rejected_before_token_conversion():
    adapter = BeliefStateAdapter(make_model(Config()))
    with pytest.raises(ValueError, match="integer"):
        adapter.history_state(torch.tensor([[.1, 1.9]]))
