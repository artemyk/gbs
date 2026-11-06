"""Attention decoder, affine cache encoder, and their induced recurrent model."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import torch
from torch import nn

from .adapters import CacheModel


@dataclass
class Architecture:
    dimension: int = 4
    queries_per_head: int = 4
    hidden: int = 64
    layers: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.layers is not None:
            self.layers = tuple(self.layers)


class AttentionDecoder(nn.Module):
    def __init__(self, adapter: CacheModel, architecture: Architecture):
        super().__init__()
        a = architecture
        if min(a.dimension, a.queries_per_head, a.hidden) < 1:
            raise ValueError("Decoder dimensions must be positive")
        if a.layers is not None and (not a.layers or len(set(a.layers)) != len(a.layers)
                or any(not isinstance(layer, int) or not 0 <= layer < adapter.n_layers for layer in a.layers)
                or tuple(sorted(a.layers)) != a.layers):
            raise ValueError("Decoder layers must be increasing, distinct, valid layer indices")
        self.layers = a.layers
        layers = adapter.n_layers if a.layers is None else len(a.layers)
        self.queries = nn.Parameter(torch.randn(layers, adapter.n_heads, a.queries_per_head, adapter.head_dim,
                                                device=adapter.device, dtype=adapter.dtype))
        inputs = adapter.token_features + layers*adapter.n_heads*a.queries_per_head*adapter.head_dim
        self.net = nn.Sequential(nn.Linear(inputs, a.hidden), nn.Tanh(), nn.Linear(a.hidden, a.dimension)).to(
            device=adapter.device, dtype=adapter.dtype)

    def forward(self, adapter, states):
        features = (adapter.features(states, self.queries) if self.layers is None else
                    adapter.features(states, self.queries, layers=self.layers))
        return self.net(features)


class AffineEncoder(nn.Module):
    """An affine writer with fixed structural metadata and train-only scaling."""
    def __init__(self, dimension, mean, scale, writable):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale)
        self.register_buffer("writable", writable)
        self.linear = nn.Linear(dimension, int(writable.sum())).to(device=mean.device, dtype=mean.dtype)

    def forward(self, code):
        result = self.mean.expand(len(code), -1).clone()
        result[:, self.writable] = self.mean[self.writable]+self.scale[self.writable]*self.linear(code)
        return result

    @classmethod
    def initialize(cls, adapter, states, codes, *, writable=None):
        """Least squares on selected coordinates; remaining nonstructural entries use their training mean."""
        x, y = codes.detach().cpu().double(), states.detach().cpu().double()
        mean, scale = y.mean(0), y.std(0, correction=0)
        scale = torch.where(scale < 1e-3, torch.ones_like(scale), scale)
        structural = ~adapter.encoder_mask(states).cpu()
        mask = ~structural if writable is None else writable.detach().cpu()
        if mask.dtype != torch.bool or mask.shape != structural.shape or bool((mask & structural).any()):
            raise ValueError("Writable mask must select nonstructural state coordinates")
        if not bool(torch.allclose(y[:, structural], y[:1, structural].expand(len(y), -1), atol=1e-6, rtol=1e-6)):
            raise ValueError("Coordinates fixed by the encoder must be constant across root histories")
        mean[structural] = y[0, structural]
        encoder = cls(codes.shape[1], mean.to(device=states.device, dtype=states.dtype),
                      scale.to(device=states.device, dtype=states.dtype), mask.to(states.device))
        design = torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], 1)
        target = ((y-mean)/scale)[:, mask]
        coefficient = torch.linalg.lstsq(design, target, rcond=1e-10).solution.to(states)
        with torch.no_grad():
            encoder.linear.weight.copy_(coefficient[:-1].T)
            encoder.linear.bias.copy_(coefficient[-1])
        return encoder


class InverseEncoder(nn.Module):
    """Differentiable decoder inversion by correcting a base writer's values.

    Keys and all other coordinates must be fixed by the base writer. The known
    decoder remains frozen. Attention and MLP factors are retained as buffers,
    so each damped Gauss--Newton correction requires only a code-sized solve.
    """
    def __init__(self, base, attention, input_weight, hidden_bias, output_weight, output_bias,
                 *, steps=5, damping=1e-6):
        super().__init__()
        if not isinstance(steps, int) or steps < 1 or not math.isfinite(damping) or damping <= 0:
            raise ValueError("Inverse steps and damping must be positive")
        self.base, self.steps, self.damping = base, steps, damping
        for name, value in [("attention", attention), ("input_weight", input_weight),
                            ("hidden_bias", hidden_bias), ("output_weight", output_weight),
                            ("output_bias", output_bias)]:
            self.register_buffer(name, value.detach().clone())

    @classmethod
    @torch.no_grad()
    def from_decoder(cls, adapter, decoder, base, *, steps=5, damping=1e-6):
        if any(parameter.requires_grad for parameter in decoder.parameters()):
            raise ValueError("Inverse construction requires a frozen decoder")
        reference = base.mean[None].clone()
        expected = adapter.value_mask(reference, layers=decoder.layers)
        if not torch.equal(base.writable, expected) or not bool(expected.any()):
            raise ValueError("The base writer must write exactly the decoder's selected value coordinates")
        attention = adapter.attention_weights(reference, decoder.queries, layers=decoder.layers)[0]
        reference[:, expected] = 0
        features = (adapter.features(reference, decoder.queries) if decoder.layers is None else
                    adapter.features(reference, decoder.queries, layers=decoder.layers))
        first, _, last = decoder.net
        return cls(base, attention, first.weight[:, adapter.token_features:], first(features)[0],
                   last.weight, last.bias, steps=steps, damping=damping)

    def forward(self, codes):
        state = self.base(codes)
        layers, heads, queries, positions = self.attention.shape
        head_dim = self.input_weight.shape[1]//(layers*heads*queries)
        values = state[:, self.base.writable].reshape(len(codes), positions, layers, heads, head_dim)
        identity = torch.eye(codes.shape[1], device=codes.device, dtype=codes.dtype)
        for _ in range(self.steps):
            features = torch.einsum("lhqt,btlhd->blhqd", self.attention, values).flatten(1)
            hidden = torch.tanh(features@self.input_weight.T+self.hidden_bias)
            prediction = hidden@self.output_weight.T+self.output_bias
            feature_jacobian = (self.output_weight[None]*(1-hidden.square())[:, None])@self.input_weight
            jacobian = torch.einsum("bzlhqd,lhqt->bztlhd",
                feature_jacobian.reshape(len(codes), codes.shape[1], layers, heads, queries, head_dim),
                self.attention).flatten(2)
            system = jacobian@jacobian.transpose(1, 2)+self.damping*identity
            correction = jacobian.transpose(1, 2)@torch.linalg.solve(system, (codes-prediction)[:, :, None])
            values = values+correction.reshape_as(values)
        result = state.clone()
        result[:, self.base.writable] = values.flatten(1)
        return result


class GBSAbstraction(nn.Module):
    def __init__(self, adapter, decoder, encoder, architecture):
        super().__init__()
        self.adapter, self.decoder, self.encoder = adapter, decoder, encoder
        self.architecture = architecture

    def decode(self, states):
        return self.decoder(self.adapter, states)

    def encode(self, codes):
        return self.encoder(codes)

    def predict(self, codes):
        """Induced prediction map: predict from a freshly written code."""
        return self.adapter.predict(self.encode(codes))

    def advance(self, codes, tokens):
        """Induced update: write, run the frozen teacher once, read back."""
        log_probabilities, successor = self.adapter.advance(self.encode(codes), tokens)
        return log_probabilities, self.decode(successor)

    def repeated(self, codes, tokens):
        """Log probabilities along repeated write/update/decode steps."""
        probabilities = []
        for token in tokens.unbind(1):
            lp, codes = self.advance(codes, token)
            probabilities.append(lp)
        return torch.stack(probabilities, 1)

    def save(self, path):
        """Save only the abstraction; a matching frozen adapter is required to load."""
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        encoder_spec = (dict(type="inverse", steps=self.encoder.steps, damping=self.encoder.damping)
                        if isinstance(self.encoder, InverseEncoder) else dict(type="affine"))
        torch.save(dict(format_version=2, architecture=asdict(self.architecture), encoder_spec=encoder_spec,
                        teacher_fingerprint=self.adapter.fingerprint(), adapter_spec=self.adapter.spec(),
                        decoder=self.decoder.state_dict(), encoder=self.encoder.state_dict()), path)

    @classmethod
    def load(cls, path, adapter):
        saved = torch.load(path, map_location=adapter.device, weights_only=True)
        if saved["format_version"] != 2:
            raise ValueError("Unsupported abstraction format; refit using normalized-attention decoder inputs")
        if saved["teacher_fingerprint"] != adapter.fingerprint():
            raise ValueError("The abstraction does not match this teacher/adapter")
        architecture = Architecture(**saved["architecture"])
        decoder = AttentionDecoder(adapter, architecture)
        state = saved["encoder"]
        decoder.load_state_dict(saved["decoder"])
        spec = saved.get("encoder_spec", dict(type="affine"))
        if spec["type"] == "affine":
            encoder = AffineEncoder(architecture.dimension, state["mean"], state["scale"], state["writable"])
        elif spec["type"] == "inverse":
            base = AffineEncoder(architecture.dimension, state["base.mean"], state["base.scale"], state["base.writable"])
            encoder = InverseEncoder(base, state["attention"], state["input_weight"], state["hidden_bias"],
                state["output_weight"], state["output_bias"], steps=spec["steps"], damping=spec["damping"])
            decoder.requires_grad_(False)
        else:
            raise ValueError(f"Unsupported encoder type: {spec['type']}")
        encoder.load_state_dict(state)
        return cls(adapter, decoder, encoder, architecture).eval()
