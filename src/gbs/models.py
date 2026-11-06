"""Attention decoder, affine cache encoder, and their induced recurrent model."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from .adapters import CacheModel
from ._training import require_integer


@dataclass
class Architecture:
    dimension: int = 4
    queries_per_head: int = 8
    hidden: int = 128

    def __post_init__(self):
        for name in ("dimension", "queries_per_head", "hidden"):
            require_integer(name, getattr(self, name), minimum=1)


class AttentionDecoder(nn.Module):
    def __init__(self, adapter: CacheModel, architecture: Architecture):
        super().__init__()
        a = architecture
        self.dimension = a.dimension
        queries = torch.randn(adapter.n_layers, adapter.n_heads, a.queries_per_head, adapter.head_dim,
                              device=adapter.device, dtype=adapter.dtype)
        self.queries = nn.Parameter(queries)
        inputs = adapter.token_features + adapter.n_layers*adapter.n_heads*a.queries_per_head*adapter.head_dim
        self.net = nn.Sequential(nn.Linear(inputs, a.hidden), nn.Tanh(), nn.Linear(a.hidden, a.dimension))
        self.net.to(device=adapter.device, dtype=adapter.dtype)

    def forward(self, adapter, states):
        return self.net(adapter.features(states, self.queries))


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
    def initialize(cls, adapter, states, codes):
        x, y = codes.detach().cpu().double(), states.detach().cpu().double()
        mean, scale = y.mean(0), y.std(0, correction=0)
        scale = torch.where(scale < 1e-3, torch.ones_like(scale), scale)
        mask = adapter.encoder_mask(states).cpu()
        if not bool(torch.allclose(y[:, ~mask], y[:1, ~mask].expand(len(y), -1), atol=1e-6, rtol=1e-6)):
            raise ValueError("Coordinates fixed by the encoder must be constant across root histories")
        mean[~mask] = y[0, ~mask]
        encoder = cls(codes.shape[1], mean.to(device=states.device, dtype=states.dtype),
                      scale.to(device=states.device, dtype=states.dtype), mask.to(states.device))
        design = torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], 1)
        target = ((y-mean)/scale)[:, mask]
        coefficient = torch.linalg.lstsq(design, target, rcond=1e-10).solution.to(states)
        with torch.no_grad():
            encoder.linear.weight.copy_(coefficient[:-1].T)
            encoder.linear.bias.copy_(coefficient[-1])
        return encoder


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
        torch.save(dict(format_version=2, architecture=asdict(self.architecture),
                        teacher_fingerprint=self.adapter.fingerprint(), adapter_spec=self.adapter.spec(),
                        decoder=self.decoder.state_dict(), encoder=self.encoder.state_dict()), path)

    @classmethod
    def load(cls, path, adapter):
        saved = torch.load(path, map_location=adapter.device, weights_only=True)
        if saved["format_version"] != 2:
            raise ValueError("Unsupported abstraction checkpoint format")
        if saved["teacher_fingerprint"] != adapter.fingerprint():
            raise ValueError("The abstraction does not match this teacher/adapter")
        architecture_fields = dict(saved["architecture"])
        # Existing final fits record this property in their checkpoint metadata.
        # It no longer selects a decoder implementation.
        if not architecture_fields.pop("learn_queries", True):
            raise ValueError("The checkpoint does not contain learned decoder queries")
        architecture = Architecture(**architecture_fields)
        decoder = AttentionDecoder(adapter, architecture)
        state = saved["encoder"]
        encoder = AffineEncoder(architecture.dimension, state["mean"], state["scale"], state["writable"])
        decoder.load_state_dict(saved["decoder"])
        encoder.load_state_dict(state)
        return cls(adapter, decoder, encoder, architecture).eval()
