"""Frozen Hugging Face models with native, differentiable DynamicCache states."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, DynamicCache

from .base import CacheModel
from ..data import prepare_histories, token_ids


class TransformersAdapter(CacheModel):
    """A native causal LM, its standard KV cache, and an unprocessed input embedding.

    Fitting flattens cache tensors in (position, layer, KV head, key/value,
    head coordinate) order, followed by the current input embedding. There is
    no position coordinate. The native model infers positions from cache length.
    ``state_from_cache`` and ``cache_from_state`` connect this numerical fitting
    representation to the library's public cache API without detaching gradients.

    Supported models use a full, uniform DynamicCache and either ordinary
    half-split RoPE or learned absolute positions. Sliding, hybrid, quantized,
    and state-space caches need a different adapter. ``token_ids`` optionally
    restricts the modeled alphabet, mapping GBS IDs to model vocabulary IDs.
    """
    def __init__(self, model, *, bos_token_id=None, token_ids=None, attend_to_bos=None,
                 attn_implementation="sdpa"):
        super().__init__()
        cfg = model.config
        if cfg.model_type not in ("llama", "starcoder2", "gpt2", "gpt_neox", "gpt_bigcode"):
            raise ValueError("This adapter supports Llama, Starcoder2, GPT-2, GPT-NeoX, and GPTBigCode")
        if (getattr(cfg, "sliding_window", None) is not None or
                any(kind != "full_attention" for kind in getattr(cfg, "layer_types", []) or [])):
            raise ValueError("Only full attention with uniform DynamicCache layers is supported")
        self.model = model.eval().requires_grad_(False)
        model.set_attn_implementation(attn_implementation)
        settings = getattr(cfg, "gbs", {})
        self.bos_token_id = cfg.bos_token_id if bos_token_id is None else bos_token_id
        if self.bos_token_id is None or not 0 <= self.bos_token_id < cfg.vocab_size:
            raise ValueError("Provide a valid bos_token_id for the initial state")
        ids = token_ids if token_ids is not None else settings.get("token_ids", list(range(cfg.vocab_size)))
        if (not ids or len(set(ids)) != len(ids) or
                any(not isinstance(i, int) or not 0 <= i < cfg.vocab_size for i in ids)):
            raise ValueError("token_ids must be distinct model vocabulary IDs")
        self.register_buffer("token_ids", torch.tensor(ids, dtype=torch.long, device=self.device))
        self.vocab_size = len(ids)
        self.attend_to_bos = settings.get("attend_to_bos", True) if attend_to_bos is None else attend_to_bos
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.token_features = model.get_input_embeddings().weight.shape[1]
        self.max_history = cfg.max_position_embeddings
        self.entry_width = self.n_layers*self.n_heads*2*self.head_dim
        self.rope = hasattr(model.base_model, "rotary_emb")
        rope = getattr(cfg, "rope_parameters", None) or {}
        if self.rope and rope.get("rope_type", "default") not in ("default", "linear", "llama3"):
            raise ValueError("RoPE must not change cached-key rotations as sequence length grows")
        self.train(False)

    @classmethod
    def from_checkpoint(cls, checkpoint, *, device="cpu", attn_implementation="sdpa"):
        """Load a local save_pretrained directory or convert a belief-state RoPE file."""
        checkpoint = Path(checkpoint)
        if checkpoint.is_file():
            from .belief_state import convert_belief_state_checkpoint
            model = convert_belief_state_checkpoint(checkpoint, attn_implementation=attn_implementation)
        else:
            model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True,
                dtype="auto", attn_implementation=attn_implementation, trust_remote_code=False)
        return cls(model.to(device), attn_implementation=attn_implementation)

    def train(self, mode=True):
        super().train(False)
        return self

    def spec(self):
        config = {key: value for key, value in self.model.config.to_dict().items()
                  if not key.startswith("_") and key not in ("transformers_version", "dtype", "torch_dtype", "architectures", "gbs")}
        return dict(**super().spec(), layout="hf-kv-embedding-v1", config=config,
                    bos_token_id=self.bos_token_id, token_ids=self.token_ids.cpu().tolist(),
                    attend_to_bos=self.attend_to_bos)

    def _unpack(self, states):
        width = states.shape[1]-self.token_features
        if width < 0 or width % self.entry_width:
            raise ValueError("Invalid packed native cache width")
        kv = states[:, :width].reshape(len(states), -1, self.n_layers, self.n_heads, 2, self.head_dim)
        return kv[..., 0, :], kv[..., 1, :], states[:, width:]

    def state_from_cache(self, cache, current_embedding):
        """Flatten a native cache and the next input embedding, preserving autograd."""
        if not isinstance(cache, DynamicCache):
            raise ValueError("Provide a full, non-sliding DynamicCache")
        count = cache.get_seq_length()
        if count:
            if len(cache.layers) != self.n_layers or any(layer.is_sliding for layer in cache.layers):
                raise ValueError("Cache layers must be uniform full-attention layers")
            keys, values = [], []
            expected = (len(current_embedding), self.n_heads, count, self.head_dim)
            for layer in cache.layers:
                if layer.keys.shape != expected or layer.values.shape != expected:
                    raise ValueError("Cache tensor shape does not match this model")
                keys.append(layer.keys.transpose(1, 2))
                values.append(layer.values.transpose(1, 2))
            kv = torch.stack([torch.stack(keys, 2), torch.stack(values, 2)], -2).flatten(1)
        else:
            kv = current_embedding.new_empty(len(current_embedding), 0)
        if current_embedding.shape != (len(current_embedding), self.token_features):
            raise ValueError("Provide one current input embedding per cache row")
        return torch.cat([kv, current_embedding], 1)

    def cache_from_state(self, states):
        """Return a fresh native cache and current embedding; model updates cannot mutate states."""
        keys, values, current = self._unpack(states)
        cache = DynamicCache(ddp_cache_data=[(keys[:, :, layer].transpose(1, 2),
            values[:, :, layer].transpose(1, 2)) for layer in range(self.n_layers)])
        return cache, current

    def _embeddings(self, tokens):
        return self.model.get_input_embeddings()(self.token_ids[tokens])

    def _run(self, cache, embeddings):
        past = cache.get_seq_length()
        horizon = embeddings.shape[1]
        if horizon < 1 or past+horizon > self.max_history:
            raise ValueError("Forecast must be nonempty and remain within the teacher context")
        mask = None
        if not self.attend_to_bos:
            # Historical masked-BOS teachers use the library's ordinary mask
            # input; their BOS entry remains in the native cache at every step.
            queries = past+torch.arange(horizon, device=embeddings.device)
            columns = torch.arange(past+horizon, device=embeddings.device)
            allowed = (columns[None] <= queries[:, None]) & ((columns[None] != 0) | (queries[:, None] == 0))
            mask = embeddings.new_zeros(1, 1, horizon, past+horizon).masked_fill(~allowed, -torch.inf)
        output = self.model(inputs_embeds=embeddings, past_key_values=cache,
            use_cache=True, attention_mask=mask, return_dict=True)
        return output.logits.index_select(-1, self.token_ids).log_softmax(-1), output.past_key_values

    @torch.no_grad()
    def history_state(self, histories):
        histories = prepare_histories(histories, self.vocab_size, device=self.device)
        batch, n = histories.shape
        if n > self.max_history:
            raise ValueError("History exceeds context")
        bos = self.model.get_input_embeddings().weight[self.bos_token_id].expand(batch, -1)
        cache = DynamicCache()
        if n == 0:
            return self.state_from_cache(cache, bos)
        embeddings = torch.cat([bos[:, None], self._embeddings(histories[:, :-1])], 1)
        _, cache = self._run(cache, embeddings)
        return self.state_from_cache(cache, self._embeddings(histories[:, -1]))

    def trajectory(self, states, tokens):
        tokens = prepare_histories(tokens, self.vocab_size, device=states.device)
        if tokens.shape[1] == 0:
            raise ValueError("Forecast must be nonempty")
        cache, current = self.cache_from_state(states)
        past = cache.get_seq_length()
        embeddings = torch.cat([current[:, None], self._embeddings(tokens[:, :-1])], 1)
        probabilities, cache = self._run(cache, embeddings)
        successors = []
        for t in range(1, tokens.shape[1]+1):
            prefix = DynamicCache(ddp_cache_data=[(layer.keys[:, :, :past+t], layer.values[:, :, :past+t])
                                                 for layer in cache.layers])
            successors.append(self.state_from_cache(prefix, self._embeddings(tokens[:, t-1])))
        return probabilities, successors

    def advance(self, states, tokens):
        tokens = token_ids(tokens, self.vocab_size, device=states.device)
        probabilities, successors = self.trajectory(states, tokens[:, None])
        return probabilities[:, 0], successors[0]

    def predict(self, states):
        cache, current = self.cache_from_state(states)
        return self._run(cache, current[:, None])[0][:, 0]

    @torch.no_grad()
    def sample(self, states, uniforms):
        tokens, probabilities = [], []
        for uniform in uniforms.unbind(1):
            cache, current = self.cache_from_state(states)
            logp, cache = self._run(cache, current[:, None])
            q = logp[:, 0].exp()
            cdf = q if q.dtype == torch.float64 else q.float()
            token = (uniform[:, None] >= cdf.cumsum(1)).sum(1).clamp_max(self.vocab_size-1)
            states = self.state_from_cache(cache, self._embeddings(token))
            tokens.append(token)
            probabilities.append(q)
        return torch.stack(tokens, 1), torch.stack(probabilities, 1)

    def features(self, states, queries):
        keys, values, current = self._unpack(states)
        batch, count = keys.shape[:2]
        if count:
            q = queries
            if self.rope:
                positions = torch.tensor([[count]], device=states.device)
                cos, sin = self.model.base_model.rotary_emb(current[:1, None], positions)
                width = cos.shape[-1]
                rotated = q[..., :width]
                first, second = rotated.chunk(2, -1)
                rotated = rotated*cos[0, 0]+torch.cat([-second, first], -1)*sin[0, 0]
                q = torch.cat([rotated, q[..., width:]], -1)
            q = q[None].expand(batch, -1, -1, -1, -1).reshape(batch*self.n_layers, self.n_heads, -1, self.head_dim)
            k = keys.permute(0, 2, 3, 1, 4).reshape(batch*self.n_layers, self.n_heads, count, self.head_dim)
            v = values.permute(0, 2, 3, 1, 4).reshape_as(k)
            mask = None
            if not self.attend_to_bos:
                mask = (torch.arange(count, device=states.device) != 0)[None]
            response = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).reshape(batch, -1)
        else:
            response = states.new_zeros(batch, queries.numel())
        return torch.cat([current, response], 1)
