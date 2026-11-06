"""Convert belief-state RoPE weights into an unmodified Hugging Face model."""
from __future__ import annotations

import hashlib
import json
import torch


def _source_fingerprint(config, weights):
    spec = dict(adapter="BeliefStateAdapter", vocab=config["vocab"], layers=config["n_layers"],
        heads=config["n_heads"], head_dim=config["d_head"], token_features=config["vocab"]+1,
        max_history=config["context"], layout="kv-token-length-v1",
        config=dict(attention_mass=False, attention_sums=False, **config))
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode())
    for name, value in sorted(weights.items()):
        value = value.detach().cpu().contiguous()
        digest.update(f"model.{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def convert_belief_state_checkpoint(checkpoint, *, attn_implementation="sdpa"):
    """Return a native Starcoder2ForCausalLM with equivalent ordinary-token logits.

    No transformer forward pass is implemented here. The conversion splits QKV
    projections, copies LayerNorm/ReLU MLP weights, and absorbs output bias into
    the final LayerNorm bias. A native head_dim preserves narrow heads.
    The input-only BOS is excluded from the modeled output alphabet through
    configuration metadata read by TransformersAdapter. Save with the model's
    normal save_pretrained method for a standard config/safetensors folder.
    Learned-position teachers should be supplied as native HF checkpoints.
    """
    from transformers import Starcoder2Config, Starcoder2ForCausalLM

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or not {"config", "model"} <= saved.keys():
        raise ValueError("Expected a belief-state checkpoint with config and model state dictionary")
    config = dict(saved["config"])
    for key in ("attention_mass", "attention_sums"):
        if config.pop(key, False):
            raise ValueError("Only standard softmax attention is supported")
    defaults = dict(vocab=2, context=32, d_model=64, n_layers=4, n_heads=1,
                    d_head=8, d_mlp=256, rope=False, bos_visible=False)
    if config.keys()-defaults.keys():
        raise ValueError(f"Unknown architecture fields: {sorted(config.keys()-defaults.keys())}")
    c = defaults | config
    if not c["rope"]:
        raise ValueError("The belief-state converter supports RoPE; use a native HF checkpoint for learned positions")
    native = Starcoder2Config(vocab_size=c["vocab"]+1, hidden_size=c["d_model"],
        intermediate_size=c["d_mlp"], num_hidden_layers=c["n_layers"],
        num_attention_heads=c["n_heads"], num_key_value_heads=c["n_heads"], head_dim=c["d_head"],
        hidden_act="relu", max_position_embeddings=c["context"], norm_epsilon=1e-5,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.},
        bos_token_id=c["vocab"], eos_token_id=None, tie_word_embeddings=False,
        attention_dropout=0., residual_dropout=0., embedding_dropout=0., use_bias=True)
    native.gbs = dict(token_ids=list(range(c["vocab"])), attend_to_bos=c["bos_visible"],
                      source_fingerprint=_source_fingerprint(c, saved["model"]))
    model = Starcoder2ForCausalLM(native).to(dtype=saved["model"]["embed.weight"].dtype)
    model.set_attn_implementation(attn_implementation)
    source = dict(saved["model"])

    def take(name):
        try:
            return source.pop(name)
        except KeyError as error:
            raise ValueError(f"Missing transformer weight: {name}") from error

    def linear(destination, prefix):
        destination.weight.copy_(take(prefix+".weight"))
        destination.bias.copy_(take(prefix+".bias"))

    with torch.no_grad():
        model.model.embed_tokens.weight.copy_(take("embed.weight"))
        for i, layer in enumerate(model.model.layers):
            prefix = f"blocks.{i}."
            linear(layer.input_layernorm, prefix+"ln1")
            linear(layer.post_attention_layernorm, prefix+"ln2")
            linear(layer.mlp.c_fc, prefix+"mlp.0")
            linear(layer.mlp.c_proj, prefix+"mlp.2")
            linear(layer.self_attn.o_proj, prefix+"out")
            weight = take(prefix+"qkv.weight").reshape(3, -1, c["d_model"])
            bias = take(prefix+"qkv.bias").reshape(3, -1)
            for index, projection in enumerate((layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj)):
                projection.weight.copy_(weight[index])
                projection.bias.copy_(bias[index])
        model.model.norm.weight.copy_(take("ln_final.weight"))
        final_bias, output_weight, output_bias = take("ln_final.bias"), take("unembed.weight"), take("unembed.bias")
        shift = torch.linalg.lstsq(output_weight.double(), output_bias.double(), rcond=1e-10).solution
        if not torch.allclose(output_weight.double()@shift, output_bias.double(), atol=1e-8, rtol=1e-6):
            raise ValueError("The output bias cannot be represented by the native bias-free LM head")
        model.model.norm.bias.copy_(final_bias+shift.to(final_bias))
        model.lm_head.weight[:c["vocab"]].copy_(output_weight)
        model.lm_head.weight[c["vocab"]:].zero_()
    if source:
        raise ValueError(f"Unexpected transformer weights: {sorted(source)}")
    model.generation_config.suppress_tokens = [c["vocab"]]
    return model.eval().requires_grad_(False)


def convert_belief_state_abstraction(checkpoint, adapter):
    """Port a fitted abstraction to the converted teacher's native cache/embedding layout.

    The source checkpoint is left intact. Cache coordinates retain their order;
    the token writer is composed with the embedding matrix, and the decoder's
    first layer with its right inverse. No fitting is performed. This requires
    an ordinary full-cache encoder and linearly independent token embeddings.
    """
    from ..models import Architecture, AttentionDecoder, AffineEncoder, GBSAbstraction

    saved = torch.load(checkpoint, map_location=adapter.device, weights_only=True)
    source_id = getattr(adapter.model.config, "gbs", {}).get("source_fingerprint")
    if source_id is None or saved["teacher_fingerprint"] != source_id:
        raise ValueError("The source abstraction does not match the converted teacher")
    fields = dict(saved["architecture"])
    if not fields.pop("learn_queries", True):
        raise ValueError("The source decoder must use learned queries")
    architecture = Architecture(**fields)
    decoder = AttentionDecoder(adapter, architecture)
    old = saved["encoder"]
    if not bool(old["writable"][:-1].all()) or bool(old["writable"][-1]):
        raise ValueError("The source encoder must write the complete cache and token")
    embedding = adapter.model.get_input_embeddings().weight.detach()
    if len(embedding) != adapter.vocab_size+1 or torch.linalg.matrix_rank(embedding.double()) != len(embedding):
        raise ValueError("Conversion requires independent ordinary-token and BOS embeddings")
    token_width = len(embedding)
    cache_width = len(old["mean"])-token_width-1
    scale = old["scale"][:-1]
    weight = scale[:, None]*old["linear.weight"]
    bias = old["mean"][:-1]+scale*old["linear.bias"]
    cache_weight, cache_bias = weight[:cache_width], bias[:cache_width]
    if not adapter.attend_to_bos:
        bos_state = adapter.history_state(torch.zeros(1, 1, dtype=torch.long, device=adapter.device))
        cache_weight = torch.cat([weight.new_zeros(adapter.entry_width, architecture.dimension), cache_weight])
        cache_bias = torch.cat([bos_state[0, :adapter.entry_width], cache_bias])
    if cache_weight.shape[0] != int(old["mean"][-1])*adapter.entry_width:
        raise ValueError("Source cache shape and history length disagree")
    weight = torch.cat([cache_weight, embedding.T@weight[cache_width:]])
    bias = torch.cat([cache_bias, embedding.T@bias[cache_width:]])
    encoder = AffineEncoder(architecture.dimension, bias.new_zeros(len(bias)),
        bias.new_ones(len(bias)), torch.ones(len(bias), device=bias.device, dtype=torch.bool))
    parameters = dict(saved["decoder"])
    readout = parameters["net.0.weight"]
    inverse = torch.linalg.pinv(embedding.double(), rtol=1e-10).to(embedding)
    parameters["net.0.weight"] = torch.cat([readout[:, :token_width]@inverse.T, readout[:, token_width:]], 1)
    decoder.load_state_dict(parameters)
    with torch.no_grad():
        encoder.linear.weight.copy_(weight)
        encoder.linear.bias.copy_(bias)
    return GBSAbstraction(adapter, decoder, encoder, architecture).eval()
