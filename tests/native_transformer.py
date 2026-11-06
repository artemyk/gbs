"""Small off-the-shelf models and an independent native forward reference."""
import torch
from transformers import GPT2Config, GPT2LMHeadModel, Starcoder2Config, Starcoder2ForCausalLM


def make_model(*, rope=True, bos_visible=False, vocab=3, context=12):
    shared = dict(vocab_size=vocab+1, bos_token_id=vocab, eos_token_id=None, tie_word_embeddings=False)
    if rope:
        config = Starcoder2Config(**shared, hidden_size=8, intermediate_size=24,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=context, hidden_act="relu",
            rope_parameters={"rope_type": "default", "rope_theta": 10000.})
        model = Starcoder2ForCausalLM(config)
    else:
        config = GPT2Config(**shared, n_embd=8, n_layer=2, n_head=2, n_inner=24,
            n_positions=context, activation_function="relu", resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.)
        model = GPT2LMHeadModel(config)
    config.gbs = dict(token_ids=list(range(vocab)), attend_to_bos=bos_visible)
    model.set_attn_implementation("sdpa")
    return model.eval().requires_grad_(False)


def native_log_probs(model, histories):
    """Run BOS + all but the final observed token, using only the HF model API."""
    settings = model.config.gbs
    input_ids = torch.cat([torch.full_like(histories[:, :1], model.config.bos_token_id), histories[:, :-1]], 1)
    mask = None
    if not settings["attend_to_bos"]:
        length = histories.shape[1]
        allowed = torch.ones(length, length, dtype=torch.bool, device=histories.device).tril()
        allowed[1:, 0] = False
        mask = model.get_input_embeddings().weight.new_zeros(1, 1, length, length).masked_fill(~allowed, -torch.inf)
    return model(input_ids=input_ids, attention_mask=mask).logits[..., settings["token_ids"]].log_softmax(-1)
