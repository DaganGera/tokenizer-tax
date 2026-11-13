"""Teacher-forced incremental decoding: feed reference tokens one at a time through the KV cache."""
import torch

PROMPT = "Translate to the target language:\n"   # short fixed prompt (same for every language)

@torch.inference_mode()
def prefill(model, prompt_ids):
    out = model(prompt_ids, use_cache=True)
    return out.past_key_values

@torch.inference_mode()
def teacher_forced_decode(model, cache, ref_ids):
    """One forward pass per reference token, batch size 1. Returns nothing: we measure cost, not outputs."""
    for t in ref_ids:
        out = model(torch.tensor([[t]], device=model.device), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
    return cache
