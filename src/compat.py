"""Workaround for a transformers 5.23 bug: Gemma4 text config rejects the MTP drafter, whose layers ALL reuse the target's KV cache."""
def patch_gemma4_assistant():
    from transformers.models.gemma4 import configuration_gemma4 as c
    orig = c.Gemma4TextConfig.kv_sharing_modes.fget
    def modes(self):
        if self.num_kv_shared_layers == self.num_hidden_layers: return ["reuse"] * self.num_hidden_layers
        return orig(self)
    c.Gemma4TextConfig.kv_sharing_modes = property(modes)
