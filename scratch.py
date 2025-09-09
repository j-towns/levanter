from jax import random
from jax import lax
from jax.numpy import allclose
import haliax as hax

from levanter.models import gpt2
from levanter.layers.attention import AttentionMask, AttentionBackend

import scanagram


vocab_size = 32
seq_length = 1024
Vocab = hax.Axis("vocab", vocab_size)
Pos = hax.Axis("position", seq_length)
config = gpt2.Gpt2Config(attn_backend=AttentionBackend.VANILLA)
k = random.PRNGKey(0)
k_init, k_inputs, k_eval = random.split(k, 3)

causal_mask = AttentionMask.causal()
model = gpt2.Gpt2LMHeadModel.init(Vocab, config, key=k_init)

input_ids = random.randint(k_inputs, seq_length, 0, vocab_size)

def gpt2_eval(input_ids):
    input_ids = hax.named(input_ids, (Pos,))
    out = model(input_ids, attn_mask=causal_mask, key=k_eval, inference=True)
    assert out.axes == (Pos, Vocab)
    return out.array

out = gpt2_eval(input_ids)

body_fn, carry_init = scanagram.as_scan(gpt2_eval, input_ids)

_, out_scanagram = lax.scan(body_fn, carry_init, input_ids)
assert allclose(out, out_scanagram)

###############################################################################
prompt_length = 512
prompt = input_ids[:prompt_length]

body_fn, carry_init, out_prefill = scanagram.as_scan_with_prefill(
    gpt2_eval, input_ids, prompt
)
