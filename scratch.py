from functools import partial

from jax import random
from jax import lax
from jax.numpy import allclose
from jax import jit
import haliax as hax

from levanter.models import gpt2, llama, gemma
from levanter.layers.attention import AttentionMask, AttentionBackend

import scanagram


vocab_size = 32
seq_length = 1024
Vocab = hax.Axis("vocab", vocab_size)
Pos = hax.Axis("position", seq_length)
k = random.PRNGKey(0)
k_init, k_inputs, k_eval = random.split(k, 3)
causal_mask = AttentionMask.causal()
input_ids = random.randint(k_inputs, seq_length, 0, vocab_size)

############################### GPT-2 #########################################
#config = gpt2.Gpt2Config(attn_backend=AttentionBackend.VANILLA)
#model = gpt2.Gpt2LMHeadModel.init(Vocab, config, key=k_init)
#
#def gpt2_eval(input_ids):
#    input_ids = hax.named(input_ids, (Pos,))
#    out = model(input_ids, attn_mask=causal_mask, key=k_eval, inference=True)
#    assert out.axes == (Pos, Vocab)
#    return out.array
#
#out = gpt2_eval(input_ids)
#
#body_fn, carry_init = scanagram.as_scan(gpt2_eval, input_ids)
#
#_, out_scanagram = lax.scan(body_fn, carry_init, input_ids)
#assert allclose(out, out_scanagram)

############################### Llama #########################################
#config = llama.LlamaConfig(attn_backend=AttentionBackend.VANILLA)
#model = llama.LlamaLMHeadModel.init(Vocab, config, key=k_init)
#
#@jit
#def llama_eval(model, input_ids):
#    input_ids = hax.named(input_ids, (Pos,))
#    out = model(input_ids, attn_mask=causal_mask, key=k_eval)
#    assert out.axes == (Pos, Vocab)
#    return out.array
#
#out = llama_eval(model, input_ids)
#
#body_fn, carry_init = scanagram.as_scan(partial(llama_eval, model), input_ids)
#_, out_scanagram = lax.scan(body_fn, carry_init, input_ids)
#assert allclose(out, out_scanagram)

############################### Gemma #########################################
config = gemma.GemmaConfig(attn_backend=AttentionBackend.VANILLA)
model = gemma.GemmaLMHeadModel.init(Vocab, config, key=k_init)

@jit
def gemma_eval(model, input_ids):
    input_ids = hax.named(input_ids, (Pos,))
    out = model(input_ids, attn_mask=causal_mask, key=k_eval)
    assert out.axes == (Pos, Vocab)
    return out.array

out = gemma_eval(model, input_ids)

body_fn, carry_init = scanagram.as_scan(partial(gemma_eval, model), input_ids)
_, out_scanagram = lax.scan(body_fn, carry_init, input_ids)
assert allclose(out, out_scanagram)
