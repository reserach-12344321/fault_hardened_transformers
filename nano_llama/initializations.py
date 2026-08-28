"""Apply the standard (llama2.c / GPT-2) initialization to a Llama skeleton.

llama.py builds models with default equinox weights; this overwrites them in one pure
eqx.tree_at pass. Most 2D weights draw at init_std; the two residual projections
(attention.wo, feed_forward.w3) draw at init_std / sqrt(2 * n_layer), holding
residual-stream variance roughly constant as depth grows; RMSNorm gains are left at ones.

llama2.c applies the depth factor to w3, the SwiGLU up-projection rather than the
down-projection w2, and that is reproduced verbatim here.
"""
import math

import jax
import jax.numpy as jnp
import equinox as eqx

from nano_llama.llama import Llama, LlamaConfig, INIT_STD


def _plain_weight_leaves(model: Llama) -> list:
    """The 2D weights drawn at the plain std, in a fixed order so keys map stably."""
    leaves = []
    for blk in model.layers:
        a = blk.attention
        leaves += [a.wq.weight, a.wk.weight, a.wv.weight]
        f = blk.feed_forward
        leaves += [f.w1.weight, f.w2.weight]
    leaves.append(model.tok_embeddings.weight)
    if not model.config.tie_embeddings:
        leaves.append(model.lm_head.weight)     # untied readout; tied reuses tok_embeddings
    return leaves


def _residual_weight_leaves(model: Llama) -> list:
    """The residual projections, which carry the 1/sqrt(2*n_layer) depth factor."""
    leaves = []
    for blk in model.layers:
        leaves += [blk.attention.wo.weight, blk.feed_forward.w3.weight]
    return leaves


def standard_init_stds(config: LlamaConfig, init_std: float = INIT_STD) -> dict:
    """The init stds this config draws with: plain, residual, and the norm gains."""
    return dict(plain=float(init_std),
                residual=init_std / math.sqrt(2 * config.n_layer),
                norm=1.0)


def standard_init(model: Llama, key: jax.Array, init_std: float = INIT_STD) -> Llama:
    """A copy of skeleton `model` with standard llama2.c-initialized weights."""
    stds = standard_init_stds(model.config, init_std)
    plain = _plain_weight_leaves(model)
    residual = _residual_weight_leaves(model)
    keys = jax.random.split(key, len(plain) + len(residual))

    new_plain = [jax.random.normal(k, w.shape, w.dtype) * stds["plain"]
                 for k, w in zip(keys[:len(plain)], plain)]
    new_residual = [jax.random.normal(k, w.shape, w.dtype) * stds["residual"]
                    for k, w in zip(keys[len(plain):], residual)]

    model = eqx.tree_at(_plain_weight_leaves, model, new_plain)
    model = eqx.tree_at(_residual_weight_leaves, model, new_residual)
    return model


def build_model(config: LlamaConfig, key: jax.Array, init_std: float = INIT_STD) -> Llama:
    """Build a Llama skeleton from `config` and apply the standard init."""
    skel_key, init_key = jax.random.split(key)
    return standard_init(Llama(config, skel_key), init_key, init_std=init_std)
