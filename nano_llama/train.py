from dataclasses import dataclass
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree, Array, Int, PRNGKeyArray, Float
from typing import Any, Optional, Tuple, Union
import optax

from nano_llama.llama import Llama, LlamaConfig, FaultSpec


def get_param_labels(model: Llama) -> PyTree:
    """Optimizer-group label for each array leaf of `model`."""
    arrays = eqx.filter(model, eqx.is_array)
    labels = jax.tree_util.tree_map(lambda _: "frozen", arrays)
    labels = eqx.tree_at(lambda m: m.tok_embeddings.weight, labels, "decay")
    if model.lm_head is not None:                       # untied readout; tied reuses tok_embeddings
        labels = eqx.tree_at(lambda m: m.lm_head.weight, labels, "decay")
    labels = eqx.tree_at(lambda m: m.norm.weight, labels, "no_decay")
    for i in range(len(model.layers)):
        labels = eqx.tree_at(lambda m: m.layers[i].attention.wq.weight, labels, "decay")
        labels = eqx.tree_at(lambda m: m.layers[i].attention.wk.weight, labels, "decay")
        labels = eqx.tree_at(lambda m: m.layers[i].attention.wv.weight, labels, "decay")
        labels = eqx.tree_at(lambda m: m.layers[i].attention.wo.weight, labels, "decay")
        labels = eqx.tree_at(lambda m: m.layers[i].feed_forward.w1.weight, labels, "decay")
        labels = eqx.tree_at(lambda m: m.layers[i].feed_forward.w2.weight, labels, "decay")
        labels = eqx.tree_at(lambda m: m.layers[i].feed_forward.w3.weight, labels, "decay")
        labels = eqx.tree_at(lambda m: m.layers[i].attention_norm.weight, labels, "no_decay")
        labels = eqx.tree_at(lambda m: m.layers[i].ffn_norm.weight, labels, "no_decay")
    return labels


def build_standard_optimizer(schedule, model: Llama, b1: float, b2: float, eps: float,
                             weight_decay: float, grad_clip: float) -> optax.GradientTransformation:
    """AdamW with global-norm clipping, one schedule for every parameter."""
    def group_tx(wd: float) -> optax.GradientTransformation:
        return optax.adamw(learning_rate=schedule, b1=b1, b2=b2, eps=eps, weight_decay=wd)

    txs = {"decay": group_tx(weight_decay), "no_decay": group_tx(0.0), "frozen": optax.set_to_zero()}
    # Pass get_param_labels as a CALLABLE, not a precomputed pytree: the label tree is
    # itself a Llama module, hence callable, and optax would mistake it for a
    # param_labels fn and call it on the params.
    return optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.multi_transform(txs, get_param_labels),
    )


def compute_loss(model: Llama, x: Int[Array, "batch block_size"], y: Int[Array, "batch block_size"],
                 keys: PRNGKeyArray, spec: FaultSpec) -> Float[Array, ""]:
    # `keys` is one fault key per sequence, (batch, 2), assigned by global batch
    # position and delivered already-sharded -- so a sequence's fault draw depends
    # on its global index alone, not on the device count or which shard it lands on.
    def forward_single(x_i: Int[Array, "block_size"], key_i: PRNGKeyArray) -> Float[Array, "block_size vocab_size"]:
        return model(x_i, spec=spec, key=key_i)

    logits = jax.vmap(forward_single)(x, keys)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    return jnp.mean(loss)


def step(model: Llama, opt_state: optax.OptState, x: Int[Array, "batch block_size"], y: Int[Array, "batch block_size"],
         keys: PRNGKeyArray, optim: optax.GradientTransformation, accum_steps: int, spec: FaultSpec,
         axis_name: Optional[str] = None) -> Tuple[Llama, optax.OptState, Float[Array, ""]]:
    # Reshape the full batch into (accum_steps, micro_batch_size, seq_len). `keys`
    # reshapes the same way, so each sequence stays paired with its own key.
    micro_batch_size = x.shape[0] // accum_steps
    x_micro = x.reshape(accum_steps, micro_batch_size, x.shape[1])
    y_micro = y.reshape(accum_steps, micro_batch_size, y.shape[1])
    keys_micro = keys.reshape(accum_steps, micro_batch_size, keys.shape[-1])

    grad_f = eqx.filter_value_and_grad(compute_loss)

    def micro_step(carry, xs):
        x_i, y_i, keys_i = xs
        loss, grads = grad_f(model, x_i, y_i, keys_i, spec)

        # compute_loss returns the micro-batch MEAN, so scale before accumulating.
        loss = loss / accum_steps
        grads = jax.tree_util.tree_map(lambda g: g / accum_steps, grads)

        acc_loss, acc_grads = carry
        acc_loss = acc_loss + loss
        acc_grads = jax.tree_util.tree_map(lambda a, b: a + b, acc_grads, grads)

        return (acc_loss, acc_grads), None

    dynamic_model, _ = eqx.partition(model, eqx.is_array)
    init_grads = jax.tree_util.tree_map(jnp.zeros_like, dynamic_model)
    init_carry = (jnp.array(0.0), init_grads)

    (total_loss, total_grads), _ = jax.lax.scan(micro_step, init_carry, (x_micro, y_micro, keys_micro))

    # Each device accumulated the mean grad over its own shard; averaging across the
    # axis gives the global-batch mean, so every replica applies the same update and
    # they stay bit-identical. No-op when axis_name is None.
    if axis_name is not None:
        total_grads = jax.lax.pmean(total_grads, axis_name)
        total_loss = jax.lax.pmean(total_loss, axis_name)

    updates, opt_state = optim.update(total_grads, opt_state, dynamic_model)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, total_loss


def per_sequence_loss(model: Llama, x: Int[Array, "batch block_size"], y: Int[Array, "batch block_size"],
                      spec: Optional[FaultSpec] = None,
                      keys: Optional[PRNGKeyArray] = None) -> Float[Array, "batch"]:
    """Mean CE for each sequence in the batch -> (batch,)."""
    if spec is not None and keys is not None:
        logits = jax.vmap(lambda x_i, k_i: model(x_i, spec=spec, key=k_i))(x, keys)
    else:
        logits = jax.vmap(lambda x_i: model(x_i, spec=None, key=None))(x)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)   # (batch, block_size)
    return jnp.mean(loss, axis=-1)
