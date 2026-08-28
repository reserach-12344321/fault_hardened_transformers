import math
from dataclasses import replace
from typing import Optional, Tuple, Any
import jax
import jax.numpy as jnp
import equinox as eqx
from dataclasses import dataclass
from jaxtyping import Array, Float, Int, PRNGKeyArray

from nano_llama.config_base import ConfigMixin



INIT_STD = 0.02


@dataclass(frozen=True)
class LlamaConfig(ConfigMixin):
    block_size: int = 256
    vocab_size: int = 32000
    n_layer: int = 6
    n_head: int = 6
    n_kv_head: Optional[int] = None
    n_embd: int = 288
    multiple_of: int = 32
    dtype: str = "float32"        # compute dtype; params stay fp32
    attn_impl: str = "manual"     # "manual" (materialized) or "flash" (cuDNN)
    tie_embeddings: bool = True   # True: readout reuses tok_embeddings; False: separate lm_head


def _resolve_dtype(name: str):
    return jnp.bfloat16 if name == "bfloat16" else jnp.float32


class FaultSpec(eqx.Module):
    """Hardware fault model for a single matmul array."""
    p: Array
    k: int = eqx.field(static=True)

    def __init__(self, p, k: int):
        self.p = jnp.asarray(p, dtype=jnp.float32)
        self.k = k


def _split(key: Optional[PRNGKeyArray], n: int) -> tuple:
    """n subkeys, or n Nones when key is None (faults off)."""
    return (None,) * n if key is None else tuple(jax.random.split(key, n))


def _fault(key: PRNGKeyArray, w: jax.Array, axis: int, block: int, p) -> jax.Array:
    """Zero `block`-sized runs along `axis`."""
    n = w.shape[axis]
    n_blocks = (n + block - 1) // block
    mshape = list(w.shape)
    mshape[axis] = n_blocks
    live = jax.random.uniform(key, tuple(mshape), dtype=jnp.float32) >= p
    live = jnp.repeat(live, block, axis=axis)
    live = jax.lax.slice_in_dim(live, 0, n, axis=axis)
    return w * live.astype(w.dtype)


def _dense(w: jax.Array, x: jax.Array, dtype,
           spec: Optional[FaultSpec] = None, key: Optional[PRNGKeyArray] = None) -> jax.Array:
    """Bias-free dense (x @ w.T) in `dtype`, optionally faulted."""
    w = w.astype(dtype)
    if spec is not None and key is not None:
        w = _fault(key, w, 1, spec.k, spec.p)        # (out, in): contraction = axis 1
    return x.astype(dtype) @ w.T


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tuple[jax.Array, jax.Array]:
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(jnp.float32) / dim))
    t = jnp.arange(end, dtype=jnp.float32)
    freqs = jnp.outer(t, freqs)
    return jnp.cos(freqs), jnp.sin(freqs)


def apply_rotary_emb(xq: jax.Array, xk: jax.Array, freqs_cos: jax.Array, freqs_sin: jax.Array) -> Tuple[
    jax.Array, jax.Array]:
    xq_r = xq.reshape(*xq.shape[:-1], -1, 2)
    xk_r = xk.reshape(*xk.shape[:-1], -1, 2)

    xq_0, xq_1 = xq_r[..., 0], xq_r[..., 1]
    xk_0, xk_1 = xk_r[..., 0], xk_r[..., 1]

    fc = freqs_cos[:, None, :]
    fs = freqs_sin[:, None, :]

    xq_out_0 = xq_0 * fc - xq_1 * fs
    xq_out_1 = xq_0 * fs + xq_1 * fc
    xq_out = jnp.stack([xq_out_0, xq_out_1], axis=-1).reshape(*xq.shape)

    xk_out_0 = xk_0 * fc - xk_1 * fs
    xk_out_1 = xk_0 * fs + xk_1 * fc
    xk_out = jnp.stack([xk_out_0, xk_out_1], axis=-1).reshape(*xk.shape)

    return xq_out, xk_out


class RMSNorm(eqx.Module):
    weight: jax.Array
    eps: float

    def __init__(self, dim: int, eps: float = 1e-5):
        self.eps = eps
        self.weight = jnp.ones(dim)

    def __call__(self, x: jax.Array) -> jax.Array:
        # Always reduce in fp32 (matches llama2.c `_norm(x.float())`), return fp32.
        x32 = x.astype(jnp.float32)
        norm_x = jnp.mean(x32 ** 2, axis=-1, keepdims=True)
        x_normed = x32 * jax.lax.rsqrt(norm_x + self.eps)
        return self.weight * x_normed


def cudnn_flash_ok(head_dim: int, n_head: int, dtype) -> bool:
    """Whether cuDNN's fused attention accepts this head_dim."""
    try:
        q = jnp.zeros((1, 128, n_head, head_dim), dtype)
        out = jax.nn.dot_product_attention(q, q, q, is_causal=True, implementation="cudnn")
        jax.block_until_ready(out)
        return True
    except Exception as e:
        print(f"  (cuDNN flash probe failed: {type(e).__name__}: {str(e)[:120]})")
        return False


def resolve_attn_impl(model_config: LlamaConfig) -> LlamaConfig:
    """Downgrade attn_impl='flash' to 'manual' when cuDNN can't take this head_dim."""
    if model_config.attn_impl == "flash":
        head_dim = model_config.n_embd // model_config.n_head
        cdt = jnp.bfloat16 if model_config.dtype == "bfloat16" else jnp.float32
        if cudnn_flash_ok(head_dim, model_config.n_head, cdt):
            print(f"Attention: cuDNN flash AVAILABLE (head_dim={head_dim}, {model_config.dtype}).")
        else:
            print(f"Attention: cuDNN flash NOT available at head_dim={head_dim} -> using manual attention.")
            model_config = replace(model_config, attn_impl="manual")
    print(f"compute dtype: {model_config.dtype} | attention: {model_config.attn_impl}")
    return model_config


class Attention(eqx.Module):
    wq: eqx.nn.Linear
    wk: eqx.nn.Linear
    wv: eqx.nn.Linear
    wo: eqx.nn.Linear
    n_head: int
    n_kv_head: int
    n_embd: int
    head_dim: int
    compute_dtype: Any = eqx.field(static=True)
    attn_impl: str = eqx.field(static=True)

    def __init__(self, config: LlamaConfig, key: PRNGKeyArray):
        keys = jax.random.split(key, 4)
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.compute_dtype = _resolve_dtype(config.dtype)
        config = resolve_attn_impl(config)
        self.attn_impl = config.attn_impl

        # Skeleton weights only; the real init is applied by initializations.standard_init.
        self.wq = eqx.nn.Linear(config.n_embd, self.n_head * self.head_dim, use_bias=False, key=keys[0])
        self.wk = eqx.nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, use_bias=False, key=keys[1])
        self.wv = eqx.nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, use_bias=False, key=keys[2])
        self.wo = eqx.nn.Linear(self.n_head * self.head_dim, config.n_embd, use_bias=False, key=keys[3])

    def __call__(self, x: Float[Array, "seq_len n_embd"], freqs_cos: jax.Array, freqs_sin: jax.Array,
                 spec: Optional[FaultSpec] = None, key: Optional[PRNGKeyArray] = None) -> Float[Array, "seq_len n_embd"]:
        T, C = x.shape
        cd = self.compute_dtype
        faults = spec is not None and key is not None
        # 6 fault sites: wq, wk, wv, wo weights + the QK^T and AV operand masks.
        kq, kk, kv, ko, kkm, kvm = _split(key, 6)

        q = _dense(self.wq.weight, x, cd, spec, kq).reshape(T, self.n_head, self.head_dim)
        k = _dense(self.wk.weight, x, cd, spec, kk).reshape(T, self.n_kv_head, self.head_dim)
        v = _dense(self.wv.weight, x, cd, spec, kv).reshape(T, self.n_kv_head, self.head_dim)

        # RoPE runs in fp32 (freqs are fp32) for accuracy, then back to compute dtype.
        q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)
        q = q.astype(cd)
        k = k.astype(cd)
        v = v.astype(cd)

        # GQA: replicate kv heads up to n_head (keeps both attention paths simple/correct).
        if self.n_kv_head < self.n_head:
            rep = self.n_head // self.n_kv_head
            k = jnp.repeat(k, rep, axis=1)
            v = jnp.repeat(v, rep, axis=1)

        # Fault the two attention matmuls by masking their STATIONARY operands, before
        # the products -- so the fused and manual paths see identical semantics.
        # Layout is (T, n_head, head_dim).
        #   QK^T: contraction over head_dim, K stationary -> mask K on head_dim (axis 2)
        #   AV:   contraction over key positions, V stationary -> mask V on key pos (axis 0)
        if faults:
            k = _fault(kkm, k, 2, spec.k, spec.p)
            v = _fault(kvm, v, 0, spec.k, spec.p)

        attn_scale = 1.0 / math.sqrt(self.head_dim)
        if self.attn_impl == "flash":
            # dot_product_attention wants (batch, seq, n_head, head_dim); we're inside a
            # per-sequence vmap, so add a singleton batch axis.
            out = jax.nn.dot_product_attention(
                q[None], k[None], v[None], scale=attn_scale, is_causal=True, implementation="cudnn",
            )[0]
            y = out.reshape(T, self.n_head * self.head_dim)
        else:
            q = q.transpose(1, 0, 2)                            # (n_head, T, head_dim)
            k = k.transpose(1, 0, 2)
            v = v.transpose(1, 0, 2)
            att = (q @ k.transpose(0, 2, 1)) * attn_scale
            att = att.astype(jnp.float32)                       # fp32 softmax for numerics
            mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
            att = jnp.where(mask, att, float('-inf'))
            att = jax.nn.softmax(att, axis=-1)
            att = att.astype(cd)
            y = att @ v
            y = y.transpose(1, 0, 2).reshape(T, self.n_head * self.head_dim)

        y = _dense(self.wo.weight, y, cd, spec, ko)
        return y


class FeedForward(eqx.Module):
    w1: eqx.nn.Linear
    w2: eqx.nn.Linear
    w3: eqx.nn.Linear
    compute_dtype: Any = eqx.field(static=True)

    def __init__(self, config: LlamaConfig, key: PRNGKeyArray):
        keys = jax.random.split(key, 3)
        self.compute_dtype = _resolve_dtype(config.dtype)
        hidden_dim = 4 * config.n_embd
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        # Skeleton weights only; standard_init applies the real init (w3 gets the
        # 1/sqrt(2*n_layer) residual-projection scaling, as does attention.wo).
        self.w1 = eqx.nn.Linear(config.n_embd, hidden_dim, use_bias=False, key=keys[0])
        self.w2 = eqx.nn.Linear(hidden_dim, config.n_embd, use_bias=False, key=keys[1])
        self.w3 = eqx.nn.Linear(config.n_embd, hidden_dim, use_bias=False, key=keys[2])

    def __call__(self, x: Float[Array, "seq_len n_embd"], spec: Optional[FaultSpec] = None,
                 key: Optional[PRNGKeyArray] = None) -> Float[Array, "seq_len n_embd"]:
        cd = self.compute_dtype
        k1, k2, k3 = _split(key, 3)
        x1 = _dense(self.w1.weight, x, cd, spec, k1)
        x3 = _dense(self.w3.weight, x, cd, spec, k3)
        h = jax.nn.silu(x1) * x3
        x = _dense(self.w2.weight, h, cd, spec, k2)
        return x


class Block(eqx.Module):
    attention_norm: RMSNorm
    attention: Attention
    ffn_norm: RMSNorm
    feed_forward: FeedForward

    def __init__(self, config: LlamaConfig, key: PRNGKeyArray):
        key1, key2 = jax.random.split(key)
        self.attention_norm = RMSNorm(config.n_embd)
        self.attention = Attention(config, key1)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.feed_forward = FeedForward(config, key2)

    def __call__(self, x: Float[Array, "seq_len n_embd"], freqs_cos: jax.Array, freqs_sin: jax.Array,
                 spec: Optional[FaultSpec] = None, key: Optional[PRNGKeyArray] = None) -> Float[Array, "seq_len n_embd"]:
        k1, k2 = _split(key, 2)
        # Plain residual adds: depth is handled in the init, not by a branch multiplier.
        # Norms and the adds are elementwise, so not on the faulty array.
        x = x + self.attention(jax.vmap(self.attention_norm)(x), freqs_cos, freqs_sin, spec=spec, key=k1)
        x = x + self.feed_forward(jax.vmap(self.ffn_norm)(x), spec=spec, key=k2)
        return x


class Llama(eqx.Module):
    config: LlamaConfig = eqx.field(static=True)
    tok_embeddings: eqx.nn.Embedding
    layers: tuple
    norm: RMSNorm
    lm_head: Optional[eqx.nn.Linear]  # None when tie_embeddings; else (vocab, n_embd) readout
    freqs_cos: jax.Array
    freqs_sin: jax.Array
    compute_dtype: Any = eqx.field(static=True)

    def __init__(self, config: LlamaConfig, key: PRNGKeyArray):
        self.config = config
        # +1 key for the untied head, drawn whether or not it's used so block key
        # indexing stays stable across tied/untied.
        keys = jax.random.split(key, 3 + config.n_layer)
        self.compute_dtype = _resolve_dtype(config.dtype)
        self.tok_embeddings = eqx.nn.Embedding(config.vocab_size, config.n_embd, key=keys[0])

        self.layers = tuple(Block(config, key=k) for k in keys[1:1 + config.n_layer])
        self.norm = RMSNorm(config.n_embd)

        if config.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = eqx.nn.Linear(config.n_embd, config.vocab_size, use_bias=False, key=keys[-1])

        fc, fs = precompute_freqs_cis(config.n_embd // config.n_head, config.block_size)
        self.freqs_cos, self.freqs_sin = fc, fs

    def __call__(self, idx: Int[Array, "seq_len"], spec: Optional[FaultSpec] = None,
                 key: Optional[PRNGKeyArray] = None) -> Float[Array, "seq_len vocab_size"]:
        cd = self.compute_dtype
        T = idx.shape[0]
        # Embedding lookup is a gather (no FMAs) -> not faulted.
        x = jax.vmap(self.tok_embeddings)(idx).astype(cd)

        # One key per block + one for the readout. Each forward draws its masks once
        # and holds them across all T positions -- a single static "chip".
        *block_keys, unembed_key = _split(key, self.config.n_layer + 1)

        fc = self.freqs_cos[:T]
        fs = self.freqs_sin[:T]

        for i, layer in enumerate(self.layers):
            x = layer(x, fc, fs, spec=spec, key=block_keys[i])

        x = jax.vmap(self.norm)(x)  # fp32 out of the final norm

        # Fault-free readout, tied (reuse the embedding) or untied (dedicated lm_head).
        readout_w = self.tok_embeddings.weight if self.config.tie_embeddings else self.lm_head.weight
        logits = _dense(readout_w, x, cd, None, unembed_key)
        return logits.astype(jnp.float32)

    def serialize(self, base_path: str) -> None:
        """Write weights to {base_path}.eqx and the config to {base_path}.json."""
        eqx.tree_serialise_leaves(f"{base_path}.eqx", self)
        self.config.save(f"{base_path}.json")

    @classmethod
    def deserialize(cls, base_path: str) -> "Llama":
        """Rebuild from the {base_path}.json config and {base_path}.eqx weights."""
        config = LlamaConfig.load(f"{base_path}.json")
        dummy_key = jax.random.PRNGKey(0)
        skeleton_model = cls(config, dummy_key)
        return eqx.tree_deserialise_leaves(f"{base_path}.eqx", skeleton_model)

    def count_params(self) -> int:
        """Trainable parameters, excluding the RoPE buffers."""
        arrays = eqx.filter(self, eqx.is_inexact_array)
        n = sum(x.size for x in jax.tree_util.tree_leaves(arrays))
        n -= self.freqs_cos.size + self.freqs_sin.size
        return n

    def count_non_embedding_params(self) -> int:
        """count_params minus the input embedding and, when untied, the lm_head."""
        n = self.count_params() - self.tok_embeddings.weight.size
        if not self.config.tie_embeddings:
            n -= self.lm_head.weight.size
        return n

    def matmul_macs_per_token(self) -> dict:
        """Per-token matmul MACs by component, counted from the real module structure."""
        c = self.config
        T = c.block_size
        comp = dict(embed_gather=0, attn_proj=0, attn_context=0, ffn=0, norms=0, readout=0)
        for blk in self.layers:
            a = blk.attention
            comp["attn_proj"] += int(a.wq.weight.size + a.wk.weight.size
                                     + a.wv.weight.size + a.wo.weight.size)
            comp["attn_context"] += int(2 * T * a.n_head * a.head_dim)   # QK^T + AV (non-causal count)
            f = blk.feed_forward
            comp["ffn"] += int(f.w1.weight.size + f.w2.weight.size + f.w3.weight.size)
        readout_w = self.tok_embeddings.weight if c.tie_embeddings else self.lm_head.weight
        comp["readout"] = int(readout_w.shape[0] * readout_w.shape[1])
        return comp

    def forward_flops_per_token(self) -> int:
        """Forward-pass matmul FLOPs per token. A training step is ~3x this."""
        return 2 * sum(self.matmul_macs_per_token().values())
