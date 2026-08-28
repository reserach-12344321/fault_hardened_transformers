r"""Post-hoc faulted evaluation: score a checkpoint at a list of fault (k, p) conditions.

p is traced and k static, so the eval block compiles once per distinct k and sweeps visit
the pairs k-major. Each sequence gets its own i.i.d. chip, so estimate_point streams batches
until their standard error reaches a target ABSOLUTE precision in nats -- absolute, so every
model on the ladder is measured to the same accuracy. Each point draws its own loader
`stream`, so points are independent and evaluation order and preemption are invisible.

Three paths sharing the fault semantics and the determinism contract: estimate_point (the
mean, to a target precision), sample_point (the loss DISTRIBUTION, at a fixed budget, since
the adaptive rule selects against the tail it would measure) and collect_marginal_predictive
(the fault-averaged next-token distribution over a fixed context set).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from nano_llama.config_base import ConfigMixin
from nano_llama.llama import Llama, LlamaConfig, FaultSpec
from nano_llama.token_data import SlidingLoader


DEFAULT_TARGET_SE = 0.005        # stop a point once se <= 0.005 NATS (absolute, not relative)
DEFAULT_MIN_EVALS = 1024         # ...but never before this many sequences (optional-stopping guard)
DEFAULT_MAX_EVALS = 65536        # ...and give up after this many (records reached_target=False)


@dataclass(frozen=True)
class EvalConfig(ConfigMixin):
    """Spec for a post-hoc fault-eval sweep of one trained model.

    kp_pairs   : explicit ((k, p), ...) rather than a cross product, so a dense p-grid on
                 one k and a sparse one on another are equally expressible. k-major.
    target_se  : stop a point once se <= this, in NATS. Absolute, so every model on the
                 ladder is measured to the same accuracy.
    min_evals  : sequences drawn before the criterion is checked (optional-stopping guard).
    max_evals  : hard cap; hitting it records reached_target=False.
    batch_size : None chooses adaptively. Also the stopping granularity.
    seed       : seeds both the eval data sampling and the fault draws.
    """
    kp_pairs: tuple = ()
    target_se: float = DEFAULT_TARGET_SE
    min_evals: int = DEFAULT_MIN_EVALS
    max_evals: int = DEFAULT_MAX_EVALS
    batch_size: Optional[int] = None
    seed: int = 0

    def __post_init__(self):
        object.__setattr__(self, "kp_pairs",
                           tuple((int(k), float(p)) for k, p in self.kp_pairs))

# ======================================================================================
# Per-sequence faulted eval block  (compiled once; p is traced so the p-grid reuses it)
# ======================================================================================
def make_per_seq_eval_block(device: jax.Device) -> Callable:
    """A jitted (model, x, y, spec, key) -> per-sequence mean CE under a fault spec."""
    def _eval(model: Llama, x: jax.Array, y: jax.Array, spec: FaultSpec, key: jax.Array):
        keys = jax.random.split(key, x.shape[0])

        def one(x_i, y_i, k_i):
            logits = model(x_i, spec=spec, key=k_i)                     # (block_size, vocab)
            ce = optax.softmax_cross_entropy_with_integer_labels(logits, y_i)   # (block_size,)
            return jnp.mean(ce)                                          # scalar: this sequence's CE

        return jax.vmap(one)(x, y, keys)                                # (batch,)

    return eqx.filter_jit(_eval, device=device)


# ======================================================================================
# Forward-only eval batch size  (adaptive to the GPU we land on -- like training's chooser)
# ======================================================================================
def choose_eval_micro_batch(model_config: LlamaConfig, n_params: int, device: jax.Device, *,
                            max_evals: int, mem_fraction: float = 0.9, reserve_gb: float = 2.0,
                            act_safety: float = 1.5, max_micro_batch: int = 256) -> int:
    """Largest forward-only eval batch estimated to fit on `device`."""
    try:
        total = int(device.memory_stats()["bytes_limit"])
    except Exception:
        total = 16 * 1024**3
    c = model_config
    T, V, d, m = c.block_size, c.vocab_size, c.n_embd, c.multiple_of
    ffn_hidden = m * ((int(2 * (4 * d) / 3) + m - 1) // m)          # SwiGLU hidden (matches the model)
    persistent = 4 * n_params                                       # fp32 weights only (no opt/grads)
    attn_scores = 12 * c.n_head * T * T if c.attn_impl == "manual" else 0   # materialized softmax
    # Forward-only bytes/seq: logits region (as in training) + ONE layer's attn/FFN working set (no xL).
    per_seq = act_safety * (16 * T * V + 24 * T * d + 6 * T * ffn_hidden + attn_scores)
    budget = total * mem_fraction - persistent - reserve_gb * 1024**3
    mb = int(max(1, budget // per_seq))
    return max(1, min(mb, int(max_evals), max_micro_batch))


# ======================================================================================
# Adaptive (target-precision) sample-mean + standard error, per (k, p)
# ======================================================================================
@dataclass
class PSweepPoint:
    """The estimate at one (k, p): mean faulted CE in nats, its standard error."""
    k: int
    p: float
    mean: float
    se: float
    n_seq: int
    reached_target: bool = True

    @property
    def rel_se(self) -> float:
        """Standard error as a fraction of the mean."""
        return self.se / self.mean if self.mean else float("inf")


def mean_se(s: float, s2: float, n: int) -> tuple:
    """Sample mean and standard error of the mean from streaming sums."""
    mean = s / n
    var = max(s2 - s * s / n, 0.0) / max(1, n - 1)
    return mean, math.sqrt(var / n)


def estimate_point(block: Callable, model: Llama, loader: SlidingLoader,
                   k: int, p: float, *, stream: int, batch_size: int, key: jax.Array,
                   target_se: float = DEFAULT_TARGET_SE,
                   min_evals: int = DEFAULT_MIN_EVALS,
                   max_evals: int = DEFAULT_MAX_EVALS) -> PSweepPoint:
    """Estimate the faulted loss at one (k, p) to a target absolute precision in nats.

    Streams batches from `loader` under independent fault draws and stops once
    se <= target_se. The criterion is not tested below `min_evals`, so a noisy early
    variance estimate cannot trigger a biased-low stop; past `max_evals` it gives up and
    returns reached_target=False.

    Data is keyed on (seed, stream, step), so a point's estimate depends on its `stream`
    and how far it has sampled -- not on evaluation order or interruptions. Points sharing
    a loader need distinct streams, or they score identical sequences.

    Batches are always full-size, since `block` is jitted on the shape, so n_seq can
    overshoot max_evals by up to batch_size - 1.
    """
    spec = FaultSpec(p=float(p), k=int(k))
    s = s2 = 0.0
    n = 0
    step = 0
    reached = False
    while n < max_evals:
        x, y = loader.batch(batch_size, step=step, stream=stream)
        step += 1
        key, ek = jax.random.split(key)
        losses = np.asarray(block(model, x, y, spec, ek), dtype=np.float64)
        s += float(losses.sum())
        s2 += float((losses * losses).sum())
        n += losses.size
        if n >= min_evals:
            _, se = mean_se(s, s2, n)
            if se <= target_se:
                reached = True
                break

    mean, se = mean_se(s, s2, n)
    return PSweepPoint(k=int(k), p=float(p), mean=mean, se=se, n_seq=n, reached_target=reached)


def k_major_order(kp_pairs: Sequence[tuple]) -> list:
    """Indices into ``kp_pairs`` ordered so all pairs sharing a ``k`` are evaluated."""
    return sorted(range(len(kp_pairs)), key=lambda i: kp_pairs[i][0])


# MARGINAL PREDICTIVE sampling: fixed contexts scored under many chips, keeping the full
# probability vector rather than a loss, because the question is the SHAPE of the predictive.
# SOFTMAX-THEN-AVERAGE -- a constant logit offset cancels in the softmax, so only the
# probability average integrates the chip out correctly. Stored: K group-means of p_bar
# (their split gives the noise floor), n_raw single-chip vectors for ONE context (a
# group-mean is already Jensen-flattened), and the clean logits z0. The fault key is folded
# on (chip index, GLOBAL context index), so batching cannot move which chip a context sees.
DEFAULT_N_CHIPS = 1000           # chips marginalised per context to form p_bar
DEFAULT_N_GROUPS = 8             # running sub-averages (noise floor / bootstrap over chips)
DEFAULT_N_RAW = 100              # single-chip distributions kept for ONE context
DEFAULT_CONTEXT_STREAM = 777     # loader stream that defines the shared, fixed context set


@dataclass(frozen=True)
class LogitSampleConfig(ConfigMixin):
    """Spec for a marginal-predictive sample of one trained model.

    n_contexts x n_chips forwards per fault level, so those two set the bill. n_groups is
    the number of running sub-averages of p_bar (>=2, so their split gives a noise floor);
    n_raw single-chip vectors are kept for `raw_context`.

    context_seed / context_stream define the SHARED context set and must be identical
    across models, or the fitted temperature is not comparable between arms and sizes.
    micro_batch is a memory knob only -- it does not change which chips a context sees.
    """
    kp_pairs: tuple = ()
    n_contexts: int = 256
    n_chips: int = DEFAULT_N_CHIPS
    n_groups: int = DEFAULT_N_GROUPS
    n_raw: int = DEFAULT_N_RAW
    raw_context: int = 0
    context_seed: int = 0
    context_stream: int = DEFAULT_CONTEXT_STREAM
    micro_batch: Optional[int] = None
    seed: int = 0

    def __post_init__(self):
        object.__setattr__(self, "kp_pairs",
                           tuple((int(k), float(p)) for k, p in self.kp_pairs))


# CONTEXT PARALLELISM: a context's chip key is a function of its GLOBAL index alone, so
# contexts split across devices with no communication and no key moves when they do -- the
# marginal is exactly equal across device counts whenever each device holds >= 2 contexts per
# forward. The mesh is 1-D and trimmed to a DIVISOR of n_contexts; shard_map rather than the
# auto-partitioner, so the placement is stated rather than inferred.
CTX_AXIS = "ctx"


def _largest_divisor_leq(n: int, cap: int) -> int:
    """Largest divisor of n that is <= cap (>= 1)."""
    for d in range(min(cap, n), 0, -1):
        if n % d == 0:
            return d
    return 1


def context_mesh(n_contexts: int, devices: Optional[Sequence] = None) -> Mesh:
    """A 1-D mesh over the context axis."""
    devs = list(devices if devices is not None else jax.local_devices())
    n_use = max(1, _largest_divisor_leq(int(n_contexts), len(devs)))
    return Mesh(np.asarray(devs[:n_use]), (CTX_AXIS,))


def context_sharding(mesh: Mesh) -> NamedSharding:
    """Sharding for a (C, ...) array laid out along the context axis of `mesh`."""
    return NamedSharding(mesh, P(CTX_AXIS))


def _resolve_mesh(target) -> Mesh:
    """Accept a Mesh, a single jax.Device, or a sequence of devices, and return a Mesh."""
    if isinstance(target, Mesh):
        return target
    if isinstance(target, (list, tuple)):
        return Mesh(np.asarray(list(target)), (CTX_AXIS,))
    return Mesh(np.asarray([target]), (CTX_AXIS,))


def choose_marginal_micro_batch(model_config: LlamaConfig, n_contexts: int, n_params: int,
                                device: jax.Device, n_dev: int = 1) -> int:
    """Forward micro-batch over the contexts: how many go through the mesh at once."""
    n_dev = max(1, int(n_dev))
    cap = choose_eval_micro_batch(model_config, n_params, device, max_evals=n_contexts) * n_dev
    if int(n_contexts) % n_dev != 0:
        return n_dev
    j = _largest_divisor_leq(int(n_contexts) // n_dev, max(1, cap // n_dev))
    return max(n_dev, n_dev * j)


def make_final_prob_block(target) -> Callable:
    """Jitted ``(model, x, spec, keys) -> (B, V)``: the softmax of the FINAL-position logits."""
    mesh = _resolve_mesh(target)
    if mesh.devices.size == 1:
        device = list(mesh.devices.flat)[0]

        def _block(model: Llama, x: jax.Array, spec: FaultSpec, keys: jax.Array):
            def one(x_i, k_i):
                logits = model(x_i, spec=spec, key=k_i)      # (T, V)
                return jax.nn.softmax(logits[-1])            # (V,) final-position predictive
            return jax.vmap(one)(x, keys)                    # (B, V)
        return eqx.filter_jit(_block, device=device)

    cache = {}

    def _block(model: Llama, x: jax.Array, spec: FaultSpec, keys: jax.Array):
        # Split off the static (non-array) leaves and close over them: shard_map maps over arrays, and
        # the config's Python ints must stay Python ints. `static` is constant for a run, so the
        # mapped fn is built once and reused across every chip and every (k, p).
        arrays, static = eqx.partition(model, eqx.is_array)
        if "fn" not in cache:
            def _sharded(arrays, x, spec, keys):
                model = eqx.combine(arrays, static)

                def one(x_i, k_i):
                    return jax.nn.softmax(model(x_i, spec=spec, key=k_i)[-1])
                return jax.vmap(one)(x, keys)                # (B/n_dev, V) on this device

            cache["fn"] = eqx.filter_jit(jax.shard_map(
                _sharded, mesh=mesh,
                # model + spec replicated; contexts and their keys split on the context axis. Outputs
                # stay sharded (no reduction), so the host sees each context exactly once, in order.
                in_specs=(P(), P(CTX_AXIS, None), P(), P(CTX_AXIS, None)),
                out_specs=P(CTX_AXIS, None), check_vma=False))
        return cache["fn"](arrays, x, spec, keys)

    return _block


def make_clean_logit_block(target) -> Callable:
    """Jitted ``(model, x) -> (B, V)``: the FINAL-position logits with the fault OFF."""
    mesh = _resolve_mesh(target)
    if mesh.devices.size == 1:
        device = list(mesh.devices.flat)[0]

        def _block(model: Llama, x: jax.Array):
            def one(x_i):
                return model(x_i, spec=None, key=None)[-1]   # (V,) clean logits, final position
            return jax.vmap(one)(x)
        return eqx.filter_jit(_block, device=device)

    cache = {}

    def _block(model: Llama, x: jax.Array):
        arrays, static = eqx.partition(model, eqx.is_array)
        if "fn" not in cache:
            def _sharded(arrays, x):
                model = eqx.combine(arrays, static)
                return jax.vmap(lambda x_i: model(x_i, spec=None, key=None)[-1])(x)

            cache["fn"] = eqx.filter_jit(jax.shard_map(
                _sharded, mesh=mesh, in_specs=(P(), P(CTX_AXIS, None)),
                out_specs=P(CTX_AXIS, None), check_vma=False))
        return cache["fn"](arrays, x)

    return _block


@dataclass
class MarginalResult:
    """The marginal predictive at one ``(k, p)`` over the fixed context set."""
    k: int
    p: float
    group_means: np.ndarray
    raw: np.ndarray
    n_chips: int

    def p_bar(self) -> np.ndarray:
        """The fault-marginalised predictive p_bar, (C, V) -- the mean of the group means."""
        return self.group_means.mean(axis=0)


def _micro_batches(n: int, mb: int):
    for s in range(0, n, mb):
        yield s, min(s + mb, n)


def collect_marginal_predictive(block: Callable, model: Llama, x_contexts: jax.Array,
                                 k: int, p: float, *, n_chips: int, n_groups: int,
                                 n_raw: int, raw_context: int, micro_batch: int,
                                 base_key: jax.Array) -> MarginalResult:
    """Marginalise the fault at ONE ``(k, p)`` over the fixed contexts ``x_contexts`` (C, T)."""
    spec = FaultSpec(p=float(p), k=int(k))
    C = int(x_contexts.shape[0])
    V = None
    group_sum = None
    group_cnt = np.zeros(n_groups, dtype=np.int64)
    raw = None
    ctx_idx = jnp.arange(C)
    for m in range(n_chips):
        # per-context keys for this pass, keyed on the GLOBAL context index (not batch position)
        km = jax.random.fold_in(base_key, m)
        keys = jax.vmap(lambda c: jax.random.fold_in(km, c))(ctx_idx)     # (C, 2)
        probs = np.empty((C, V), dtype=np.float64) if V is not None else None
        parts = []
        for a, b in _micro_batches(C, micro_batch):
            pm = np.asarray(block(model, x_contexts[a:b], spec, keys[a:b]), dtype=np.float64)
            parts.append(pm)
        probs = np.concatenate(parts, axis=0)                            # (C, V)
        if V is None:
            V = probs.shape[1]
            group_sum = np.zeros((n_groups, C, V), dtype=np.float64)
            raw = np.zeros((max(n_raw, 0), V), dtype=np.float32)
        g = m % n_groups
        group_sum[g] += probs
        group_cnt[g] += 1
        if m < n_raw:
            raw[m] = probs[raw_context].astype(np.float32)
    group_means = (group_sum / group_cnt[:, None, None]).astype(np.float32)
    return MarginalResult(k=int(k), p=float(p), group_means=group_means, raw=raw, n_chips=int(n_chips))
