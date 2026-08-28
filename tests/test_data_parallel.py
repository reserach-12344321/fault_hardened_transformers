"""Data-parallel (GSPMD) correctness test, runnable with NO GPU.

Forces several *CPU* devices (via XLA_FLAGS, set before jax is imported) and checks the core DP
invariant: with the global batch held fixed, replicating the model/optimizer across N devices and
sharding the batch produces the SAME result as a single device -- up to floating-point reduction
order. This exercises the exact compiled blocks used in training (_make_train_block /
_make_eval_block), so it catches sharding/all-reduce mistakes without touching the shared GPU.

Because the CPU-device count is fixed at first jax import, this module must be the FIRST thing to
import jax to see all N_DEV devices. Run it STANDALONE:
    JAX_PLATFORMS=cpu PYTHONPATH=<repo> /home/trevor/scienv/bin/python -m unittest tests.test_data_parallel -v
In a mixed suite where jax was already initialized with one device, the tests skip (not fail).
"""
import os
# Must be set BEFORE jax initializes its backends.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from nano_llama.llama import LlamaConfig, Llama, FaultSpec
from nano_llama.train_core import TrainConfig, build_optimizer, _make_train_block, _make_eval_block

N_DEV = 8            # forced CPU devices
B = 16               # global batch (per optimizer step); divisible by 1 and 8
T = 16               # block size
N_STEPS = 3          # scanned optimizer steps in one train block
ACCUM = 2            # gradient-accumulation micro-steps -> micro = B / ACCUM = 8 (divisible by 8)


def _build():
    """A tiny model + optimizer + fresh opt_state (identical every call)."""
    cfg = LlamaConfig(block_size=T, vocab_size=64, n_layer=2, n_head=2, n_embd=32, multiple_of=8)
    model = Llama(cfg, jax.random.PRNGKey(0))
    optim = build_optimizer(TrainConfig(learning_rate=1e-2, warmup_iters=1, lr_decay_iters=100,
                                        weight_decay=0.0, grad_clip=1.0), model)
    opt_state = optim.init(eqx.partition(model, eqx.is_array)[0])
    return cfg, model, optim, opt_state


def _synthetic(cfg, batch_axis):
    """Deterministic synthetic (xs, ys), int32 (n_steps, batch_axis, T). Same values every call."""
    kx, ky = jax.random.split(jax.random.PRNGKey(123), 2)
    xs = jax.random.randint(kx, (N_STEPS, batch_axis, T), 0, cfg.vocab_size, dtype=jnp.int32)
    ys = jax.random.randint(ky, (N_STEPS, batch_axis, T), 0, cfg.vocab_size, dtype=jnp.int32)
    return xs, ys


def _mesh(n_dev):
    return Mesh(np.array(jax.devices()[:n_dev]), ("data",))


def _run_train(n_dev, p):
    """Run one train block on n_dev devices; return (updated model, losses) as host arrays."""
    mesh = _mesh(n_dev)
    cfg, model, optim, opt_state = _build()
    block = _make_train_block(optim, ACCUM, mesh)
    xs, ys = _synthetic(cfg, B)

    repl = NamedSharding(mesh, P())
    shard = NamedSharding(mesh, P(None, "data", None))
    model = eqx.filter_shard(model, repl)
    opt_state = eqx.filter_shard(opt_state, repl)
    key = eqx.filter_shard(jax.random.PRNGKey(7), repl)
    xs, ys = jax.device_put(xs, shard), jax.device_put(ys, shard)
    if n_dev > 1:                          # guard against a degenerate (unsharded) test
        assert len(xs.addressable_shards) == n_dev, "batch is not actually sharded across devices"
    spec = FaultSpec(p=p, k=4)

    model, _, losses = block(model, opt_state, xs, ys, key, 0, spec)   # base_key, start_step=0
    return model, np.asarray(losses)


def _run_train_blocks(n_dev, p, donate, n_blocks=3):
    """Thread state through `n_blocks` train blocks the way the real loop does."""
    mesh = _mesh(n_dev)
    cfg, model, optim, opt_state = _build()
    block = _make_train_block(optim, ACCUM, mesh, donate=donate)
    repl = NamedSharding(mesh, P())
    shard = NamedSharding(mesh, P(None, "data", None))
    model = eqx.filter_shard(model, repl)
    opt_state = eqx.filter_shard(opt_state, repl)
    base_key = eqx.filter_shard(jax.random.PRNGKey(7), repl)     # REUSED every block (must not be donated)
    spec = FaultSpec(p=p, k=4)
    losses_all, dk = [], jax.random.PRNGKey(999)
    for b in range(n_blocks):
        kx, ky, dk = jax.random.split(dk, 3)
        xs = jax.random.randint(kx, (N_STEPS, B, T), 0, cfg.vocab_size, dtype=jnp.int32)
        ys = jax.random.randint(ky, (N_STEPS, B, T), 0, cfg.vocab_size, dtype=jnp.int32)
        xs, ys = jax.device_put(xs, shard), jax.device_put(ys, shard)
        model, opt_state, losses = block(model, opt_state, xs, ys, base_key, b * N_STEPS, spec)
        losses_all.append(np.asarray(losses))
    return model, np.concatenate(losses_all)


N_EVAL = 32          # fixed eval probe set (sequences); divisible by every chunk size tested


def _run_eval(n_dev, p, eval_mb=None):
    """Score a FIXED probe set of N_EVAL sequences on n_dev devices."""
    mesh = _mesh(n_dev)
    cfg, model, optim, opt_state = _build()
    block = _make_eval_block(mesh)
    mb = eval_mb if eval_mb is not None else B // ACCUM
    assert N_EVAL % mb == 0 and mb % n_dev == 0
    n_chunk = N_EVAL // mb

    kx, ky = jax.random.split(jax.random.PRNGKey(123), 2)
    xs = jax.random.randint(kx, (N_EVAL, T), 0, cfg.vocab_size, dtype=jnp.int32).reshape(n_chunk, mb, T)
    ys = jax.random.randint(ky, (N_EVAL, T), 0, cfg.vocab_size, dtype=jnp.int32).reshape(n_chunk, mb, T)
    # One key per sequence, split over the WHOLE probe set then reshaped -- sequence i meets chip i
    # regardless of how the set is chunked.
    keys = jax.random.split(jax.random.PRNGKey(7), N_EVAL).reshape(n_chunk, mb, -1)

    repl = NamedSharding(mesh, P())
    shard = NamedSharding(mesh, P(None, "data", None))
    model = eqx.filter_shard(model, repl)
    xs, ys = jax.device_put(xs, shard), jax.device_put(ys, shard)
    keys = jax.device_put(keys, shard)
    if n_dev > 1:
        assert len(xs.addressable_shards) == n_dev, "batch is not actually sharded across devices"
    spec = FaultSpec(p=p, k=4)

    losses = np.asarray(block(model, xs, ys, spec, keys))
    assert losses.size == N_EVAL, "eval must return one loss per sequence"
    return float(losses.mean())


def _arrays(model):
    return [np.asarray(a) for a in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))]


@unittest.skipUnless(jax.device_count() >= N_DEV,
                     f"needs >= {N_DEV} devices (run this module standalone so it forces the CPU "
                     f"device count before jax initializes); got {jax.device_count()}")
class DataParallelTest(unittest.TestCase):
    def _assert_models_close(self, m_ref, m_dp, tag, rtol=2e-3, atol=2e-4):
        ref, dp = _arrays(m_ref), _arrays(m_dp)
        self.assertEqual(len(ref), len(dp), "pytree structure differs")
        for a, b in zip(ref, dp):
            self.assertTrue(np.allclose(a, b, rtol=rtol, atol=atol),
                            f"[{tag}] leaf mismatch: max|Δ|={np.max(np.abs(a - b)):.3e}")

    def test_train_block_clean_equivalence(self):
        """CLEAN (p=0): DP over 8 devices reproduces the single-device optimizer update + losses."""
        m1, l1 = _run_train(1, p=0.0)
        m8, l8 = _run_train(N_DEV, p=0.0)
        self._assert_models_close(m1, m8, "train clean")
        self.assertTrue(np.allclose(l1, l8, rtol=2e-3, atol=2e-4), f"clean losses differ: {l1} vs {l8}")

    def test_eval_block_clean_equivalence(self):
        """At p=0 the eval mean matches between 1 and 8 devices."""
        c1, c8 = _run_eval(1, p=0.0), _run_eval(N_DEV, p=0.0)
        self.assertTrue(np.allclose(c1, c8, rtol=2e-3, atol=2e-4), f"p=0 eval differs: {c1} vs {c8}")

    def test_eval_is_chunk_size_invariant(self):
        """The eval CHUNK is a per-machine memory choice."""
        for p in (0.0, 0.05):
            ref = _run_eval(1, p=p, eval_mb=N_EVAL)                   # one chunk: the whole set at once
            for mb in (4, 8, 16):
                v = _run_eval(1, p=p, eval_mb=mb)
                self.assertTrue(np.allclose(v, ref, rtol=2e-3, atol=2e-4),
                                f"eval moved with chunk {mb} at p={p}: {v} vs {ref}")
        self.assertGreater(abs(_run_eval(1, p=0.05, eval_mb=N_EVAL) - _run_eval(1, p=0.0, eval_mb=N_EVAL)),
                           1e-4, "p=0.05 and p=0 agree -- the faulted path may be inert")

    def test_faulted_dp_is_device_count_invariant(self):
        """FAULTED (p>0): because per-sequence fault keys are now assigned by GLOBAL batch."""
        m1, l1 = _run_train(1, p=0.05)
        m8, l8 = _run_train(N_DEV, p=0.05)
        self._assert_models_close(m1, m8, "train faulted")
        self.assertTrue(np.allclose(l1, l8, rtol=2e-3, atol=2e-4), f"faulted losses differ: {l1} vs {l8}")

        f1, f8 = _run_eval(1, p=0.05), _run_eval(N_DEV, p=0.05)
        self.assertTrue(np.allclose(f1, f8, rtol=2e-3, atol=2e-4), f"faulted eval differs: {f1} vs {f8}")
        self.assertGreater(abs(f8 - _run_eval(N_DEV, p=0.0)), 1e-4,
                           "faulted eval matches p=0 under DP -- the faulted path may be inert")

    def test_donation_is_bit_identical(self):
        """Buffer donation in _make_train_block is a pure memory optimization (it lets XLA."""
        for n_dev in (1, N_DEV):
            for p in (0.0, 0.05):
                m_off, l_off = _run_train_blocks(n_dev, p, donate="none")
                m_on, l_on = _run_train_blocks(n_dev, p, donate="all-except-first")
                a_off, a_on = _arrays(m_off), _arrays(m_on)
                self.assertEqual(len(a_off), len(a_on), "pytree structure differs")
                for x, y in zip(a_off, a_on):
                    self.assertTrue(np.array_equal(x, y),
                                    f"[n_dev={n_dev} p={p}] donation changed a param: "
                                    f"max|Δ|={np.max(np.abs(x - y)):.3e}")
                self.assertTrue(np.array_equal(l_off, l_on),
                                f"[n_dev={n_dev} p={p}] donation changed losses:\n off={l_off}\n on ={l_on}")


if __name__ == "__main__":
    unittest.main()
