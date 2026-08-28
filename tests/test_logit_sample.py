"""Unit tests for the marginal-predictive path in nano_llama/fault_eval.py.

CPU, tiny synthetic Llama, no bins/checkpoints/GPU. Covers the properties the marginal collector is
relied on for: p_bar is softmax-THEN-average (a valid distribution), the group means average to the
full marginal, determinism is keyed on (chip, global context) and independent of micro-batching, p=0
is the deterministic clean predictive, raw samples are individual chips of the chosen context, and the
config round-trips.

Runnable under pytest or directly (``python tests/test_logit_sample.py``).
"""
import unittest

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from nano_llama.llama import LlamaConfig, Llama
from nano_llama import fault_eval as fe


def _tiny(seed=0):
    cfg = LlamaConfig(block_size=16, vocab_size=32, n_layer=1, n_head=1, n_embd=8,
                      multiple_of=4, dtype="float32", attn_impl="manual", tie_embeddings=True)
    return Llama(cfg, jax.random.PRNGKey(seed)), cfg


def _contexts(cfg, C=6, seed=3):
    return jax.random.randint(jax.random.PRNGKey(seed), (C, cfg.block_size), 0, cfg.vocab_size)


def _blocks():
    dev = jax.local_devices()[0]
    return fe.make_final_prob_block(dev), fe.make_clean_logit_block(dev)


class LogitSampleTest(unittest.TestCase):
    def test_pbar_is_a_distribution_and_softmax_then_average(self):
        """Every group mean is a normalised distribution, and p_bar equals the mean of the groups."""
        model, cfg = _tiny(); x = _contexts(cfg)
        block, _ = _blocks()
        res = fe.collect_marginal_predictive(block, model, x, 4, 0.1, n_chips=40, n_groups=8, n_raw=5,
                                             raw_context=0, micro_batch=3,
                                             base_key=jax.random.PRNGKey(0))
        assert res.group_means.shape == (8, x.shape[0], cfg.vocab_size)
        assert np.allclose(res.group_means.sum(axis=-1), 1.0, atol=1e-5)      # each a distribution
        assert np.allclose(res.p_bar(), res.group_means.mean(axis=0), atol=1e-6)
        assert np.allclose(res.p_bar().sum(axis=-1), 1.0, atol=1e-5)


    def test_groups_are_equal_sized_and_partition_the_chips(self):
        """n_chips split across n_groups by chip index -> equal counts when divisible."""
        model, cfg = _tiny(); x = _contexts(cfg)
        block, _ = _blocks()
        # 40 chips / 8 groups = 5 each; the mean-of-group-means then equals the flat mean of all 40.
        res = fe.collect_marginal_predictive(block, model, x, 4, 0.1, n_chips=40, n_groups=8, n_raw=0,
                                             raw_context=0, micro_batch=6,
                                             base_key=jax.random.PRNGKey(1))
        # recompute the flat marginal directly
        base = jax.random.PRNGKey(1)
        C = x.shape[0]; acc = np.zeros((C, cfg.vocab_size))
        for m in range(40):
            km = jax.random.fold_in(base, m)
            keys = jnp.stack([jax.random.fold_in(km, c) for c in range(C)])
            acc += np.asarray(block(model, x, fe.FaultSpec(0.1, 4), keys))
        assert np.allclose(res.p_bar(), acc / 40, atol=1e-5)


    def test_determinism_is_independent_of_micro_batch(self):
        """Same (chips, contexts) -> identical p_bar regardless of the forward micro-batch."""
        model, cfg = _tiny(); x = _contexts(cfg, C=6)
        block, _ = _blocks()
        kw = dict(n_chips=16, n_groups=4, n_raw=0, raw_context=0)
        a = fe.collect_marginal_predictive(block, model, x, 4, 0.05, micro_batch=2,
                                           base_key=jax.random.PRNGKey(7), **kw)
        b = fe.collect_marginal_predictive(block, model, x, 4, 0.05, micro_batch=5,
                                           base_key=jax.random.PRNGKey(7), **kw)
        assert np.allclose(a.p_bar(), b.p_bar(), atol=1e-6)


    def test_raw_samples_are_individual_chips_of_the_chosen_context(self):
        """raw[m] is the m-th chip's predictive for raw_context, not an average."""
        model, cfg = _tiny(); x = _contexts(cfg, C=5)
        block, _ = _blocks()
        RC = 2
        res = fe.collect_marginal_predictive(block, model, x, 4, 0.1, n_chips=20, n_groups=4, n_raw=6,
                                             raw_context=RC, micro_batch=5,
                                             base_key=jax.random.PRNGKey(9))
        assert res.raw.shape == (6, cfg.vocab_size)
        assert np.allclose(res.raw.sum(axis=-1), 1.0, atol=1e-5)              # each a distribution
        # chip 0 of context RC, recomputed
        km = jax.random.fold_in(jax.random.PRNGKey(9), 0)
        keys = jnp.stack([jax.random.fold_in(km, c) for c in range(x.shape[0])])
        expect = np.asarray(block(model, x, fe.FaultSpec(0.1, 4), keys))[RC]
        assert np.allclose(res.raw[0], expect, atol=1e-5)
        # individual chips differ from the marginal (fault actually varies things)
        assert not np.allclose(res.raw[0], res.p_bar()[RC], atol=1e-3)


    def test_clean_block_matches_p0_forward(self):
        """The clean logit block equals a p=0 (fault-off) forward; softmax of it is the p=0 marginal."""
        model, cfg = _tiny(); x = _contexts(cfg, C=4)
        prob_block, clean_block = _blocks()
        z0 = np.asarray(clean_block(model, x))
        assert z0.shape == (4, cfg.vocab_size)
        # p=0 marginal (fault off) via the prob block, any key -> softmax(clean logits)
        keys = jnp.stack([jax.random.fold_in(jax.random.PRNGKey(0), c) for c in range(4)])
        p0 = np.asarray(prob_block(model, x, fe.FaultSpec(0.0, 4), keys))
        sm = np.exp(z0 - z0.max(1, keepdims=True)); sm /= sm.sum(1, keepdims=True)
        assert np.allclose(p0, sm, atol=1e-5)


    def test_config_roundtrip(self, tmp_path=None):
        import tempfile
        d = str(tmp_path) if tmp_path is not None else tempfile.mkdtemp()
        cfg = fe.LogitSampleConfig(kp_pairs=[(4, 0.01), (4, 0.1)], n_contexts=128, n_chips=500,
                                   n_groups=8, n_raw=100, raw_context=3, context_seed=1, context_stream=2,
                                   seed=5)
        path = os.path.join(d, "logit_sample_config.json")
        cfg.save(path)
        back = fe.LogitSampleConfig.load(path)
        assert back.kp_pairs == ((4, 0.01), (4, 0.1))
        assert (back.n_contexts, back.n_chips, back.n_groups, back.n_raw) == (128, 500, 8, 100)
        assert back.raw_context == 3 and back.context_seed == 1 and back.seed == 5


if __name__ == "__main__":
    unittest.main()
