"""Context-parallel correctness for the marginal-predictive path, runnable with no GPU.

Forces several CPU devices and pins the invariant that makes multi-GPU safe on an array
already in flight: splitting the fixed context set across N devices returns EXACTLY what one
device returns. That holds structurally -- no reduction crosses the sharded axis, a
context's chip key is a function of its GLOBAL index, and the vectors are accumulated
host-side in a fixed order.

The one exception: at a per-device batch of 1, XLA compiles a matrix-vector product instead
of a matmul and reassociates by ~1 float32 ulp. Production never reaches it, but it is
pinned so a change making it worse fails. The CPU-device count is fixed at first jax import,
so run this module STANDALONE; in a mixed suite the multi-device tests skip.
"""
import os
# Must be set BEFORE jax initializes its backends.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest

import numpy as np
import jax

from nano_llama.llama import LlamaConfig, Llama
from nano_llama import fault_eval as fe

N_DEV = 8


def _tiny(seed=0):
    cfg = LlamaConfig(block_size=16, vocab_size=32, n_layer=2, n_head=1, n_embd=8,
                      multiple_of=4, dtype="float32", attn_impl="manual", tie_embeddings=True)
    return Llama(cfg, jax.random.PRNGKey(seed)), cfg


def _contexts(cfg, C, seed=3):
    return jax.random.randint(jax.random.PRNGKey(seed), (C, cfg.block_size), 0, cfg.vocab_size)


def _collect(model, x, mesh, micro_batch, *, k=4, p=0.08, n_chips=12, n_groups=4, n_raw=3):
    """One full marginal collection on `mesh`, with the contexts placed as the worker places them."""
    block = fe.make_final_prob_block(mesh)
    xs = jax.device_put(x, fe.context_sharding(mesh)) if mesh.devices.size > 1 else x
    return fe.collect_marginal_predictive(block, model, xs, k, p, n_chips=n_chips, n_groups=n_groups,
                                          n_raw=n_raw, raw_context=1, micro_batch=micro_batch,
                                          base_key=jax.random.PRNGKey(11))


class TestContextParallelMarginals(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.n_dev = jax.local_device_count()
        if cls.n_dev < 4:
            raise unittest.SkipTest(f"needs >=4 devices, jax sees {cls.n_dev} "
                                    f"(run this module standalone -- see the docstring)")

    # ---- the headline invariant -------------------------------------------------------------
    def test_marginals_are_exact_across_device_counts(self):
        """1 vs 2 vs 4 devices (per-device batch 8/4/2) -> EXACTLY equal groups, p_bar and raw."""
        model, cfg = _tiny(); x = _contexts(cfg, C=8)
        devs = jax.local_devices()
        ref = _collect(model, x, fe.context_mesh(8, devs[:1]), micro_batch=8)
        for n in (2, 4):
            got = _collect(model, x, fe.context_mesh(8, devs[:n]), micro_batch=8)
            np.testing.assert_array_equal(got.group_means, ref.group_means,
                                          err_msg=f"group means differ on {n} devices")
            np.testing.assert_array_equal(got.p_bar(), ref.p_bar())
            np.testing.assert_array_equal(got.raw, ref.raw)

    def test_exact_across_device_count_AND_micro_batch_together(self):
        """The two memory knobs are independent: any (n_dev."""
        model, cfg = _tiny(); x = _contexts(cfg, C=12)
        devs = jax.local_devices()
        ref = _collect(model, x, fe.context_mesh(12, devs[:1]), micro_batch=12)
        for n, mb in ((2, 6), (2, 12), (4, 12), (6, 12), (2, 4), (3, 12)):
            self.assertGreaterEqual(mb // n, 2, "this case is meant to be non-degenerate")
            got = _collect(model, x, fe.context_mesh(12, devs[:n]), micro_batch=mb)
            np.testing.assert_array_equal(got.p_bar(), ref.p_bar(),
                                          err_msg=f"differs at n_dev={n}, micro_batch={mb}")

    def test_degenerate_per_device_batch_of_one_stays_within_one_ulp(self):
        """The documented exception, pinned: a per-device batch of 1 compiles to a matrix-vector
        product and reassociates. It must stay at ulp scale, so a real regression -- a mis-sharded
        key, a dropped context -- cannot hide behind it.
        """
        model, cfg = _tiny(); x = _contexts(cfg, C=8)
        devs = jax.local_devices()
        ref = _collect(model, x, fe.context_mesh(8, devs[:1]), micro_batch=8)
        for n, mb in ((8, 8), (1, 1)):                     # 1 context per device / per forward
            got = _collect(model, x, fe.context_mesh(8, devs[:n]), micro_batch=mb)
            diff = np.abs(got.p_bar() - ref.p_bar()).max()
            self.assertLess(diff, 1e-6, f"n_dev={n}, micro_batch={mb} drifted by {diff:.3g}")
            # ...and still a valid distribution, i.e. the drift is reassociation, not a lost context.
            np.testing.assert_allclose(got.p_bar().sum(axis=-1), 1.0, atol=1e-5)

    def test_production_shape_never_hits_the_degenerate_case(self):
        """C=256 over any allocation this code can get leaves >= 2 contexts per device."""
        cfg = LlamaConfig(block_size=1024, vocab_size=8192, n_layer=13, n_head=16, n_embd=512,
                          multiple_of=32, dtype="float32", attn_impl="manual")
        dev = jax.local_devices()[0]
        for n in (1, 2, 4, 8):
            mesh = fe.context_mesh(256, jax.local_devices()[:n])
            self.assertEqual(mesh.devices.size, n)
            mb = fe.choose_marginal_micro_batch(cfg, 256, 49_000_000, dev, n_dev=n)
            self.assertGreaterEqual(mb // n, 2, f"per-device batch {mb // n} at n_dev={n}")

    def test_clean_logits_are_bit_identical_across_device_counts(self):
        """The fault-free reference z0 -- the baseline every KL is measured against."""
        model, cfg = _tiny(); x = _contexts(cfg, C=8)
        devs = jax.local_devices()
        ref = np.asarray(fe.make_clean_logit_block(fe.context_mesh(8, devs[:1]))(model, x))
        for n in (2, 4, 8):
            mesh = fe.context_mesh(8, devs[:n])
            xs = jax.device_put(x, fe.context_sharding(mesh))
            got = np.asarray(fe.make_clean_logit_block(mesh)(model, xs))
            np.testing.assert_array_equal(got, ref, err_msg=f"clean logits differ on {n} devices")

    def test_single_device_path_is_unchanged_by_a_1_device_mesh(self):
        """Passing a bare Device (every pre-existing caller) and a 1-device Mesh are the same path."""
        model, cfg = _tiny(); x = _contexts(cfg, C=6)
        dev = jax.local_devices()[0]
        a = np.asarray(fe.make_clean_logit_block(dev)(model, x))
        b = np.asarray(fe.make_clean_logit_block(fe.context_mesh(6, [dev]))(model, x))
        np.testing.assert_array_equal(a, b)

    # ---- mesh + batch geometry --------------------------------------------------------------
    def test_context_mesh_trims_to_a_divisor_of_n_contexts(self):
        devs = jax.local_devices()
        self.assertEqual(fe.context_mesh(8, devs[:8]).devices.size, 8)     # exact fit
        self.assertEqual(fe.context_mesh(6, devs[:8]).devices.size, 6)     # 6 | 6
        self.assertEqual(fe.context_mesh(10, devs[:8]).devices.size, 5)    # 8 -> largest divisor 5
        self.assertEqual(fe.context_mesh(7, devs[:4]).devices.size, 1)     # prime -> 1 device
        self.assertEqual(fe.context_mesh(256, devs[:8]).devices.size, 8)   # the production case

    def test_chosen_micro_batch_tiles_the_contexts_and_shards_evenly(self):
        cfg = LlamaConfig(block_size=64, vocab_size=256, n_layer=2, n_head=2, n_embd=32,
                          multiple_of=8, dtype="float32", attn_impl="manual")
        dev = jax.local_devices()[0]
        for C in (64, 256):
            for n in (1, 2, 4, 8):
                mb = fe.choose_marginal_micro_batch(cfg, C, 5_000_000, dev, n_dev=n)
                self.assertEqual(C % mb, 0, f"micro_batch {mb} must tile C={C} (n_dev={n})")
                self.assertEqual(mb % n, 0, f"micro_batch {mb} must split over {n} devices")
                self.assertLessEqual(mb, C)

    def test_micro_batch_grows_with_the_device_count(self):
        """Each device holds only its own shard of the (mb, T, V) readout."""
        cfg = LlamaConfig(block_size=256, vocab_size=8192, n_layer=13, n_head=16, n_embd=512,
                          multiple_of=32, dtype="float32", attn_impl="manual")
        dev = jax.local_devices()[0]
        one = fe.choose_marginal_micro_batch(cfg, 256, 49_000_000, dev, n_dev=1)
        four = fe.choose_marginal_micro_batch(cfg, 256, 49_000_000, dev, n_dev=4)
        self.assertGreaterEqual(four, one)


if __name__ == "__main__":
    unittest.main(verbosity=2)
