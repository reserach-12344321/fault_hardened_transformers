"""Saturated-fault (p=1.0) verification that the fault model really reaches the model.

At p=1.0 every block draw fires, so every faulted weight becomes exactly zero and a fault
silently not applied is a structural difference, not a statistical one -- hence every
assertion here is bitwise.

WHAT p=1.0 DOES NOT DO, before you change the assertions: it does not stop the model
training. The embedding lookup is a gather and the readout is passed spec=None, and every
faulted matmul is inside a block that enters only through a residual add -- so at p=1.0 the
residual stream is the embedding and the model degenerates to a trainable context-free
bigram predictor. Only the BLOCK parameters are cut off from the loss. The invariant is
therefore bitwise equality with the same model with every faulted weight zeroed.

The K/V operand masks are the blind spot p=1.0 cannot cover (wv is already zeroed), so
test_every_fault_site_is_invoked covers them structurally. The frozen-parameter tests set
weight_decay=0.0, since decoupled decay moves a parameter with zero gradient.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import json
import math
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from jax.sharding import PartitionSpec as P

import nano_llama.llama as llama_mod
from nano_llama.llama import FaultSpec, _fault
from tests._test_util import build_model_config
from nano_llama.initializations import build_model
from nano_llama.fault import FaultConfig
from nano_llama.train import compute_loss
from nano_llama.train_core import (TrainConfig, build_optimizer, run_training,
                                   _make_train_block, _make_eval_block)
from nano_llama.fault_eval import make_per_seq_eval_block

VOCAB, EMBD, LAYERS, BLOCK, BATCH = 512, 64, 2, 16, 4
K = 4                              # fault block size (FMAs between error checks)
P1 = FaultSpec(p=1.0, k=K)         # saturated: every block drop fires
P0 = FaultSpec(p=0.0, k=K)         # clean, but still down the faulted code path (mask is all ones)


# ======================================================================================
# Fixtures: a tiny model, and the "every faulted matrix is zero" reference model
# ======================================================================================
def _config(**kw):
    """A tiny two-block config: float32 + manual attention, so this runs on CPU and the
    comparisons are exact. The flash path carries the same masks but cuDNN is unavailable here.
    """
    return build_model_config(EMBD, LAYERS, vocab_size=VOCAB, block_size=BLOCK,
                              dtype="float32", attn_impl="manual", **kw)


def _model(seed=0, **kw):
    """Built through build_model -- the same constructor worker.py uses."""
    return build_model(_config(**kw), jax.random.PRNGKey(seed), init_std=0.02)


def _hidden(m):
    """The 7 per-block weight matrices that sit ON the faulty array, for every layer."""
    out = []
    for l in m.layers:
        a, f = l.attention, l.feed_forward
        out += [a.wq.weight, a.wk.weight, a.wv.weight, a.wo.weight,
                f.w1.weight, f.w2.weight, f.w3.weight]
    return out


def _block_norms(m):
    """The per-block RMSNorm gains: not faulted themselves."""
    return [w for l in m.layers for w in (l.attention_norm.weight, l.ffn_norm.weight)]


def _readout_and_embedding(m):
    """The parameters on the FAULT-FREE path: they still train at p=1.0 (see the module docstring)."""
    ps = [m.tok_embeddings.weight, m.norm.weight]
    if m.lm_head is not None:
        ps.append(m.lm_head.weight)
    return ps


def _zero_hidden(m):
    """The reference model: every faulted weight matrix replaced by zeros."""
    return eqx.tree_at(_hidden, m, replace=[jnp.zeros_like(w) for w in _hidden(m)])


def _perturb_hidden(m, seed=1234, scale=0.5):
    """Replace every faulted weight matrix with large random values."""
    ws = _hidden(m)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(ws))
    return eqx.tree_at(_hidden, m,
                       replace=[scale * jax.random.normal(k, w.shape, w.dtype)
                                for k, w in zip(keys, ws)])


def _tokens(seed=0, n=BATCH):
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    x = jax.random.randint(k1, (n, BLOCK), 0, VOCAB, dtype=jnp.int32)
    y = jax.random.randint(k2, (n, BLOCK), 0, VOCAB, dtype=jnp.int32)
    return x, y


def _leaves(m):
    return [np.asarray(a) for a in jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_inexact_array))]


# ======================================================================================
# 1. The mask itself
# ======================================================================================
class SaturatedMaskTest(unittest.TestCase):
    """p=1.0 must leave NO survivors."""

    def test_no_survivors_at_p1(self):
        for shape, axis, k in (((64, 256), 1, 4), ((64, 256), 1, 1), ((37, 91), 1, 7),
                               ((BLOCK, 2, 32), 2, 4), ((BLOCK, 2, 32), 0, 4)):
            with self.subTest(shape=shape, axis=axis, k=k):
                w = jnp.ones(shape)
                out = _fault(jax.random.PRNGKey(0), w, axis, k, 1.0)
                self.assertEqual(int(jnp.count_nonzero(out)), 0,
                                 "p=1.0 left survivors -- the fault is not saturating")

    def test_p0_is_the_identity(self):
        """The other end of the same comparison: at p=0 the mask is all ones."""
        w = jax.random.normal(jax.random.PRNGKey(1), (64, 256))
        np.testing.assert_array_equal(np.asarray(_fault(jax.random.PRNGKey(0), w, 1, K, 0.0)),
                                      np.asarray(w))


# ======================================================================================
# 2. Forward pass / inference
# ======================================================================================
class SaturatedForwardTest(unittest.TestCase):

    def test_forward_at_p1_ignores_hidden_weights(self):
        """THE CANARY. At p=1.0 the block weights cannot influence the logits at all."""
        m = _model()
        pert = _perturb_hidden(m)
        x, _ = _tokens()
        key = jax.random.PRNGKey(3)

        base_p1 = np.asarray(m(x[0], spec=P1, key=key))
        pert_p1 = np.asarray(pert(x[0], spec=P1, key=key))
        np.testing.assert_array_equal(
            base_p1, pert_p1,
            "p=1.0 logits moved when only BLOCK weights changed -- some faulted matmul is not "
            "actually being faulted (spec/key not threaded, or a new matmul with no fault site)")

        base_p0 = np.asarray(m(x[0], spec=P0, key=key))
        pert_p0 = np.asarray(pert(x[0], spec=P0, key=key))
        self.assertTrue(np.all(np.isfinite(pert_p0)), "control perturbation overflowed; lower `scale`")
        self.assertGreater(float(np.max(np.abs(pert_p0 - base_p0))), 1e-3,
                           "control: the perturbation must visibly move a FAULT-FREE forward, else the "
                           "p=1.0 invariance above is vacuous")

    def test_forward_at_p1_equals_zero_hidden_model(self):
        """p=1.0 is bitwise the same computation as zeroing every faulted matrix (both untied."""
        for tie in (False, True):
            with self.subTest(tie_embeddings=tie):
                m = _model(tie_embeddings=tie)
                ref = _zero_hidden(m)
                x, _ = _tokens()
                for seed in (3, 17):
                    got = np.asarray(m(x[0], spec=P1, key=jax.random.PRNGKey(seed)))
                    np.testing.assert_array_equal(
                        got, np.asarray(ref(x[0], spec=P0, key=jax.random.PRNGKey(seed))))
                    self.assertTrue(np.all(np.isfinite(got)))

    def test_logits_are_context_free_at_p1(self):
        """What the model degenerates INTO at p=1.0: every block output is zero, so the residual
        stream is the embedding and logits at position t depend on token t alone.
        """
        m = _model()
        key = jax.random.PRNGKey(5)
        idx = jnp.arange(BLOCK, dtype=jnp.int32) % VOCAB
        lg = np.asarray(m(idx, spec=P1, key=key))
        rev = np.asarray(m(idx[::-1], spec=P1, key=key))
        # reversing the context must just reverse the rows: no position sees any other position
        np.testing.assert_array_equal(lg, rev[::-1])
        # ...and identical tokens at different positions must get identical logits (no positional
        # information survives either, since RoPE only enters the faulted attention)
        pair = jnp.array([7, 7] + [0] * (BLOCK - 2), dtype=jnp.int32)
        two = np.asarray(m(pair, spec=P1, key=key))
        np.testing.assert_array_equal(two[0], two[1])

    def test_every_fault_site_is_invoked(self):
        """Structural coverage of the fault SITES."""
        m = _model()
        calls = []
        real = llama_mod._fault

        def rec(key, w, axis, block, p):
            calls.append((tuple(w.shape), axis, block))
            return real(key, w, axis, block, p)

        with mock.patch.object(llama_mod, "_fault", rec):
            m(_tokens()[0][0], spec=P1, key=jax.random.PRNGKey(0))

        self.assertEqual(len(calls), 9 * LAYERS,
                         f"expected 9 fault sites per block x {LAYERS} blocks, got {len(calls)}: {calls}")
        for shape, axis, block in calls:
            self.assertEqual(block, K, "a fault site used a block size other than spec.k")
            if len(shape) == 2:                      # a weight (out, in): contraction is axis 1
                self.assertEqual(axis, 1, f"weight site {shape} faulted along axis {axis}, not the "
                                          f"contraction axis 1")
            self.assertNotEqual(shape, (VOCAB, EMBD),
                                "the readout matrix was faulted; llama.py documents it as fault-free "
                                "(if that changed on purpose, update this test AND the p=1.0 "
                                "context-free invariants above, which assume it)")
        per_layer = [c for c in calls[:9]]
        self.assertEqual(sum(1 for s, _, _ in per_layer if len(s) == 3), 2,
                         "expected exactly 2 non-weight (K/V operand) fault sites per block")


# ======================================================================================
# 3. Gradients
# ======================================================================================
class SaturatedGradientTest(unittest.TestCase):
    """What the fault does to the gradient."""

    def _grads(self, m, spec, seed=7):
        x, y = _tokens()
        keys = jax.random.split(jax.random.PRNGKey(seed), x.shape[0])
        return eqx.filter_value_and_grad(compute_loss)(m, x, y, keys, spec)

    def test_hidden_gradients_are_exactly_zero_at_p1(self):
        m = _model()
        loss, g = self._grads(m, P1)
        self.assertTrue(math.isfinite(float(loss)))
        for i, w in enumerate(_hidden(g) + _block_norms(g)):
            self.assertEqual(int(jnp.count_nonzero(w)), 0,
                             f"block parameter {i} has a nonzero gradient at p=1.0 -- the loss can "
                             f"still see through the faulted blocks")
        # ...and the fault-free path is still learning, so this is a live model, not a dead graph
        for w in _readout_and_embedding(g):
            self.assertGreater(float(jnp.max(jnp.abs(w))), 0.0,
                               "the fault-free (embedding/norm/readout) path has no gradient either -- "
                               "the test model is broken, not the fault")

    def test_hidden_gradients_are_nonzero_at_p0(self):
        """Control: the same call at p=0 must train the blocks."""
        _, g = self._grads(_model(), P0)
        for i, w in enumerate(_hidden(g)):
            self.assertGreater(float(jnp.max(jnp.abs(w))), 0.0, f"hidden weight {i} has no gradient at p=0")


# ======================================================================================
# 4. Training: the real compiled block and the real optimizer
# ======================================================================================
class SaturatedTrainBlockTest(unittest.TestCase):
    """Through `_make_train_block` -- the exact jitted + shard_mapped block run_training."""

    N_STEPS = 3

    def setUp(self):
        self.mesh = jax.make_mesh((1,), ("data",))

    def _tc(self, wd=0.0):
        n = self.N_STEPS
        return TrainConfig(batch_size=BATCH, learning_rate=1e-2, max_iters=n, warmup_iters=1,
                           lr_decay_iters=n, eval_interval=n, eval_seqs=BATCH, weight_decay=wd,
                           )

    def _run_block(self, spec, wd=0.0, seed=0):
        m = _model(seed)
        optim = build_optimizer(self._tc(wd), m)
        opt_state = optim.init(eqx.partition(m, eqx.is_array)[0])
        block = _make_train_block(optim, 1, self.mesh, donate="none")
        kx, ky = jax.random.split(jax.random.PRNGKey(99))
        xs = jax.random.randint(kx, (self.N_STEPS, BATCH, BLOCK), 0, VOCAB, dtype=jnp.int32)
        ys = jax.random.randint(ky, (self.N_STEPS, BATCH, BLOCK), 0, VOCAB, dtype=jnp.int32)
        new, _, losses = block(m, opt_state, xs, ys, jax.random.PRNGKey(7), 0, spec)
        return m, new, np.asarray(losses)

    def test_hidden_weights_are_bitwise_frozen_at_p1(self):
        """With wd=0 the block weights get a zero gradient AND a zero AdamW update."""
        before, after, losses = self._run_block(P1)
        self.assertTrue(np.all(np.isfinite(losses)), f"non-finite losses: {losses}")
        for i, (b, a) in enumerate(zip(_hidden(before) + _block_norms(before),
                                       _hidden(after) + _block_norms(after))):
            np.testing.assert_array_equal(
                np.asarray(a), np.asarray(b),
                err_msg=f"block parameter {i} MOVED during training at p=1.0 -- gradient is reaching "
                        f"the faulted blocks")
        moved = [float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
                 for b, a in zip(_readout_and_embedding(before), _readout_and_embedding(after))]
        self.assertTrue(all(d > 0 for d in moved),
                        f"nothing trained at all ({moved}) -- the block is a no-op, so 'frozen' above "
                        f"proves nothing about the fault")

    def test_hidden_weights_train_at_p0(self):
        """Control for the above: the same block, same steps, p=0 -> every block weight moves."""
        before, after, _ = self._run_block(P0)
        for i, (b, a) in enumerate(zip(_hidden(before), _hidden(after))):
            self.assertGreater(float(np.max(np.abs(np.asarray(a) - np.asarray(b)))), 0.0,
                               f"hidden weight {i} did not train at p=0")

    def test_hidden_weights_only_decay_under_weight_decay(self):
        """At a REALISTIC wd the block weights do move."""
        before, after, _ = self._run_block(P1, wd=0.1)
        ratios = []
        for i, (b, a) in enumerate(zip(_hidden(before), _hidden(after))):
            b, a = np.asarray(b, dtype=np.float64), np.asarray(a, dtype=np.float64)
            keep = np.abs(b) > 1e-6                      # ratio is ill-conditioned at ~0 entries
            r = a[keep] / b[keep]
            np.testing.assert_allclose(
                r, r.flat[0], rtol=1e-4,
                err_msg=f"hidden weight {i} did not change by one uniform factor -- something other "
                        f"than weight decay (i.e. a gradient) moved it at p=1.0")
            ratios.append(float(r.flat[0]))
        self.assertTrue(all(0.0 < x < 1.0 for x in ratios), f"expected pure decay, got {ratios}")
        np.testing.assert_allclose(ratios, ratios[0], rtol=1e-4,
                                   err_msg="all hidden weights share one optimizer group, so one "
                                           "decay factor should apply to all of them")


# ======================================================================================
# 5. The eval / inference paths
# ======================================================================================
class SaturatedEvalTest(unittest.TestCase):
    """Both faulted-eval entrypoints: the in-training monitoring probe."""

    def setUp(self):
        self.mesh = jax.make_mesh((1,), ("data",))
        self.model = _model()
        self.ref = _zero_hidden(self.model)
        self.x, self.y = _tokens(seed=11)
        self.keys = jax.random.split(jax.random.PRNGKey(21), BATCH)

    def test_training_eval_block_matches_zero_hidden_model(self):
        block = _make_eval_block(self.mesh)
        xs, ys = self.x[None], self.y[None]                     # (n_chunk=1, eval_mb, block_size)
        seq_keys = self.keys[None]
        got = np.asarray(block(self.model, xs, ys, P1, seq_keys))
        ref = np.asarray(block(self.ref, xs, ys, P0, seq_keys))
        np.testing.assert_array_equal(got, ref, err_msg="the in-training faulted eval does not apply "
                                                        "the fault as a zeroing of the block matmuls")
        self.assertTrue(np.all(np.isfinite(got)))

    def test_offline_eval_block_matches_zero_hidden_model(self):
        block = make_per_seq_eval_block(jax.local_devices()[0])
        got = np.asarray(block(self.model, self.x, self.y, P1, jax.random.PRNGKey(21)))
        ref = np.asarray(block(self.ref, self.x, self.y, P0, jax.random.PRNGKey(21)))
        np.testing.assert_array_equal(got, ref, err_msg="fault_eval's scorer does not apply the fault "
                                                        "as a zeroing of the block matmuls")

    def test_offline_eval_ignores_hidden_weights_at_p1(self):
        """The canary again, on the offline scorer: perturbing the block weights must not move."""
        block = make_per_seq_eval_block(jax.local_devices()[0])
        pert = _perturb_hidden(self.model)
        k = jax.random.PRNGKey(21)
        np.testing.assert_array_equal(np.asarray(block(self.model, self.x, self.y, P1, k)),
                                      np.asarray(block(pert, self.x, self.y, P1, k)))
        self.assertGreater(float(np.max(np.abs(np.asarray(block(self.model, self.x, self.y, P0, k))
                                               - np.asarray(block(pert, self.x, self.y, P0, k))))),
                           1e-3, "control: the perturbation must move a fault-free eval")


# ======================================================================================
# 6. End to end through the real training entrypoint
# ======================================================================================
class SaturatedRunTrainingTest(unittest.TestCase):
    """The whole worker path: FaultConfig(p=1.0).to_spec() -> run_training over real token."""

    N_STEPS = 4

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="faultp1_")
        self.data_dir = os.path.join(self.dir, "data")
        os.makedirs(self.data_dir)
        rng = np.random.default_rng(0)
        for name, n in (("train", 60_000), ("val", 8_000), ("test", 4_000)):
            rng.integers(0, VOCAB, size=n, dtype=np.uint16).tofile(
                os.path.join(self.data_dir, f"{name}.bin"))
        self.mesh = jax.make_mesh((1,), ("data",))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _train(self, p, tag):
        n = self.N_STEPS
        tc = TrainConfig(batch_size=BATCH, learning_rate=1e-2, max_iters=n, warmup_iters=1,
                         lr_decay_iters=n, eval_interval=2, eval_seqs=BATCH, weight_decay=0.0,
                         )
        model = _model()
        optim = build_optimizer(tc, model)
        opt_state = optim.init(eqx.partition(model, eqx.is_array)[0])
        spec = FaultConfig(p=p, k=K).to_spec()           # exactly how the worker builds it
        results = os.path.join(self.dir, "r_" + tag)
        trained, *_, final = run_training(
            model, opt_state, optim, spec, self.data_dir, jax.random.PRNGKey(0), seed=0, start_step=0,
            n_steps=n, best_val=math.inf, batch_size=BATCH, micro_batch=BATCH, eval_interval=2,
            eval_seqs=BATCH, results_dir=results, checkpoint_dir=os.path.join(self.dir, "c_" + tag),
            mesh=self.mesh, buffer=None)
        self.assertEqual(final, n)
        with open(os.path.join(results, "metrics.json")) as f:
            metrics = json.load(f)
        return model, trained, metrics

    def test_run_training_at_p1_leaves_blocks_untouched(self):
        before, after, metrics = self._train(1.0, "p1")
        for i, (b, a) in enumerate(zip(_hidden(before) + _block_norms(before),
                                       _hidden(after) + _block_norms(after))):
            np.testing.assert_array_equal(
                np.asarray(a), np.asarray(b),
                err_msg=f"block parameter {i} moved in a real p=1.0 run -- the fault is not reaching "
                        f"training through the worker's entrypoint")
        self.assertTrue(all(math.isfinite(m["val_loss_fault"]) for m in metrics), metrics)
        # the surviving (fault-free) path trained, so this was a real run and not a stalled one
        self.assertTrue(any(float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) > 0
                            for b, a in zip(_readout_and_embedding(before),
                                            _readout_and_embedding(after))),
                        "no parameter moved at all -- the run did nothing")

    def test_run_training_at_p0_trains_blocks(self):
        before, after, _ = self._train(0.0, "p0")
        for i, (b, a) in enumerate(zip(_hidden(before), _hidden(after))):
            self.assertGreater(float(np.max(np.abs(np.asarray(a) - np.asarray(b)))), 0.0,
                               f"hidden weight {i} did not train in a real p=0 run")


if __name__ == "__main__":
    unittest.main()
