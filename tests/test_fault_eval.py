"""Unit tests for the fault-eval core in nano_llama/fault_eval.py (+ EvalResult.from_sweep_points).

Runs on CPU with a tiny synthetic Llama and synthetic token batches -- no OWT bins, no checkpoints,
no GPU (the workstation GPU is shared with live training).  Covers: the per-sequence block's mean
equals an independent scalar route; the adaptive stopping rule and both its guards;
k-major ordering and its irrelevance to the results; determinism; the forward-only batch chooser; the
EvalConfig JSON round-trip; and the parallel-list EvalResult (construction, fan-out, back-compat).

Runnable under pytest or directly (``python tests/test_fault_eval.py``).
"""
import unittest

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")   # never fight live training on the shared GPU

import jax
import jax.numpy as jnp
import numpy as np

from nano_llama.llama import LlamaConfig, Llama, FaultSpec
from nano_llama.train import per_sequence_loss
from nano_llama.train_core import TrainConfig
from nano_llama import fault_eval as fe
from nano_llama.eval_result import EvalPoint, EvalResult


def _tiny_model(seed=0):
    """A minimal Llama (fast to compile/run on CPU)."""
    cfg = LlamaConfig(block_size=16, vocab_size=32, n_layer=1, n_head=1, n_embd=8,
                      multiple_of=4, dtype="float32", attn_impl="manual", tie_embeddings=True)
    return Llama(cfg, jax.random.PRNGKey(seed)), cfg


class _StubLoader:
    """Minimal stand-in for SlidingLoader: synthetic (x, y) batches keyed on (seed, stream, step)."""

    def __init__(self, cfg, seed=0):
        self.cfg, self.seed = cfg, seed

    def batch(self, n, step=0, stream=0):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(stream), int(step)]))
        toks = rng.integers(0, self.cfg.vocab_size, size=(n, self.cfg.block_size + 1), dtype=np.int32)
        return jnp.asarray(toks[:, :-1]), jnp.asarray(toks[:, 1:])


def _scalar_faulted_loss(model, x, y, spec, keys):
    """Mean CE over the batch by the OTHER route -- train.per_sequence_loss rather than
    fault_eval's own inline vmap -- so the two are an independent cross-check."""
    return float(jnp.mean(per_sequence_loss(model, x, y, spec=spec, keys=keys)))


def _sweep(model, loader, pairs, *, seed=0, **kw):
    """The eval WORKER's loop, verbatim (orchestrator_hooks/eval_worker.py): k-major visit
    order, one loader stream per point, fault key folded on the point's INDEX, and results
    written back in the caller's order.

    Asserted on directly rather than through a library wrapper, because this is the loop
    that actually runs on the cluster.
    """
    block = fe.make_per_seq_eval_block(jax.local_devices()[0])
    base = jax.random.PRNGKey(seed)
    out = [None] * len(pairs)
    for i in fe.k_major_order(pairs):
        k, p = pairs[i]
        out[i] = fe.estimate_point(block, model, loader, k, p, stream=i,
                                   key=jax.random.fold_in(base, i), **kw)
    return out


class FaultEvalTest(unittest.TestCase):
    def test_per_seq_mean_matches_the_independent_scalar_route(self):
        """mean over the per-sequence block == the same quantity via per_sequence_loss."""
        model, cfg = _tiny_model()
        x, y = _StubLoader(cfg).batch(64)
        spec = FaultSpec(p=0.05, k=4)
        key = jax.random.PRNGKey(7)

        block = fe.make_per_seq_eval_block(jax.local_devices()[0])
        per_seq = np.asarray(block(model, x, y, spec, key))
        # the oracle takes per-sequence keys; build them the same way the per-seq block does.
        scalar = _scalar_faulted_loss(model, x, y, spec, jax.random.split(key, x.shape[0]))

        assert per_seq.shape == (64,)
        assert abs(per_seq.mean() - scalar) < 1e-4, (per_seq.mean(), scalar)


    def test_sweep_returns_a_point_per_pair_in_caller_order(self):
        """One point per (k, p), in the CALLER's order (not the k-major evaluation order)."""
        model, cfg = _tiny_model()
        pairs = [(4, 0.0), (2, 0.02), (4, 0.1)]           # deliberately not k-sorted
        sweep = _sweep(model, _StubLoader(cfg), pairs, batch_size=64,
                       min_evals=128, max_evals=256, seed=1)
        assert [(pt.k, pt.p) for pt in sweep] == pairs
        for pt in sweep:
            assert np.isfinite(pt.mean) and pt.mean > 0
            assert pt.se >= 0.0 and pt.se < pt.mean          # SE of a mean is small vs the mean itself


    def test_k_major_order_groups_k_and_is_stable(self):
        """The evaluation order visits each k contiguously, preserving p-order within a k."""
        pairs = [(4, 0.0), (2, 0.01), (4, 0.1), (2, 0.2), (4, 0.3)]
        order = fe.k_major_order(pairs)
        ks = [pairs[i][0] for i in order]
        assert ks == sorted(ks)                              # all of one k before the next
        # stability: within k=4 the p's keep their original relative order
        assert [pairs[i][1] for i in order if pairs[i][0] == 4] == [0.0, 0.1, 0.3]


    def test_point_estimate_is_independent_of_evaluation_order(self):
        """A point's estimate depends on its INDEX, not on when it was reached."""
        model, cfg = _tiny_model()
        pairs = [(4, 0.0), (2, 0.05), (4, 0.05)]        # not k-sorted -> the sweep really does reorder
        kw = dict(batch_size=32, min_evals=64, max_evals=128)
        sweep = _sweep(model, _StubLoader(cfg, seed=5), pairs, seed=11, **kw)

        block = fe.make_per_seq_eval_block(jax.local_devices()[0])
        base = jax.random.PRNGKey(11)
        for i, (k, p) in enumerate(pairs):
            solo = fe.estimate_point(block, model, _StubLoader(cfg, seed=5), k, p, stream=i,
                                     key=jax.random.fold_in(base, i), **kw)
            assert (solo.k, solo.p) == (sweep[i].k, sweep[i].p)
            assert (solo.mean, solo.se, solo.n_seq) == (sweep[i].mean, sweep[i].se, sweep[i].n_seq), \
                f"point {i} ({k}, {p}) differs when scored alone -> a resume would change it"


    def test_distinct_streams_give_points_independent_data(self):
        """Points at the SAME (k, p) but different streams see different sequences."""
        model, cfg = _tiny_model()
        loader = _StubLoader(cfg, seed=5)
        a0, _ = loader.batch(8, step=0, stream=0)
        b0, _ = loader.batch(8, step=0, stream=1)
        a0_again, _ = loader.batch(8, step=0, stream=0)
        assert not np.array_equal(np.asarray(a0), np.asarray(b0)), "different streams must differ"
        assert np.array_equal(np.asarray(a0), np.asarray(a0_again)), "same (stream, step) must repeat"


    def test_adaptive_stop_respects_min_and_max_evals(self):
        """min_evals is a hard floor; max_evals is a hard-ish cap (overshoot < one batch) which, when
        it binds, flags the point reached_target=False.
        """
        model, cfg = _tiny_model()
        block = fe.make_per_seq_eval_block(jax.local_devices()[0])
        loader = _StubLoader(cfg)

        # An unreachable target -> must run to the cap and report failure.
        tight = fe.estimate_point(block, model, loader, 4, 0.05, stream=0, batch_size=32,
                                  key=jax.random.PRNGKey(0), target_se=1e-9,
                                  min_evals=32, max_evals=128)
        assert tight.reached_target is False
        assert 128 <= tight.n_seq < 128 + 32          # cap, plus at most one full batch of overshoot

        # A trivially loose target -> stops as soon as it is allowed to, i.e. at the min_evals floor.
        loose = fe.estimate_point(block, model, loader, 4, 0.05, stream=1, batch_size=32,
                                  key=jax.random.PRNGKey(0), target_se=10.0,
                                  min_evals=96, max_evals=100_000)
        assert loose.reached_target is True
        assert 96 <= loose.n_seq < 96 + 32
        assert loose.se <= 10.0


    def test_adaptive_stop_actually_hits_the_precision_target(self):
        """When the target is attainable, the returned se really is at or under target_se (nats)."""
        model, cfg = _tiny_model()
        block = fe.make_per_seq_eval_block(jax.local_devices()[0])
        pt = fe.estimate_point(block, model, _StubLoader(cfg), 4, 0.02, stream=0, batch_size=64,
                               key=jax.random.PRNGKey(3), target_se=0.05,
                               min_evals=64, max_evals=100_000)
        assert pt.reached_target is True
        assert pt.se <= 0.05


    def test_p0_is_deterministic_across_seeds(self):
        """At p=0 there is no fault, so the loss is fault-seed-independent (mask is all-ones)."""
        model, cfg = _tiny_model()
        x, y = _StubLoader(cfg).batch(32)
        block = fe.make_per_seq_eval_block(jax.local_devices()[0])
        a = np.asarray(block(model, x, y, FaultSpec(p=0.0, k=4), jax.random.PRNGKey(1)))
        b = np.asarray(block(model, x, y, FaultSpec(p=0.0, k=4), jax.random.PRNGKey(999)))
        assert np.allclose(a, b), "p=0 must be the clean, fault-free path"


    def test_sweep_is_deterministic(self):
        """Same model + same sampler seed + same fault seed -> identical estimates."""
        model, cfg = _tiny_model()
        kw = dict(batch_size=64, min_evals=128, max_evals=256, seed=3)
        s1 = _sweep(model, _StubLoader(cfg, seed=5), [(4, 0.0), (4, 0.05)], **kw)
        s2 = _sweep(model, _StubLoader(cfg, seed=5), [(4, 0.0), (4, 0.05)], **kw)
        for a, b in zip(s1, s2):
            assert a.p == b.p and a.n_seq == b.n_seq
            assert a.mean == b.mean and a.se == b.se


    def test_eval_batch_chooser_clamps_and_scales(self):
        """Forward-only chooser: clamped to max_evals and max_micro_batch; bigger model -> smaller batch."""
        dev = jax.local_devices()[0]
        small = LlamaConfig(block_size=256, vocab_size=8192, n_layer=4, n_head=4, n_embd=256,
                            multiple_of=64, dtype="bfloat16", attn_impl="flash")
        big = LlamaConfig(block_size=1024, vocab_size=8192, n_layer=16, n_head=16, n_embd=1024,
                          multiple_of=256, dtype="bfloat16", attn_impl="flash")
        b_small = fe.choose_eval_micro_batch(small, 5_000_000, dev, max_evals=100_000, max_micro_batch=256)
        b_big = fe.choose_eval_micro_batch(big, 200_000_000, dev, max_evals=100_000, max_micro_batch=256)
        assert 1 <= b_big <= b_small <= 256, (b_big, b_small)       # heavier model needs a smaller batch
        assert fe.choose_eval_micro_batch(small, 5_000_000, dev, max_evals=8) == 8   # clamped to the cap


    def test_eval_config_roundtrip(self, tmp_path=None):
        """EvalConfig round-trips through JSON (ConfigMixin) and normalises kp_pairs to (int, float)."""
        import tempfile
        ec = fe.EvalConfig(kp_pairs=[[4, 0.0], [4, 0.01], [8, 0.05]], target_se=0.02,
                           min_evals=512, max_evals=4096, batch_size=None, seed=7)
        assert ec.kp_pairs == ((4, 0.0), (4, 0.01), (8, 0.05))   # __post_init__ -> tuple of (int, float)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "eval_config.json")
            ec.save(path)
            loaded = fe.EvalConfig.load(path)
        # JSON renders tuples as lists; __post_init__ re-tuples on load so frozen eq holds
        assert loaded == ec
        assert loaded.batch_size is None and loaded.target_se == 0.02 and loaded.min_evals == 512
        # a pinned batch_size and the defaults also round-trip
        assert fe.EvalConfig(batch_size=32).batch_size == 32
        assert fe.EvalConfig().target_se == fe.DEFAULT_TARGET_SE


    def test_eval_result_from_sweep_points(self):
        """from_sweep_points unzips a whole sweep into ONE record with parallel eval tuples."""
        model, cfg = _tiny_model()
        tc = TrainConfig()
        pairs = [(4, 0.0), (4, 0.05), (2, 0.05)]
        sweep = _sweep(model, _StubLoader(cfg), pairs, batch_size=64,
                       min_evals=128, max_evals=256, seed=0)
        r = EvalResult.from_sweep_points(sweep, model_config=cfg, train_config=tc, k_train=4,
                                         p_train=0.01, total_n_params=model.count_params(),
                                         n_non_embedding_params=model.count_non_embedding_params(),
                                         n_train_tokens=123456)
        assert r.n_eval_points == 3
        assert r.k_eval == (4.0, 4.0, 2.0) and r.p_eval == (0.0, 0.05, 0.05)
        assert r.k_train == 4 and r.p_train == 0.01            # train fault stays scalar, and is retained
        assert r.eval_loss == tuple(pt.mean for pt in sweep)
        assert r.n_eval_seq == tuple(pt.n_seq for pt in sweep)
        assert r.eval_reached_se_target == tuple(pt.reached_target for pt in sweep)
        assert r.from_dict(r.to_dict()) == r                    # JSON round-trip (tuples survive)


    def test_eval_result_points_fan_out(self):
        """points() flattens into one row per eval condition, each carrying the model-level (N, D)."""
        model, cfg = _tiny_model()
        r = EvalResult(model_config=cfg, train_config=TrainConfig(), k_train=4, p_train=0.01,
                       k_eval=(4, 4, 2), p_eval=(0.0, 0.05, 0.05), eval_loss=(3.1, 3.4, 3.9),
                       total_n_params=1000, n_train_tokens=123456, n_non_embedding_params=400,
                       eval_se=(0.01, 0.02, 0.03), n_eval_seq=(100, 200, 300),
                       eval_reached_se_target=(True, True, False))
        pts = r.points()
        assert len(pts) == 3 and all(isinstance(p, EvalPoint) for p in pts)
        assert [(p.k_eval, p.p_eval, p.eval_loss) for p in pts] == [
            (4.0, 0.0, 3.1), (4.0, 0.05, 3.4), (2.0, 0.05, 3.9)]
        assert pts[2].reached_se_target is False
        for p in pts:                                          # model-level scalars ride along on each row
            assert p.total_n_params == 1000 and p.n_train_tokens == 123456
            assert p.n_non_embedding_params == 400 and p.size_key == r.size_key


    def test_eval_result_rejects_ragged_parallel_fields(self):
        """A length mismatch between the parallel fields is caught at construction."""
        model, cfg = _tiny_model()
        try:
            EvalResult(model_config=cfg, train_config=TrainConfig(), k_train=4, p_train=0.01,
                       k_eval=(4, 4), p_eval=(0.0,), eval_loss=(3.1, 3.4),      # p_eval is short
                       total_n_params=1000, n_train_tokens=123456,
                       n_non_embedding_params=400, eval_se=(0.01, 0.02),
                       n_eval_seq=(100, 200), eval_reached_se_target=(True, True))
        except ValueError as e:
            assert "parallel" in str(e)
        else:
            raise AssertionError("expected a ValueError for ragged parallel fields")


if __name__ == "__main__":
    unittest.main()
