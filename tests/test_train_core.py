"""End-to-end integration tests for nano_llama.train_core.run_training.

Exercises BOTH loader modes selected by the `buffer` argument:
  * buffer=None      -> whole train.bin resident in RAM (a single-block SlidingLoader)
  * buffer=<small>   -> sliding buffer small enough to force a mid-training megablock refresh
driving a tiny real model through the actual loop, and a resume-from-checkpoint for each.

Run on CPU:
    JAX_PLATFORMS=cpu PYTHONPATH=<repo> /home/trevor/scienv/bin/python -m unittest tests.test_train_core -v
"""
import os
import json
import math
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import jax
import equinox as eqx

from nano_llama.token_data import SlidingLoader

from tests._test_util import build_model_config
from nano_llama.llama import Llama
from nano_llama.fault import FaultConfig
import nano_llama.train_core as tc_mod
from nano_llama.train_core import (build_optimizer, TrainConfig, load_checkpoint, run_training,
                                   run_training_loop, choose_loader_buffer, run_slides,
                                   resolve_eval_seqs, choose_eval_chunk, probe_checkpoint_iter,
                                   BUFFER_GB)

VOCAB = 8192


def _write_random_bins(dir_, n_train=200_000, n_val=20_000, n_test=10_000, seed=0):
    rng = np.random.default_rng(seed)
    for name, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        rng.integers(0, VOCAB, size=n, dtype=np.uint16).tofile(os.path.join(dir_, f"{name}.bin"))


class TrainCoreTest(unittest.TestCase):
    BLOCK, BS = 64, 8
    BUF_TOKENS = 1024                       # sliding: R = 1024 // (8*64) = 2 -> refresh mid-training

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="traincore_")
        self.data_dir = os.path.join(self.dir, "data"); os.makedirs(self.data_dir)
        _write_random_bins(self.data_dir)
        self.sliding_gb = self.BUF_TOKENS * 2 / 1e9
        self.mesh = jax.make_mesh((1,), ("data",))
        self.spec = FaultConfig(p=0.0, k=4).to_spec()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    # ---- fixtures ----
    def _model_opt(self, tc, seed=0):
        mc = build_model_config(self.BS * 8, 2, block_size=self.BLOCK, dtype="float32", attn_impl="manual")
        key, mk = jax.random.split(jax.random.PRNGKey(seed))
        model = Llama(mc, mk)
        optim = build_optimizer(tc, model)
        return model, optim, optim.init(eqx.partition(model, eqx.is_array)[0]), key

    def _tc(self, max_iters=4):
        return TrainConfig(batch_size=self.BS, learning_rate=1e-3, max_iters=max_iters,
                           warmup_iters=1, lr_decay_iters=max_iters, eval_interval=2, eval_seqs=self.BS,
                           )

    def _run(self, buffer, model, optim, opt_state, key, start_step, n_steps, best):
        return run_training(
            model, opt_state, optim, self.spec, self.data_dir, key, seed=0, start_step=start_step,
            n_steps=n_steps, best_val=best, batch_size=self.BS, micro_batch=self.BS,
            eval_interval=2, eval_seqs=self.BS, results_dir=os.path.join(self.dir, "results"),
            checkpoint_dir=os.path.join(self.dir, "checkpoint"), mesh=self.mesh, buffer=buffer)

    def _metric_steps(self):
        with open(os.path.join(self.dir, "results", "metrics.json")) as f:
            return json.load(f)

    def _train_log(self):
        with open(os.path.join(self.dir, "results", "train_loss.jsonl")) as f:
            return [json.loads(l) for l in f if l.strip()]

    def _loop_raw(self, spec, start, n, model, opt_state, key, best, checkpoint_iter, tag,
                  eval_interval=2):
        """Call run_training_loop DIRECTLY (no time-probe): build step-keyed loader objects."""
        train_loader = SlidingLoader(os.path.join(self.data_dir, "train.bin"), self.BLOCK, seed=0)
        val_loader = SlidingLoader(os.path.join(self.data_dir, "val.bin"), self.BLOCK, seed=0)
        return run_training_loop(
            model, opt_state, build_optimizer(self._tc(6), model), spec, key, start, n, best,
            self.BS, self.BS, eval_interval, self.BS, os.path.join(self.dir, "r_" + tag),
            os.path.join(self.dir, "c_" + tag), self.mesh, train_loader, val_loader, checkpoint_iter)

    def _train_log_for(self, tag):
        with open(os.path.join(self.dir, "r_" + tag, "train_loss.jsonl")) as f:
            return [json.loads(l) for l in f if l.strip()]

    def _metric_steps_for(self, tag):
        with open(os.path.join(self.dir, "r_" + tag, "metrics.json")) as f:
            return [m["step"] for m in json.load(f)]

    def _worst_leaf_delta(self, a, b):
        la = jax.tree_util.tree_leaves(eqx.filter(a, eqx.is_inexact_array))
        lb = jax.tree_util.tree_leaves(eqx.filter(b, eqx.is_inexact_array))
        self.assertEqual(len(la), len(lb))
        return max(float(np.max(np.abs(np.asarray(x) - np.asarray(y)))) for x, y in zip(la, lb))

    # ---- shared bodies ----
    def _end_to_end(self, buffer):
        tc = self._tc(4)
        model, optim, opt_state, key = self._model_opt(tc)
        *_, final = self._run(buffer, model, optim, opt_state, key, 0, 4, math.inf)
        self.assertEqual(final, 4)
        metrics = self._metric_steps()
        self.assertEqual([m["step"] for m in metrics], [2, 4])
        for m in metrics:
            for kf in ("val_loss_fault", "val_loss_fault_se"):
                self.assertTrue(math.isfinite(m[kf]), f"{kf}@{m['step']} not finite")
            self.assertNotIn("train_loss", m, "the train curve belongs in train_loss.jsonl, not metrics")
        # The training curve is logged per BLOCK (checkpoint cadence), independently of evals: 4 steps
        # in blocks of <= 2 -> the steps covered must tile [0, 4) exactly once.
        rows = self._train_log()
        self.assertEqual(sum(r["n_steps"] for r in rows), 4, "train log must cover every step once")
        self.assertEqual([r["step"] for r in rows], sorted(r["step"] for r in rows))
        for r in rows:
            self.assertTrue(math.isfinite(r["train_loss"]), f"train_loss@{r['step']} not finite")
        self.assertTrue(os.path.isfile(os.path.join(self.dir, "checkpoint", "meta.json")))
        self.assertTrue(os.path.isfile(os.path.join(self.dir, "results", "best_model.eqx")))

    def _resume(self, buffer):
        tc = self._tc(4)
        model, optim, opt_state, key = self._model_opt(tc)
        self._run(buffer, model, optim, opt_state, key, 0, 2, math.inf)     # 0 -> 2
        skel = Llama(build_model_config(self.BS * 8, 2, block_size=self.BLOCK,
                                        dtype="float32", attn_impl="manual"), jax.random.PRNGKey(1))
        skel_opt = build_optimizer(tc, skel).init(eqx.partition(skel, eqx.is_array)[0])
        loaded = load_checkpoint(os.path.join(self.dir, "checkpoint"), skel, skel_opt)
        self.assertIsNotNone(loaded)
        model, opt_state, start_step, best, key = loaded
        self.assertEqual(start_step, 2)
        *_, final = self._run(buffer, model, build_optimizer(tc, model), opt_state, key, start_step,
                              4 - start_step, best)                          # 2 -> 4
        self.assertEqual(final, 4)
        self.assertEqual([m["step"] for m in self._metric_steps()], [2, 4])

    # ---- tests: both loader modes ----
    def test_resolve_eval_seqs(self):
        """The eval sample size comes from the CONFIG (eval_seqs)."""
        base = dict(batch_size=128, max_iters=100, warmup_iters=1, lr_decay_iters=100, eval_interval=10)
        self.assertEqual(resolve_eval_seqs(TrainConfig(**base)), TrainConfig.eval_seqs)
        # an explicit size is snapped UP to a whole number of nominal batch_size draws
        self.assertEqual(resolve_eval_seqs(TrainConfig(**base, eval_seqs=2048)), 2048)
        self.assertEqual(resolve_eval_seqs(TrainConfig(**base, eval_seqs=2000)), 2048)
        self.assertEqual(resolve_eval_seqs(TrainConfig(**base, eval_seqs=1)), 128)

    def test_choose_eval_chunk_divides_and_shards(self):
        """The chunking is a pure memory choice."""
        mc = build_model_config(64, 2, block_size=self.BLOCK, dtype="float32", attn_impl="manual")
        dev = jax.local_devices()[0]
        for n_eval in (128, 1024, 8192):
            for n_dev in (1, 2, 4):
                mb = choose_eval_chunk(mc, n_eval, 10_000_000, dev, n_dev)
                self.assertEqual(n_eval % mb, 0, f"{mb} must divide {n_eval}")
                self.assertEqual(mb % n_dev, 0, f"{mb} must be a multiple of n_dev {n_dev}")

    def test_run_slides(self):
        # gates the train prefetch: True iff a run [start, start+n) crosses a refresh boundary (steps
        # start..start+n-1, since the last training step is end-1). R = refresh cadence in steps.
        R = 9536
        self.assertFalse(run_slides(0, 6480, R))      # search: whole run in one megablock -> no prefetch
        self.assertTrue(run_slides(0, 20000, R))      # long run: slides -> prefetch
        self.assertTrue(run_slides(0, R + 1, R))      # steps 0..R -> crosses the boundary at R
        self.assertFalse(run_slides(0, R, R))         # steps 0..R-1 -> fills the block, never crosses
        self.assertFalse(run_slides(R, 600, R))       # resume starting ON a boundary, short segment
        self.assertTrue(run_slides(R - 5, 100, R))    # segment straddles the boundary at R
        self.assertFalse(run_slides(0, 0, R))         # zero-length segment (resume at end)

    def test_end_to_end_simple(self):
        self._end_to_end(buffer=None)

    def test_end_to_end_sliding(self):
        self._end_to_end(buffer=self.sliding_gb)

    def test_resume_simple(self):
        self._resume(buffer=None)

    def test_resume_sliding(self):
        self._resume(buffer=self.sliding_gb)

    # ---- background prefetch actually serves the training loop ----
    def test_background_prefetch_serves_training(self):
        """During a sliding-buffer training run."""
        stats = {"ready": 0, "pending": 0, "sync": 0}

        class _SpyLoader(SlidingLoader):
            def _ensure_resident(self, pos):
                if not self.path.endswith("train.bin"):        # only the sliding TRAIN loader; skip val
                    return super()._ensure_resident(pos)
                # classify BEFORE delegating (super() mutates prefetch state)
                if not (self._resident_pos == pos and self._resident is not None):
                    if self._exec is not None and self._prefetch_pos == pos and self._prefetch_future is not None:
                        stats["ready" if self._prefetch_future.done() else "pending"] += 1
                    else:
                        stats["sync"] += 1
                super()._ensure_resident(pos)

        tc = self._tc(6)                          # 3 blocks (steps 0,2,4) -> 2 refreshes after the first
        model, optim, opt_state, key = self._model_opt(tc)
        with mock.patch("nano_llama.train_core.SlidingLoader", _SpyLoader):
            self._run(self.sliding_gb, model, optim, opt_state, key, 0, 6, math.inf)

        self.assertEqual(stats["sync"], 1, "only the very first megablock should be a synchronous load")
        self.assertGreaterEqual(stats["ready"] + stats["pending"], 2,
                                "every later refresh must be a prefetch hit, not a fresh sync load")
        self.assertGreaterEqual(stats["ready"], 1,
                                "at least one refresh must be served from a prefetch already COMPLETED "
                                "in the background (proves the background thread ran ahead of the loop)")

    # ---- checkpoint cadence (decoupled from eval) ----
    def _record_checkpoint_steps(self, checkpoint_seconds):
        """Run a short training with save_checkpoint mocked to record the steps it's called at."""
        import threading
        recorded, lock = [], threading.Lock()

        def rec(ckpt_dir, m, o, step, best, key):
            with lock:
                recorded.append(step)

        tc = self._tc(4)                                   # eval_interval=2, 4 steps -> eval @ 0,2,4
        model, optim, opt_state, key = self._model_opt(tc)
        with mock.patch("nano_llama.train_core.save_checkpoint", rec):
            run_training(model, opt_state, optim, self.spec, self.data_dir, key, seed=0, start_step=0,
                         n_steps=4, best_val=math.inf, batch_size=self.BS, micro_batch=self.BS,
                         eval_interval=2, eval_seqs=self.BS, results_dir=os.path.join(self.dir, "r"),
                         checkpoint_dir=os.path.join(self.dir, "c"), mesh=self.mesh, buffer=None,
                         checkpoint_seconds=checkpoint_seconds)
        return sorted(set(recorded))

    def test_checkpoint_cadence_short_is_sub_eval(self):
        # tiny checkpoint_seconds -> block shrinks to 1 -> checkpoints between evals (at odd steps too)
        steps = self._record_checkpoint_steps(1e-9)
        self.assertTrue(any(s % 2 == 1 for s in steps),
                        f"expected sub-eval-boundary checkpoints, got {steps}")

    def test_checkpoint_cadence_long_is_every_eval(self):
        # huge checkpoint_seconds -> block == eval_interval -> checkpoints only on eval boundaries
        steps = self._record_checkpoint_steps(1e9)
        self.assertTrue(all(s % 2 == 0 for s in steps),
                        f"expected only eval-boundary checkpoints, got {steps}")

    # ---- the block need not divide eval_interval ----
    # Snapping the block down to a divisor was a cliff: eval_intervals here are mostly prime,
    # so no divisor sits near the wall-clock target and the block collapses to 1-2 steps. The
    # loop never needed the divisibility -- it clamps each block to the next eval boundary.
    # These pin both halves of that claim.

    def test_probe_block_does_not_collapse_on_prime_eval_interval(self):
        """The regression test. With a PRIME eval_interval and a wall-clock target of ~100."""
        tc = self._tc(6)
        model, optim, opt_state, key = self._model_opt(tc)
        ei = 1787                                            # prime, and a real value from the sweep
        # First probe: a target so large only stage_cap/eval_interval can bind -> also gives step_time.
        wide, step_time = probe_checkpoint_iter(model, opt_state, optim, self.spec, key, self.BS,
                                                self.BS, ei, self.mesh, checkpoint_seconds=1e9)
        self.assertEqual(wide, ei, "an unbounded target must give exactly one eval span per block")
        # Second probe: aim at ~100 steps of wall clock. The timing is noisy (two independent probes),
        # so assert a wide band -- the point is that it is nowhere near the collapsed value of 1.
        chunk, _ = probe_checkpoint_iter(model, opt_state, optim, self.spec, key, self.BS, self.BS, ei,
                                         self.mesh, checkpoint_seconds=100 * step_time)
        self.assertGreater(chunk, 2, f"block collapsed to {chunk} on a prime eval_interval")
        self.assertTrue(10 <= chunk <= 1000, f"block {chunk} is not near the ~100-step target")
        self.assertLessEqual(chunk, ei)

    def test_probe_block_is_clamped_to_one_step_by_a_tiny_target(self):
        """The other end: a target below one step still yields a legal block (>= 1), not 0."""
        tc = self._tc(6)
        model, optim, opt_state, key = self._model_opt(tc)
        chunk, _ = probe_checkpoint_iter(model, opt_state, optim, self.spec, key, self.BS, self.BS,
                                         1787, self.mesh, checkpoint_seconds=1e-9)
        self.assertEqual(chunk, 1)

    def test_probe_block_respects_the_stage_budget(self):
        """The memory cap still binds: per-device staged bytes are chunk * (batch/n_dev) *."""
        cap_gb = (16 * self.BS * self.BLOCK * 8) / 1024 ** 3     # budget for exactly 16 steps
        tc = self._tc(6)
        model, optim, opt_state, key = self._model_opt(tc)
        chunk, _ = probe_checkpoint_iter(model, opt_state, optim, self.spec, key, self.BS, self.BS,
                                         1787, self.mesh, checkpoint_seconds=1e9, stage_budget_gb=cap_gb)
        self.assertEqual(chunk, 16)

    def test_evals_stay_on_the_grid_with_a_non_dividing_block(self):
        """A block that does NOT divide eval_interval must still put every eval exactly on the grid:
        the loop clamps the last block of each span short.
        """
        m, _, o, k = self._model_opt(self._tc(12))
        self._loop_raw(self.spec, 0, 12, m, o, k, math.inf, 4, "grid", eval_interval=6)
        self.assertEqual(self._metric_steps_for("grid"), [6, 12])
        rows = self._train_log_for("grid")
        self.assertEqual([r["step"] for r in rows], [4, 6, 10, 12], "blocks must break AT each boundary")
        self.assertEqual([r["n_steps"] for r in rows], [4, 2, 4, 2])
        self.assertEqual(sum(r["n_steps"] for r in rows), 12, "every step trained exactly once")

    def test_faulted_trajectory_is_invariant_to_a_non_dividing_block(self):
        """The trajectory must not depend on the block size EVEN WHEN the block does not."""
        spec = FaultConfig(p=0.5, k=4).to_spec()             # high p: any draw drift would diverge loudly
        m0, _, o0, k0 = self._model_opt(self._tc(12))
        odd = self._loop_raw(spec, 0, 12, m0, o0, k0, math.inf, 4, "nd_odd", eval_interval=6)[0]
        m1, _, o1, k1 = self._model_opt(self._tc(12))
        one = self._loop_raw(spec, 0, 12, m1, o1, k1, math.inf, 1, "nd_one", eval_interval=6)[0]
        worst = self._worst_leaf_delta(odd, one)
        self.assertLess(worst, 1e-6, f"weights differ across block sizes (max|delta|={worst:.3e})")

    def test_faulted_trajectory_is_chunk_invariant(self):
        """A FAULTED run's final weights are the same regardless of the checkpoint block size."""
        spec = FaultConfig(p=0.5, k=4).to_spec()             # high p: any draw drift would diverge loudly
        m0, _, o0, k0 = self._model_opt(self._tc(6), seed=0)
        big = self._loop_raw(spec, 0, 6, m0, o0, k0, math.inf, 2, "big")[0]      # block = 2 (== eval_interval)
        m1, _, o1, k1 = self._model_opt(self._tc(6), seed=0)
        small = self._loop_raw(spec, 0, 6, m1, o1, k1, math.inf, 1, "small")[0]  # block = 1
        worst = self._worst_leaf_delta(big, small)
        self.assertLess(worst, 1e-6, f"faulted weights differ across block sizes (max|Δ|={worst:.3e})")

    def test_resume_matches_continuous_run(self):
        """Capstone: a preempted+resumed run reproduces the never-preempted trajectory."""
        spec = FaultConfig(p=0.3, k=4).to_spec()

        # continuous 0 -> 6, block = 2
        m0, _, o0, k0 = self._model_opt(self._tc(6), seed=0)
        cont = self._loop_raw(spec, 0, 6, m0, o0, k0, math.inf, 2, "cont")[0]

        # preempted with block = 1: 0 -> 4, checkpoint, then resume 4 -> 6 (same "pre" dirs -> continues).
        # Resume points are eval-interval-aligned (the loop requires it); data/faults/key are all
        # reconstructed, and the block size differs from the continuous run.
        m1, _, o1, k1 = self._model_opt(self._tc(6), seed=0)
        self._loop_raw(spec, 0, 4, m1, o1, k1, math.inf, 1, "pre")
        skel = Llama(build_model_config(self.BS * 8, 2, block_size=self.BLOCK,
                                        dtype="float32", attn_impl="manual"), jax.random.PRNGKey(1))
        skel_opt = build_optimizer(self._tc(6), skel).init(eqx.partition(skel, eqx.is_array)[0])
        model, opt_state, start_step, best, key = load_checkpoint(os.path.join(self.dir, "c_pre"),
                                                                  skel, skel_opt)
        self.assertEqual(start_step, 4)
        resumed = self._loop_raw(spec, start_step, 6 - start_step, model, opt_state, key, best, 1, "pre")[0]

        worst = self._worst_leaf_delta(cont, resumed)
        self.assertLess(worst, 1e-5, f"resumed run diverged from continuous (max|Δ|={worst:.3e})")

    def test_run_log_records_per_segment(self):
        """log_run_provenance APPENDS one record per (re)start to results_dir/run_log.jsonl."""
        m0, _, o0, k0 = self._model_opt(self._tc(6), seed=0)
        self._loop_raw(self.spec, 0, 4, m0, o0, k0, math.inf, 2, "log")   # segment 1: block 2
        self._loop_raw(self.spec, 0, 2, m0, o0, k0, math.inf, 1, "log")   # segment 2 (same dir): block 1
        with open(os.path.join(self.dir, "r_log", "run_log.jsonl")) as f:
            recs = [json.loads(line) for line in f if line.strip()]
        # each segment appends a provenance record (start/resume) AND an io_summary record.
        prov = [r for r in recs if r.get("event") in ("start", "resume")]
        summ = [r for r in recs if r.get("event") == "io_summary"]
        self.assertEqual(len(prov), 2, "each (re)start must append a provenance record")
        self.assertEqual(len(summ), 2, "each segment must append an io_summary record")
        for kf in ("time", "host", "pid", "n_devices", "device_kind", "jax_version", "backend",
                   "batch_size", "micro_batch", "accum_steps", "checkpoint_iter", "model", "fault"):
            self.assertIn(kf, prov[0])
        self.assertEqual(prov[0]["checkpoint_iter"], 2)
        self.assertEqual(prov[1]["checkpoint_iter"], 1)          # the differing runtime choice is recorded
        self.assertEqual(prov[0]["n_devices"], self.mesh.devices.size)
        self.assertEqual(prov[0]["batch_size"], self.BS)
        self.assertEqual(prov[0]["model"]["n_params"], prov[1]["model"]["n_params"])
        # io_summary carries the data-staging accounting used to quantify FS-boundedness
        for kf in ("segment_wall_s", "steps_trained", "train_stage_s", "train_stage_frac",
                   "val_read_s", "n_evals"):
            self.assertIn(kf, summ[0])

    # ---- early-stop on divergence ----
    def test_divergence_stops_before_max_iters(self):
        """An absurd LR NaNs the loss within a couple of steps."""
        tc = TrainConfig(batch_size=self.BS, learning_rate=1e10, max_iters=10, warmup_iters=1,
                         lr_decay_iters=10, eval_interval=2, eval_seqs=8,
                         )
        model, optim, opt_state, key = self._model_opt(tc)
        *_, final = self._run(None, model, optim, opt_state, key, 0, 10, math.inf)
        self.assertLess(final, 10, "diverged run must stop before max_iters, not train to the end")
        metrics = self._metric_steps()
        last = max(metrics, key=lambda r: r["step"])
        self.assertFalse(math.isfinite(last["val_loss_fault"]),
                         "final recorded val loss must be non-finite on divergence")
        self.assertEqual(last["step"], final, "the divergence sentinel is recorded at the stop step")

    def test_max_loss_stops_before_max_iters_while_finite(self):
        """A FINITE block-mean loss above `max_loss` must also early-stop (a run whose loss."""
        tc = self._tc(10)
        model, optim, opt_state, key = self._model_opt(tc)
        *_, final = run_training(
            model, opt_state, optim, self.spec, self.data_dir, key, seed=0, start_step=0,
            n_steps=10, best_val=math.inf, batch_size=self.BS, micro_batch=self.BS,
            eval_interval=2, eval_seqs=self.BS, results_dir=os.path.join(self.dir, "results"),
            checkpoint_dir=os.path.join(self.dir, "checkpoint"), mesh=self.mesh, buffer=None,
            max_loss=1.0)   # far below the ~9 (ln vocab) starting loss -> the first finite block exceeds it
        self.assertLess(final, 10, "a run over max_loss must stop before max_iters")
        last = max(self._metric_steps(), key=lambda r: r["step"])
        self.assertFalse(math.isfinite(last["val_loss_fault"]),
                         "the divergence sentinel (non-finite final metric) must be recorded")
        self.assertEqual(last["step"], final)

    # ---- best-model write toggle ----
    def _run_save_best(self, save_best, tag):
        """One 4-step run into its own results/checkpoint dirs. Returns (best_val, results, ckpt)."""
        tc = self._tc(4)
        model, optim, opt_state, key = self._model_opt(tc)
        results, ckpt = os.path.join(self.dir, "r_" + tag), os.path.join(self.dir, "c_" + tag)
        *_, best, _final = run_training(
            model, opt_state, optim, self.spec, self.data_dir, key, seed=0, start_step=0,
            n_steps=4, best_val=math.inf, batch_size=self.BS, micro_batch=self.BS,
            eval_interval=2, eval_seqs=self.BS, results_dir=results,
            checkpoint_dir=ckpt, mesh=self.mesh, buffer=None, save_best=save_best)
        return best, results, ckpt

    def test_save_best_false_writes_no_best_model(self):
        """save_best=False must suppress BOTH best_model files -- and nothing else."""
        best_on, res_on, ck_on = self._run_save_best(True, "on")
        best_off, res_off, ck_off = self._run_save_best(False, "off")

        for suffix in ("eqx", "json"):
            self.assertTrue(os.path.isfile(os.path.join(res_on, f"best_model.{suffix}")),
                            f"default (save_best=True) must still write best_model.{suffix}")
            self.assertFalse(os.path.exists(os.path.join(res_off, f"best_model.{suffix}")),
                             f"save_best=False must not write best_model.{suffix}")

        # everything else identical: the toggle gates a WRITE, not the computation
        self.assertTrue(math.isfinite(best_off), "best_val must still be tracked with save_best=False")
        self.assertEqual(best_on, best_off, "save_best must not change the tracked best_val")
        for r in (res_on, res_off):
            self.assertTrue(os.path.isfile(os.path.join(r, "metrics.json")))
        with open(os.path.join(res_on, "metrics.json")) as f:
            m_on = json.load(f)
        with open(os.path.join(res_off, "metrics.json")) as f:
            m_off = json.load(f)
        self.assertEqual(m_on, m_off, "save_best must not change the recorded metrics")

        # ...and the resumable state is byte-for-byte the same decision: the checkpoint's best_val_loss
        # (what a resume restores its best tracking from) must not move either. NOTE it is not asserted
        # equal to the RETURNED best: finish() skips the rewrite when the last maybe_write already
        # covered this step, so the recorded scalar can lag the final eval by one -- pre-existing, and
        # orthogonal to this toggle. What matters here is that save_best doesn't perturb it.
        metas = []
        for ck in (ck_on, ck_off):
            with open(os.path.join(ck, "meta.json")) as f:
                metas.append(json.load(f))
        self.assertTrue(math.isfinite(metas[1]["best_val_loss"]),
                        "best_val must still be tracked into the checkpoint meta with save_best=False")
        self.assertEqual(metas[0]["best_val_loss"], metas[1]["best_val_loss"],
                         "save_best must not change the checkpoint's recorded best_val_loss")
        self.assertEqual(metas[0]["step"], metas[1]["step"])

    # ---- loader selection helper ----
    def test_choose_loader_buffer_small_is_simple(self):
        # the synthetic train.bin is tiny (<< SIMPLE_MAX_GB) -> None (whole-in-RAM)
        self.assertIsNone(choose_loader_buffer(self.data_dir))

    def test_choose_loader_buffer_large_is_sliding(self):
        # force the threshold below the tiny file -> BUFFER_GB (SlidingLoader), without a 64 GB file
        orig = tc_mod.SIMPLE_MAX_GB
        tc_mod.SIMPLE_MAX_GB = 0.0
        try:
            self.assertEqual(choose_loader_buffer(self.data_dir), BUFFER_GB)
        finally:
            tc_mod.SIMPLE_MAX_GB = orig


if __name__ == "__main__":
    #unittest.main()
    a = TrainCoreTest()
    a.setUp()
    a.test_faulted_trajectory_is_chunk_invariant()