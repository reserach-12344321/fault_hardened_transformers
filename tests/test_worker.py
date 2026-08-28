"""Local end-to-end test of the live cluster training worker.

Drives the real worker in-process on a tiny synthetic job, covering the fresh run,
preemption + resume, the already-complete no-op, and the sliding loader. The suite runs
TWICE -- once with a horizon on the eval grid and once off it -- with every expectation
derived from (MAX_ITERS, EVAL_INTERVAL), so both alignments are asserted by the same code.

Needs cluster_orchestrator on the path; skips cleanly if it is absent.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
import json
import math
import time
import shutil
import tempfile
import subprocess
import unittest
from unittest import mock

import numpy as np
import optax

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# cluster_orchestrator lives in a sibling checkout, not on scienv's path by default.
_CO = "/home/trevor/cluster_orchestrator"
if os.path.isdir(_CO) and _CO not in sys.path:
    sys.path.insert(0, _CO)
try:
    import orchestrator_hooks.worker as worker
    from cluster_orchestrator import worker_api
    _HAVE_CO = True
except ModuleNotFoundError:
    _HAVE_CO = False

import nano_llama.train_core as train_core
from tests._test_util import build_model_config
from nano_llama.train_core import TrainConfig
from nano_llama.fault import FaultConfig

VOCAB = 8192
BLOCK = 64
BATCH = 8


def expected_metric_steps(max_iters: int, eval_interval: int):
    """The steps at which the run MUST record a metric: every eval boundary strictly inside."""
    return list(range(eval_interval, max_iters, eval_interval)) + [max_iters]


class _WorkerFixture(unittest.TestCase):
    """The tiny synthetic job every test below drives: data bins, the 3 input configs + seed."""

    SEED = 0
    MAX_ITERS = 8            # ON the eval grid (8 % 2 == 0)
    EVAL_INTERVAL = 2

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="worker_")
        self.inputs = os.path.join(self.root, "inputs")     # read-only job spec
        self.results = os.path.join(self.root, "results")   # checkpoint/metrics/best_model/DONE
        self.data_dir = os.path.join(self.root, "data")
        os.makedirs(self.inputs); os.makedirs(self.data_dir)

        # --- data: tiny synthetic train/val bins ---
        rng = np.random.default_rng(0)
        for name, n in (("train", 200_000), ("val", 20_000)):
            rng.integers(0, VOCAB, size=n, dtype=np.uint16).tofile(os.path.join(self.data_dir, f"{name}.bin"))

        # --- inputs/: the 3 configs + meta.json (seed) the worker reads ---
        build_model_config(64, 2, block_size=BLOCK, dtype="float32", attn_impl="manual") \
            .save(os.path.join(self.inputs, "model_config.json"))
        self._write_train_config()
        FaultConfig(p=0.0, k=4).save(os.path.join(self.inputs, "fault_config.json"))
        with open(os.path.join(self.inputs, "meta.json"), "w") as f:
            json.dump({"seed": self.SEED}, f)

        # --- static-data JSON: the worker reads data_dir from its "data" key ---
        self.static_data_json = os.path.join(self.root, "static_data.json")
        with open(self.static_data_json, "w") as f:
            json.dump({"data": self.data_dir}, f)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    # ---- helpers ----
    def _write_train_config(self, **overrides):
        """Write inputs/train_config.json from the class's (MAX_ITERS, EVAL_INTERVAL)."""
        kw = dict(batch_size=BATCH, learning_rate=1e-3, max_iters=self.MAX_ITERS, warmup_iters=1,
                  lr_decay_iters=self.MAX_ITERS, eval_interval=self.EVAL_INTERVAL,
                  )
        kw.update(overrides)
        tc = TrainConfig(**kw)
        tc.save(os.path.join(self.inputs, "train_config.json"))
        return tc

    def _train_config(self):
        return TrainConfig.load(os.path.join(self.inputs, "train_config.json"))

    def _expected_metric_steps(self):
        return expected_metric_steps(self.MAX_ITERS, self.EVAL_INTERVAL)

    def _run_worker(self, max_seconds):
        argv = ["worker.py", self.inputs, self.results, self.static_data_json, str(max_seconds)]
        with mock.patch.object(sys, "argv", argv):
            worker.main()

    def _done(self):
        return worker_api.is_done(self.results)

    def _metric_steps(self):
        with open(os.path.join(self.results, "metrics.json")) as f:
            return [m["step"] for m in json.load(f)]

    def _ckpt_step(self):
        with open(os.path.join(self.results, "checkpoint", "meta.json")) as f:
            return json.load(f)["step"]

    def _lr_schedule(self, tc):
        """The peak-LR schedule build_optimizer hands to the optimizer."""
        h = {"learning_rate": tc.learning_rate, "min_learning_rate": tc.min_lr}
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=h["learning_rate"], warmup_steps=tc.warmup_iters,
            decay_steps=tc.lr_decay_iters, end_value=h["min_learning_rate"]), h


@unittest.skipUnless(_HAVE_CO, "cluster_orchestrator not importable (needed by orchestrator_hooks.worker)")
class WorkerE2ETest(_WorkerFixture):

    # ---- the horizon contract: what a DONE marker is allowed to mean ----
    def test_done_implies_the_full_horizon_was_trained(self):
        """DONE must mean max_iters steps were TAKEN -- not 'as many as fit under some."""
        self._run_worker(0)
        self.assertTrue(self._done(), "DONE marker not written")
        self.assertEqual(self._ckpt_step(), self.MAX_ITERS,
                         "a run marked DONE must have taken every one of its max_iters steps")
        self.assertEqual(self._metric_steps()[-1], self.MAX_ITERS,
                         "the final metric -- the number the scaling-law fit consumes -- must be at max_iters")

    def test_done_implies_the_lr_schedule_ran_to_its_end(self):
        """The consequence that makes a truncated run scientifically WRONG rather than merely."""
        tc = self._train_config()
        self._run_worker(0)
        final = self._ckpt_step()
        self.assertEqual(final, tc.lr_decay_iters,
                         "the run must end exactly where the LR schedule ends (lr_decay_iters)")
        sched, h = self._lr_schedule(tc)
        self.assertAlmostEqual(float(sched(final)), h["min_learning_rate"], places=12,
                               msg="LR at the final step must be min_lr (schedule fully annealed)")
        self.assertGreater(float(sched(final - 1)), float(sched(final)),
                           "sanity: the LR must still be decaying one step earlier, so stopping short "
                           "of max_iters would leave the schedule unannealed and this test would fire")

    # ---- lifecycle ----
    def test_fresh_run_to_completion(self):
        self._run_worker(0)                                   # 0 -> no wall limit
        self.assertTrue(self._done(), "DONE marker not written")
        self.assertEqual(self._metric_steps(), self._expected_metric_steps())
        self.assertEqual(self._ckpt_step(), self.MAX_ITERS)
        self.assertTrue(os.path.isfile(os.path.join(self.results, "best_model.eqx")))
        self.assertTrue(os.path.isfile(os.path.join(self.results, "best_model.json")))

    def _launch_worker_subprocess(self):
        """Start the real worker as a killable subprocess (no wall limit)."""
        env = {**os.environ, "JAX_PLATFORMS": "cpu",
               "PYTHONPATH": os.pathsep.join([REPO, _CO])}
        self._stderr = open(os.path.join(self.root, "worker.stderr"), "w+")
        return subprocess.Popen(
            [sys.executable, os.path.join(REPO, "orchestrator_hooks", "worker.py"),
             self.inputs, self.results, self.static_data_json, "0"],
            env=env, stdout=subprocess.DEVNULL, stderr=self._stderr)

    def _poll_until_trained_block(self, proc, min_step=2, deadline_s=180):
        """Poll the worker's checkpoint until it records a TRAINED block (step >= min_step)."""
        meta = os.path.join(self.results, "checkpoint", "meta.json")
        end = time.time() + deadline_s
        while time.time() < end:
            if os.path.isfile(meta):
                try:
                    with open(meta) as f:
                        step = json.load(f).get("step", 0)
                except (json.JSONDecodeError, OSError):
                    step = 0                                  # mid-write; try again next poll
                if step >= min_step:
                    return step
            if proc.poll() is not None:                       # worker died before making progress
                self._stderr.seek(0)
                self.fail(f"worker exited early (rc={proc.returncode}):\n{self._stderr.read()[-2000:]}")
            time.sleep(0.05)
        self.fail(f"worker did not checkpoint step>={min_step} within {deadline_s}s (failsafe)")

    def test_preemption_then_resume(self):
        # Deterministic preemption: run the worker, poll until it has actually TRAINED a block, then
        # hard-kill it -- no magic wall-time. Then resume in-process and require clean completion.
        proc = self._launch_worker_subprocess()
        try:
            partial = self._poll_until_trained_block(proc, min_step=self.EVAL_INTERVAL)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
            self._stderr.close()

        self.assertLess(partial, self.MAX_ITERS, "should have been killed before reaching the target")
        self.assertFalse(self._done(), "a mid-run kill must not leave a DONE marker")

        self._run_worker(0)                                   # resume from the partial checkpoint
        self.assertTrue(self._done())
        # A resume must reach the SAME horizon an uninterrupted run does -- the truncation bug's other
        # face is a resumed segment that rounds its remaining steps down and stalls short forever.
        self.assertEqual(self._ckpt_step(), self.MAX_ITERS)
        self.assertEqual(self._metric_steps(), self._expected_metric_steps(),
                         "resume must extend metrics without gaps or duplicates")

    def test_rerun_at_completion_is_idempotent(self):
        # A re-run at start==target no longer short-circuits: it re-runs the (deterministic, 0-step)
        # final eval. metrics.json must be UNCHANGED -- the loop reloads metrics filtered to step<target,
        # so the re-eval overwrites the target row rather than duplicating it.
        self._run_worker(0)
        self.assertTrue(self._done())
        before = self._metric_steps()
        self._run_worker(0)                                   # start >= target -> re-eval, no retrain
        self.assertTrue(self._done())
        self.assertEqual(self._metric_steps(), before, "a completed re-run must not duplicate / drop metrics")

    def test_resume_at_target_without_metrics_writes_final_eval(self):
        # THE BUG this guards against: a run preempted after its last step but BEFORE the end-only eval
        # wrote metrics.json resumes with start == target. The old `start>=target -> mark_done` guard
        # then skipped the eval, leaving NO metrics.json -> the driver harvested raw None -> a healthy
        # run mislabeled INFEASIBLE. Now the worker must run the final eval and record a finite loss.
        self._run_worker(0)                                   # full run -> checkpoint at target + metrics
        self.assertEqual(self._ckpt_step(), self.MAX_ITERS)
        os.remove(os.path.join(self.results, "metrics.json"))  # simulate: eval never persisted pre-preempt

        self._run_worker(0)                                   # resume at start==target (worker doesn't gate on DONE)
        self.assertTrue(self._done(), "resume-at-target must still mark done")
        self.assertTrue(os.path.isfile(os.path.join(self.results, "metrics.json")),
                        "the final eval must have written metrics.json (not skipped)")
        with open(os.path.join(self.results, "metrics.json")) as f:
            metrics = json.load(f)
        last = max(metrics, key=lambda r: r["step"])
        self.assertEqual(last["step"], self.MAX_ITERS, "final metric must be at the target step")
        self.assertTrue(math.isfinite(last["val_loss_fault"]),
                        "resumed final eval must record a finite loss (NOT harvest as raw None)")

    def test_divergence_marks_done_before_target(self):
        # An absurd LR NaNs the loss within a couple of steps. The worker must mark the job DONE even
        # though it stopped before target (resuming would only reproduce the divergence), and the final
        # recorded val loss must be non-finite (the divergence signal for the driver).
        self._write_train_config(learning_rate=1e10)
        self._run_worker(0)
        self.assertTrue(self._done(), "a diverged run must be marked done (not left to resume forever)")
        with open(os.path.join(self.results, "metrics.json")) as f:
            metrics = json.load(f)
        last = max(metrics, key=lambda r: r["step"])
        self.assertLess(last["step"], self.MAX_ITERS, "must have stopped before the target")
        self.assertFalse(math.isfinite(last["val_loss_fault"]),
                         "final recorded val loss must be non-finite on divergence")

    def test_static_save_best_false_suppresses_best_model(self):
        # The static-data map's "save_best" is the deployment switch the full sweep uses to stop each
        # run leaving a second full copy of its weights on disk. It must suppress best_model and NOTHING
        # else: same DONE, same metric steps, same final checkpoint step, best_val still recorded.
        with open(self.static_data_json, "w") as f:
            json.dump({"data": self.data_dir, "save_best": False}, f)
        self._run_worker(0)

        self.assertTrue(self._done())
        self.assertEqual(self._metric_steps(), self._expected_metric_steps())
        self.assertEqual(self._ckpt_step(), self.MAX_ITERS)
        for suffix in ("eqx", "json"):
            self.assertFalse(os.path.exists(os.path.join(self.results, f"best_model.{suffix}")),
                             f"save_best=False must not write best_model.{suffix}")
        with open(os.path.join(self.results, "checkpoint", "meta.json")) as f:
            self.assertTrue(math.isfinite(json.load(f)["best_val_loss"]),
                            "best_val must still be tracked into the checkpoint meta")

    def test_static_save_best_defaults_to_on(self):
        # Absent key -> unchanged behavior. Guards the default: an old static-data map (every launcher
        # writes one without "save_best") must keep writing best_model exactly as before.
        with open(self.static_data_json) as f:
            self.assertNotIn("save_best", json.load(f))
        self._run_worker(0)
        self.assertTrue(os.path.isfile(os.path.join(self.results, "best_model.eqx")))
        self.assertTrue(os.path.isfile(os.path.join(self.results, "best_model.json")))

    def test_sliding_loader_path(self):
        # Force SlidingLoader on the tiny dataset by dropping the size threshold the worker consults.
        with mock.patch.object(train_core, "SIMPLE_MAX_GB", 0.0):
            self._run_worker(0)
        self.assertTrue(self._done())
        self.assertEqual(self._metric_steps(), self._expected_metric_steps())


@unittest.skipUnless(_HAVE_CO, "cluster_orchestrator not importable (needed by orchestrator_hooks.worker)")
class OffGridWorkerE2ETest(WorkerE2ETest):
    """THE REGRESSION FIXTURE: the entire suite above."""

    MAX_ITERS = 7            # 7 % 2 == 1 -> the last eval boundary (6) is NOT the horizon
    EVAL_INTERVAL = 2

    def setUp(self):
        assert self.MAX_ITERS % self.EVAL_INTERVAL != 0, (
            "this fixture is only meaningful with a horizon OFF the eval grid -- do not 'fix' it by "
            "snapping MAX_ITERS, that is precisely the condition the bug hid behind")
        super().setUp()

    def test_final_metric_is_recorded_off_the_eval_grid(self):
        """The end-of-run eval is NOT on the eval cadence, and must still be recorded."""
        self._run_worker(0)
        steps = self._metric_steps()
        self.assertEqual(steps, self._expected_metric_steps())
        self.assertNotEqual(steps[-1] % self.EVAL_INTERVAL, 0,
                            "the fixture must actually exercise an off-grid final step")


@unittest.skipUnless(_HAVE_CO, "cluster_orchestrator not importable (needed by orchestrator_hooks.worker)")
class ShortHorizonTest(_WorkerFixture):
    """The DEGENERATE limit of the truncation bug: a horizon SHORTER than one eval interval."""

    MAX_ITERS = 5
    EVAL_INTERVAL = 8        # > MAX_ITERS -> the old expression floors to 0

    def setUp(self):
        assert self.EVAL_INTERVAL > self.MAX_ITERS, "fixture must keep the horizon under one eval interval"
        super().setUp()

    def test_horizon_shorter_than_one_eval_interval_still_trains(self):
        self._run_worker(0)
        self.assertTrue(self._done())
        self.assertEqual(self._ckpt_step(), self.MAX_ITERS,
                         "a horizon under one eval interval must still be TRAINED, not floored to 0 steps")
        self.assertEqual(self._metric_steps(), [self.MAX_ITERS],
                         "exactly one metric, at the end of the run (no interior eval boundary exists)")
        # Independent evidence that optimizer steps were actually taken -- metrics alone cannot tell an
        # untrained model from a trained one, which is what made the 0-step version of this bug invisible.
        with open(os.path.join(self.results, train_core.TRAIN_LOG)) as f:
            logged = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(sum(r["n_steps"] for r in logged), self.MAX_ITERS,
                         "train_loss.jsonl must account for every step of the horizon")


@unittest.skipUnless(_HAVE_CO, "cluster_orchestrator not importable (needed by orchestrator_hooks.worker)")
class ShortResumeSpanTest(_WorkerFixture):
    """A RESUME whose remaining span is shorter than one eval interval."""

    MAX_ITERS = 6            # trained first, on the grid
    EVAL_INTERVAL = 2
    EXTENDED = 7             # then resumed with 1 step left: 1 < EVAL_INTERVAL

    def test_resume_span_shorter_than_one_eval_interval_completes(self):
        self._run_worker(0)
        self.assertEqual(self._ckpt_step(), self.MAX_ITERS)

        tc = self._write_train_config(max_iters=self.EXTENDED, lr_decay_iters=self.EXTENDED)
        self._run_worker(0)                                    # start = 6, remaining = 1 < eval_interval

        self.assertTrue(self._done())
        self.assertEqual(self._ckpt_step(), self.EXTENDED,
                         "a resume with fewer than eval_interval steps left must still finish the horizon")
        self.assertEqual(self._metric_steps(), [2, 4, 6, self.EXTENDED])
        sched, h = self._lr_schedule(tc)
        self.assertAlmostEqual(float(sched(self._ckpt_step())), h["min_learning_rate"], places=12,
                               msg="the resumed run must land on the END of the extended cosine")


if __name__ == "__main__":
    unittest.main()
