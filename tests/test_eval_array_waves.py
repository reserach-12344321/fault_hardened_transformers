"""Test the INCREMENTAL (wave-based) staging in experiment_util/prepare_eval_array.py.

The property under test is the one the whole scheme exists for: you can prepare an eval array from a
sweep that is still training, come back when more of it has finished, and stage ONLY the new models --
without re-running anything already staged, and without the second wave's seeds colliding with the
first's.

Builds a fake training sweep on disk (configs + a stub checkpoint + DONE markers -- no jax model, no
real weights) and drives prepare_eval_array's own functions against it.

    JAX_PLATFORMS=cpu PYTHONPATH=<nano_llama> /home/trevor/scienv/bin/python -m unittest tests.test_eval_array_waves -v
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")   # never fight live training on the shared GPU

import json
import shutil
import tempfile
import unittest

from cluster_orchestrator import worker_api

from nano_llama.fault import FaultConfig
from nano_llama.fault_eval import EvalConfig
from nano_llama.llama import LlamaConfig
from nano_llama.train_core import TrainConfig

from experiment_util import prepare_eval_array as pea, process_eval_arrays as pae

EXTRAS = [(4, 0.001), (4, 0.01), (4, 0.1)]
SPEC_KW = dict(target_se=0.005, min_evals=8, max_evals=1024, batch_size=None, seed=2323)


def _spec(extras=EXTRAS, **over):
    return pea.build_spec(extras, **{**SPEC_KW, **over})


def _train_config():
    return TrainConfig(batch_size=8, learning_rate=1e-3, max_iters=100, warmup_iters=10,
                       lr_decay_iters=100, min_lr=1e-4, eval_interval=50,
                       weight_decay=1e-3, beta1=0.9, beta2=0.95, adam_eps=1e-8,
                       init_std=0.02, grad_clip=10.0,
                       divergence_loss_factor=1.5)


def _model_config():
    return LlamaConfig(block_size=16, vocab_size=32, n_layer=1, n_head=1, n_embd=8, multiple_of=4,
                       dtype="float32", attn_impl="manual", tie_embeddings=True)


def _write_gridpoint(sweep_dir, name, *, p_train=0.0, done=True, val_loss=3.0, checkpoint=True):
    """One gridpoint of a fake training sweep."""
    gp = os.path.join(sweep_dir, name)
    results = os.path.join(gp, "results")
    ckpt = os.path.join(results, "checkpoint")
    os.makedirs(ckpt, exist_ok=True)
    _model_config().save(os.path.join(gp, "model_config.json"))
    _train_config().save(os.path.join(gp, "train_config.json"))
    FaultConfig(p=p_train, k=4).save(os.path.join(gp, "fault_config.json"))
    if checkpoint:
        with open(os.path.join(ckpt, "model.eqx"), "wb") as fh:      # stub weights; never deserialized
            fh.write(b"\0" * 32)
        with open(os.path.join(ckpt, "meta.json"), "w") as fh:
            json.dump({"step": 100, "best_val_loss": val_loss, "key": None}, fh)
    with open(os.path.join(results, "metrics.json"), "w") as fh:            # metrics.VAL_LOSS_KEYS
        json.dump([{"step": 100, "val_loss_fault": val_loss, "train_loss": val_loss}], fh)
    if done:
        worker_api.mark_done(results)
    return gp


def _finish_job(job_dir, *, points=None):
    """Pretend the eval worker ran this staged job: write a results file and mark it DONE."""
    results = os.path.join(job_dir, "results")
    os.makedirs(results, exist_ok=True)
    ec = EvalConfig.load(os.path.join(job_dir, "eval_config.json"))
    fc = FaultConfig.load(os.path.join(job_dir, "fault_config.json"))
    pts = points if points is not None else [
        {"k": k, "p": p, "loss": 3.0 + p, "se": 0.004, "n": 128, "reached_target": True}
        for k, p in ec.kp_pairs]
    with open(os.path.join(results, "eval_results.json"), "w") as fh:
        json.dump({"n_params": 1000, "n_train_tokens": 12800, "final_step": 100,
                   "k_train": fc.k, "p_train": fc.p, "target_se": ec.target_se,
                   "min_evals": ec.min_evals, "max_evals": ec.max_evals, "batch_size": 64,
                   "points": pts}, fh)
    worker_api.mark_done(results)


class WaveStagingTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.sweep = os.path.join(self.tmp, "sweep_2026-07-28")
        self.root = os.path.join(self.tmp, "eval_experiment")
        os.makedirs(self.sweep)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _stage(self, spec=None, sweep_dirs=None):
        """One run of the script's logic: scan, subtract what is staged, stage the rest."""
        spec = spec or _spec()
        sweeps = sweep_dirs or [self.sweep]
        spec, _, _ = pea.load_or_create_spec(self.root, spec, sweep_dirs=sweeps)
        candidates, stats = pea.scan_candidates(sweeps)
        staged = pea.staged_job_names(self.root)
        new = [c for c in candidates if c.job_name not in staged]
        wave = pea.stage_wave(self.root, new, spec, sweep_dirs=sweeps, hardlink=False) if new else None
        return wave, new, stats

    # ---- the core incremental property -------------------------------------------------------

    def test_second_wave_stages_only_newly_finished_runs(self):
        _write_gridpoint(self.sweep, "d64_L2_p0.0", done=True)
        _write_gridpoint(self.sweep, "d64_L2_p0.04", p_train=0.04, done=True)
        still_training = _write_gridpoint(self.sweep, "d128_L4_p0.0", done=False)

        wave1, new1, _ = self._stage()
        self.assertEqual(sorted(c.job_name for c in new1),
                         ["sweep_2026-07-28__d64_L2_p0.0", "sweep_2026-07-28__d64_L2_p0.04"])
        self.assertEqual(len(pea.wave_jobs(wave1)), 2)

        # nothing has changed -> no second wave at all
        wave_none, new_none, _ = self._stage()
        self.assertIsNone(wave_none)
        self.assertEqual(new_none, [])
        self.assertEqual(len(pea.wave_dirs(self.root)), 1)

        # the third run finishes training (even while wave 1 is still unfinished) -> exactly one new job
        worker_api.mark_done(os.path.join(still_training, "results"))
        wave2, new2, _ = self._stage()
        self.assertEqual([c.job_name for c in new2], ["sweep_2026-07-28__d128_L4_p0.0"])
        self.assertEqual(pea.wave_jobs(wave2), ["sweep_2026-07-28__d128_L4_p0.0"])
        self.assertEqual(len(pea.wave_dirs(self.root)), 2)
        # and wave 1's jobs are untouched by the second staging
        self.assertEqual(len(pea.wave_jobs(wave1)), 2)

    def test_finished_jobs_are_never_restaged(self):
        """A model whose eval has RUN stays staged: results must not be recomputed."""
        _write_gridpoint(self.sweep, "d64_L2_p0.0")
        wave1, _, _ = self._stage()
        _finish_job(os.path.join(wave1, "sweep_2026-07-28__d64_L2_p0.0"))

        _write_gridpoint(self.sweep, "d64_L2_p0.04", p_train=0.04)
        wave2, new2, _ = self._stage()
        self.assertEqual([c.job_name for c in new2], ["sweep_2026-07-28__d64_L2_p0.04"])
        self.assertTrue(worker_api.is_done(
            os.path.join(wave1, "sweep_2026-07-28__d64_L2_p0.0", "results")))

    # ---- staged job contents -----------------------------------------------------------------

    def test_staged_job_has_the_inputs_the_eval_worker_expects(self):
        _write_gridpoint(self.sweep, "d64_L2_p0.04", p_train=0.04)
        wave, _, _ = self._stage()
        job = os.path.join(wave, "sweep_2026-07-28__d64_L2_p0.04")
        for fname in ("final_model.eqx", "final_model.json", "checkpoint_meta.json",
                      "train_config.json", "fault_config.json", "eval_config.json"):
            self.assertTrue(os.path.isfile(os.path.join(job, fname)), fname)
        ec = EvalConfig.load(os.path.join(job, "eval_config.json"))
        # own two baselines first, then the shared extras (the 0.001/0.01/0.1 grid)
        self.assertEqual(ec.kp_pairs, ((4, 0.0), (4, 0.04), (4, 0.001), (4, 0.01), (4, 0.1)))
        self.assertEqual((ec.target_se, ec.min_evals, ec.max_evals), (0.005, 8, 1024))
        self.assertIsNone(ec.batch_size)

    def test_clean_run_collapses_its_two_baselines(self):
        _write_gridpoint(self.sweep, "d64_L2_p0.0", p_train=0.0)
        wave, _, _ = self._stage()
        ec = EvalConfig.load(os.path.join(wave, "sweep_2026-07-28__d64_L2_p0.0", "eval_config.json"))
        self.assertEqual(ec.kp_pairs, ((4, 0.0), (4, 0.001), (4, 0.01), (4, 0.1)))

    # ---- seeds -------------------------------------------------------------------------------

    def test_seeds_are_name_derived_and_distinct_across_waves(self):
        """The old ``base + running index`` restarted at 0 in each wave."""
        _write_gridpoint(self.sweep, "a_p0.0")
        wave1, _, _ = self._stage()
        _write_gridpoint(self.sweep, "b_p0.0")
        wave2, _, _ = self._stage()

        seed1 = EvalConfig.load(os.path.join(wave1, "sweep_2026-07-28__a_p0.0",
                                             "eval_config.json")).seed
        seed2 = EvalConfig.load(os.path.join(wave2, "sweep_2026-07-28__b_p0.0",
                                             "eval_config.json")).seed
        self.assertNotEqual(seed1, seed2)
        # and each is exactly the name-derived value, i.e. independent of wave and of ordering
        self.assertEqual(seed1, pea.job_seed(SPEC_KW["seed"], "sweep_2026-07-28__a_p0.0"))
        self.assertEqual(seed2, pea.job_seed(SPEC_KW["seed"], "sweep_2026-07-28__b_p0.0"))

    def test_job_seed_is_stable_across_processes_and_tracks_the_base(self):
        self.assertEqual(pea.job_seed(7, "sweep__x"), pea.job_seed(7, "sweep__x"))   # not salted
        self.assertNotEqual(pea.job_seed(7, "sweep__x"), pea.job_seed(8, "sweep__x"))
        self.assertLess(pea.job_seed(7, "sweep__x"), 2 ** 31)

    # ---- divergence, on every wave -----------------------------------------------------------

    def test_diverged_runs_stay_excluded_and_keep_being_reported(self):
        _write_gridpoint(self.sweep, "d64_L2_p0.0")
        _write_gridpoint(self.sweep, "d64_L2_blown", val_loss=float("nan"))   # DONE but diverged
        wave1, new1, stats1 = self._stage()
        self.assertEqual([c.job_name for c in new1], ["sweep_2026-07-28__d64_L2_p0.0"])
        self.assertEqual(stats1[0].diverged, ["sweep_2026-07-28__d64_L2_blown"])

        # It is DONE, so it is a candidate forever: the check has to run again, and report again.
        _, new2, stats2 = self._stage()
        self.assertEqual(new2, [])
        self.assertEqual(stats2[0].diverged, ["sweep_2026-07-28__d64_L2_blown"])
        self.assertNotIn("sweep_2026-07-28__d64_L2_blown", pea.staged_job_names(self.root))

    def test_incomplete_checkpoint_is_not_a_candidate(self):
        _write_gridpoint(self.sweep, "no_ckpt", done=True, checkpoint=False)
        _, new, stats = self._stage()
        self.assertEqual(new, [])
        self.assertEqual(stats[0].n_incomplete, 1)

    # ---- the spec guard ----------------------------------------------------------------------

    def test_changed_spec_aborts_but_can_be_forced(self):
        _write_gridpoint(self.sweep, "d64_L2_p0.0")
        self._stage()
        with self.assertRaises(SystemExit):
            pea.load_or_create_spec(self.root, _spec(target_se=0.001), sweep_dirs=[self.sweep])
        spec, created, changes = pea.load_or_create_spec(self.root, _spec(target_se=0.001),
                                                         sweep_dirs=[self.sweep], allow_change=True)
        self.assertFalse(created)
        self.assertEqual([c[0] for c in changes], ["target_se"])
        with open(os.path.join(self.root, pea.SPEC_FILE)) as fh:
            doc = json.load(fh)
        self.assertEqual(doc["spec"]["target_se"], 0.001)
        self.assertEqual(doc["superseded"][0]["spec"]["target_se"], 0.005)

    def test_spec_survives_a_json_round_trip_unchanged(self):
        """The drift guard is a plain ==, so a spec read back from disk must compare equal to a fresh
        one -- otherwise every second run aborts on phantom drift.
        """
        _write_gridpoint(self.sweep, "d64_L2_p0.0")
        self._stage()
        _, created, changes = pea.load_or_create_spec(self.root, _spec(), sweep_dirs=[self.sweep])
        self.assertFalse(created)
        self.assertEqual(changes, [])

    def test_sweep_dirs_are_accumulated_not_guarded(self):
        """Adding a newly-finished sweep (or moving one between mounts) is normal, not spec drift."""
        _write_gridpoint(self.sweep, "d64_L2_p0.0")
        self._stage()
        other = os.path.join(self.tmp, "sweep_2026-07-29")
        os.makedirs(other)
        _write_gridpoint(other, "d64_L2_p0.0")
        wave2, new2, _ = self._stage(sweep_dirs=[self.sweep, other])
        self.assertEqual([c.job_name for c in new2], ["sweep_2026-07-29__d64_L2_p0.0"])
        with open(os.path.join(self.root, pea.SPEC_FILE)) as fh:
            self.assertEqual(len(json.load(fh)["sweep_dirs"]), 2)

    # ---- wave bookkeeping --------------------------------------------------------------------

    def test_colliding_job_names_abort(self):
        """Two sweep dirs sharing a basename would map two runs onto one job name."""
        twin_parent = os.path.join(self.tmp, "elsewhere")
        twin = os.path.join(twin_parent, os.path.basename(self.sweep))
        os.makedirs(twin)
        _write_gridpoint(self.sweep, "d64_L2_p0.0")
        _write_gridpoint(twin, "d64_L2_p0.0")
        with self.assertRaises(SystemExit):
            pea.scan_candidates([self.sweep, twin])

    def test_stray_dirs_are_not_mistaken_for_waves(self):
        _write_gridpoint(self.sweep, "d64_L2_p0.0")
        wave1, _, _ = self._stage()
        os.makedirs(os.path.join(self.root, ".staging-interrupted", "some_job"))   # a crashed run
        os.makedirs(os.path.join(self.root, "notes"))
        self.assertEqual(pea.wave_dirs(self.root), [wave1])
        self.assertEqual(list(pea.staged_job_names(self.root)), ["sweep_2026-07-28__d64_L2_p0.0"])

    def test_wave_progress_counts_done_jobs(self):
        _write_gridpoint(self.sweep, "a_p0.0")
        _write_gridpoint(self.sweep, "b_p0.0")
        wave, _, _ = self._stage()
        self.assertEqual(pea.wave_progress(wave), (0, 2))
        _finish_job(os.path.join(wave, "sweep_2026-07-28__a_p0.0"))
        self.assertEqual(pea.wave_progress(wave), (1, 2))


class WaveDownstreamTest(unittest.TestCase):
    """Waves must remain ordinary array dirs: process_eval_arrays consumes them unchanged."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.sweep = os.path.join(self.tmp, "sweep_2026-07-28")
        self.root = os.path.join(self.tmp, "eval_experiment")
        os.makedirs(self.sweep)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_materialize_over_all_waves(self):
        spec, _, _ = pea.load_or_create_spec(self.root, _spec(), sweep_dirs=[self.sweep])
        _write_gridpoint(self.sweep, "a_p0.0")
        cands, _ = pea.scan_candidates([self.sweep])
        wave1 = pea.stage_wave(self.root, cands, spec, sweep_dirs=[self.sweep], hardlink=False)
        _finish_job(os.path.join(wave1, "sweep_2026-07-28__a_p0.0"))

        _write_gridpoint(self.sweep, "b_p0.0")
        cands, _ = pea.scan_candidates([self.sweep])
        staged = pea.staged_job_names(self.root)
        wave2 = pea.stage_wave(self.root, [c for c in cands if c.job_name not in staged], spec,
                               sweep_dirs=[self.sweep], hardlink=False)
        _finish_job(os.path.join(wave2, "sweep_2026-07-28__b_p0.0"))

        out = os.path.join(self.tmp, "summaries")
        written, skipped = pae.materialize(pae.eval_array_dirs(self.root), out)
        self.assertEqual((written, skipped), (2, 0))
        # FLAT: both waves land in one dir as <job_name>.json, so the campaign is a single loadable
        # population whose file count IS the model count (materialize aborts on a name collision).
        self.assertEqual(sorted(os.listdir(out)),
                         ["sweep_2026-07-28__a_p0.0.json", "sweep_2026-07-28__b_p0.0.json"])
        # eval_array_dirs must have found the waves under the root -- passing the root itself is the
        # silent-empty-output mistake load_array now refuses.
        self.assertEqual(pae.eval_array_dirs(self.root), pea.wave_dirs(self.root))
        with self.assertRaises(SystemExit) as cm:
            pae.load_array(self.root)
        self.assertIn("experiment ROOT", str(cm.exception))

        from scaling_law.data_loading import load_eval_results
        cohorts = load_eval_results(out)
        self.assertEqual(len(cohorts[(4, 0.1)]), 2)     # both models land in the shared cohort


if __name__ == "__main__":
    unittest.main()
