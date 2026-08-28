"""Unit tests for nano_llama.metrics -- the ONE definition of "this run's val loss".

Every consumer of results/metrics.json (worker._diverged, hp_opt's search objective, the transfer /
duration / rung-tpp analyses, the cluster monitors, the scaling-law loader) now reads through these
helpers, so the back-compat rule lives in exactly one place and is tested here rather than re-derived
per script. Stdlib-only, so this runs without jax.
"""
import json
import math
import os
import shutil
import tempfile
import unittest

from nano_llama.metrics import (val_loss_of, load_metrics, final_record,
                                final_val_loss, is_diverged)


class MetricsTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="metrics_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _write(self, obj):
        with open(os.path.join(self.dir, "metrics.json"), "w") as f:
            json.dump(obj, f)
        return self.dir

    # ---- which field carries the loss ----
    def test_reads_current_key(self):
        self.assertEqual(val_loss_of({"step": 1, "val_loss_fault": 3.25}), 3.25)

    def test_no_loss_field(self):
        self.assertIsNone(val_loss_of({"step": 1}))

    def test_final_is_by_max_step_not_file_order(self):
        recs = [{"step": 9, "val_loss_fault": 1.0}, {"step": 100, "val_loss_fault": 5.0},
                {"step": 50, "val_loss_fault": 2.0}]
        self.assertEqual(final_record(recs)["step"], 100)
        self.assertEqual(final_val_loss(recs), 5.0)          # FINAL, not best -- a blow-up must show

    # ---- divergence sentinel ----
    def test_diverged_on_non_finite_final(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            recs = [{"step": 1, "val_loss_fault": 2.0}, {"step": 2, "val_loss_fault": bad}]
            self.assertTrue(is_diverged(recs), f"{bad} must read as diverged")
            self.assertFalse(math.isfinite(final_val_loss(recs)))

    def test_healthy_and_missing_are_not_diverged(self):
        """Absent/empty metrics must read as a wall-clock stop, so a preempted run still resumes."""
        self.assertFalse(is_diverged([{"step": 2, "val_loss_fault": 2.0}]))
        self.assertFalse(is_diverged([]))
        self.assertFalse(is_diverged([{"step": 2}]))         # eval-less record -> no signal

    # ---- file loading is total ----
    def test_load_metrics_tolerates_missing_and_malformed(self):
        self.assertEqual(load_metrics(os.path.join(self.dir, "nope")), [])
        with open(os.path.join(self.dir, "metrics.json"), "w") as f:
            f.write("{not json")
        self.assertEqual(load_metrics(self.dir), [])
        self._write({"step": 1})                              # an object, not a list
        self.assertEqual(load_metrics(self.dir), [])
        self._write([{"step": 1, "val_loss_fault": 2.0}])
        self.assertEqual(final_val_loss(load_metrics(self.dir)), 2.0)


if __name__ == "__main__":
    unittest.main()
