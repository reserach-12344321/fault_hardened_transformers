"""Offline tests for data_loading.filter_cohorts, the uniform run screen.

Synthetic EvalPoints built in memory, so this exercises the criteria themselves: what each one
drops, that a run must pass all of them, that `by_reason` blames the FIRST failure, that list
order (and so index alignment with eval_cohort_arrays) survives, and that an emptied cohort
raises rather than silently losing a rung.

    JAX_PLATFORMS=cpu PYTHONPATH=<nano_llama> /home/trevor/scienv/bin/python -m unittest tests.test_filter_cohorts -v
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import math
import unittest

from nano_llama.eval_result import EvalResult
from nano_llama.llama import LlamaConfig
from nano_llama.train_core import TrainConfig
from scaling_law import data_loading as dl

BLOCK, BATCH = 1024, 128
VOCAB = 8192                      # ln(8192) = 9.01, the sweep's ceiling reference


def _point(*, max_iters=10000, final_step=None, loss=3.0,
           n_params=10_000_000, n_ne=8_000_000):
    """One EvalPoint with the knobs the screens read."""
    final_step = max_iters if final_step is None else final_step
    tc = TrainConfig(batch_size=BATCH, max_iters=max_iters)
    mc = LlamaConfig(block_size=BLOCK, vocab_size=VOCAB, n_layer=2, n_head=2, n_embd=64)
    return EvalResult(model_config=mc, train_config=tc, k_train=4.0, p_train=0.0,
                      k_eval=(4.0,), p_eval=(0.0,), eval_loss=(loss,),
                      total_n_params=n_params, n_non_embedding_params=n_ne,
                      n_train_tokens=final_step * BATCH * BLOCK,
                      eval_se=(0.005,), n_eval_seq=(4096,),
                      eval_reached_se_target=(True,)).points()[0]


class TestScreens(unittest.TestCase):
    """Cross-cutting behaviour: the ceiling screen, combination, blame order, ordering, emptying."""

    def _run(self, points, **kw):
        kept, report = dl.filter_cohorts({0.0: list(points)}, **kw)
        return kept[0.0], report[0.0]

    def test_ceiling_screen(self):
        """The default criterion: drop runs at or above max_ceiling_frac * ln(vocab)."""
        pts = [_point(loss=0.70 * math.log(VOCAB) + 0.1), _point(loss=3.0)]
        kept, rep = self._run(pts, max_ceiling_frac=0.70)
        self.assertEqual([p.eval_loss for p in kept], [3.0])
        self.assertEqual(rep["by_reason"], {"ceiling": 1})
        self.assertAlmostEqual(rep["L_max_after"], 3.0)

    def test_screens_are_off_by_default_except_the_ceiling(self):
        """min_tpp / min_ne_frac default to None."""
        kept, rep = self._run([_point(loss=3.0), _point(loss=4.0)], max_ceiling_frac=None)
        self.assertEqual(len(kept), 2)
        self.assertEqual(rep["n_dropped"], 0)

    def test_a_run_must_pass_every_active_criterion(self):
        thin = dict(n_params=1_155_392, n_ne=106_816)          # ne_frac 0.092
        fat = dict(n_params=10_621_184, n_ne=6_426_880)        # ne_frac 0.605
        pts = [_point(loss=3.0, **thin), _point(loss=3.0, **fat)]
        kept, _ = self._run(pts, max_ceiling_frac=0.70, min_ne_frac=0.15)
        self.assertEqual(len(kept), 1)

    def test_by_reason_blames_the_first_failure(self):
        """A run failing several criteria is counted ONCE, under the first checked (ceiling."""
        bad = _point(loss=8.0, n_params=1_155_392, n_ne=106_816)   # over ceiling AND mostly embedding
        _, rep = self._run([bad, _point(loss=3.0)], max_ceiling_frac=0.70, min_ne_frac=0.15)
        self.assertEqual(rep["by_reason"], {"ceiling": 1})
        self.assertEqual(sum(rep["by_reason"].values()), rep["n_dropped"])

    def test_order_is_preserved_so_arrays_stay_aligned(self):
        pts = [_point(loss=3.3), _point(loss=8.0), _point(loss=3.2)]
        kept, _ = self._run(pts, max_ceiling_frac=0.70)
        self.assertEqual([p.eval_loss for p in kept], [3.3, 3.2])

    def test_emptying_a_cohort_raises(self):
        with self.assertRaises(ValueError) as cm:
            self._run([_point(loss=8.0)], max_ceiling_frac=0.70)
        self.assertIn("max_ceiling_frac", str(cm.exception))

    def test_rejects_out_of_range_thresholds(self):
        pts = [_point()]
        with self.assertRaises(ValueError):
            self._run(pts, max_ceiling_frac=70)       # a percentage, not a fraction of ln(vocab)
        with self.assertRaises(ValueError):
            self._run(pts, max_ceiling_frac=0.0)
        with self.assertRaises(ValueError):
            self._run(pts, min_tpp=0.0)               # a floor of 0 would be a no-op, not a screen


class TestMinTpp(unittest.TestCase):
    """Tokens per TOTAL parameter."""
    def _run(self, points, **kw):
        kept, report = dl.filter_cohorts({0.0: list(points)}, **kw)
        return kept[0.0], report[0.0]

    @staticmethod
    def _at_tpp(tpp, *, steps=100, loss=3.0):
        """A point whose tokens/total-param is exactly `tpp`."""
        tokens = steps * BATCH * BLOCK
        n_params = int(tokens / tpp)
        return _point(max_iters=steps, loss=loss, n_params=n_params, n_ne=int(0.8 * n_params))

    def test_drops_undertrained_runs(self):
        pts = [self._at_tpp(1, loss=5.2), self._at_tpp(10, loss=3.3), self._at_tpp(20, loss=3.1)]
        kept, rep = self._run(pts, max_ceiling_frac=None, min_tpp=5)
        self.assertEqual([p.eval_loss for p in kept], [3.3, 3.1])
        self.assertEqual(rep["by_reason"], {"min_tpp": 1})

    def test_is_measured_against_total_params_not_non_embedding(self):
        """The distinguishing property."""
        pt = _point(max_iters=100, n_params=(100 * BATCH * BLOCK) // 4,
                    n_ne=(100 * BATCH * BLOCK) // 20)
        kept, rep = self._run([pt, self._at_tpp(10)], max_ceiling_frac=None, min_tpp=5)
        self.assertEqual(len(kept), 1)
        self.assertEqual(rep["by_reason"], {"min_tpp": 1})

    def test_off_by_default(self):
        kept, rep = self._run([self._at_tpp(1)], max_ceiling_frac=None)
        self.assertEqual(len(kept), 1)
        self.assertEqual(rep["n_dropped"], 0)

    def test_report_carries_the_tpp_range_even_when_the_screen_is_off(self):
        pts = [self._at_tpp(1), self._at_tpp(20)]
        _, rep = self._run(pts, max_ceiling_frac=None)
        self.assertAlmostEqual(rep["tpp_min_before"], 1.0, places=4)
        self.assertAlmostEqual(rep["tpp_max_before"], 20.0, places=4)
        self.assertEqual(rep["tpp_min_before"], rep["tpp_min_after"])

    def test_raises_on_an_unpopulated_param_count(self):
        with self.assertRaises(ValueError) as cm:
            self._run([_point(n_params=0, n_ne=0)], max_ceiling_frac=None, min_tpp=5)
        self.assertIn("process_eval_arrays", str(cm.exception))

    def test_emptying_a_cohort_names_the_screen(self):
        with self.assertRaises(ValueError) as cm:
            self._run([self._at_tpp(1)], max_ceiling_frac=None, min_tpp=5)
        self.assertIn("min_tpp", str(cm.exception))


class TestPrintReport(unittest.TestCase):
    def test_print_report_labels_the_active_criteria(self):
        import contextlib, io
        pts = [_point(loss=8.0), _point(loss=3.0)]
        _, report = dl.filter_cohorts({0.0: pts}, max_ceiling_frac=0.70, min_tpp=5)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            dl.print_filter_report(report, max_ceiling_frac=0.70, min_tpp=5)
        out = buf.getvalue()
        self.assertIn("L < 0.7 ln(vocab)", out)
        self.assertIn("tokens/total-param >= 5", out)
        self.assertIn("1 by ceiling", out)

    def test_fully_disabled_filter_says_so_and_still_shows_the_columns(self):
        import contextlib, io
        _, report = dl.filter_cohorts({0.0: [_point()]}, max_ceiling_frac=None)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            dl.print_filter_report(report, max_ceiling_frac=None)
        out = buf.getvalue()
        self.assertIn("OFF", out)
        self.assertIn("TPP range", out)
        self.assertIn("ne_frac min", out)


class TestMinNeFrac(unittest.TestCase):
    """The DESIGN screen on model SHAPE: how much of the parameter count is not the token embedding."""
    def _run(self, points, **kw):
        kept, report = dl.filter_cohorts({0.0: list(points)}, **kw)
        return kept[0.0], report[0.0]

    def test_ne_frac_is_non_embedding_over_total(self):
        self.assertAlmostEqual(dl._ne_frac(_point(n_params=1_155_392, n_ne=106_816)),
                               106_816 / 1_155_392)          # the real d0064_L02 shape: 9.2%
        self.assertAlmostEqual(dl._ne_frac(_point(n_params=10_000_000, n_ne=8_000_000)), 0.8)

    def test_drops_embedding_dominated_shapes(self):
        pts = [_point(n_params=1_155_392, n_ne=106_816, loss=5.2),     # d0064_L02, 0.092
               _point(n_params=1_208_768, n_ne=160_192, loss=5.1),     # d0064_L03, 0.133
               _point(n_params=1_905_312, n_ne=332_448, loss=4.8),     # d0096_L03, 0.175
               _point(n_params=10_621_184, n_ne=6_426_880, loss=3.3)]  # d0256_L08, 0.605
        kept, rep = self._run(pts, max_ceiling_frac=None, min_ne_frac=0.15)
        self.assertEqual([p.eval_loss for p in kept], [4.8, 3.3])
        self.assertEqual(rep["by_reason"], {"min_ne_frac": 2})
        self.assertAlmostEqual(rep["ne_frac_min_before"], 106_816 / 1_155_392)
        self.assertAlmostEqual(rep["ne_frac_min_after"], 332_448 / 1_905_312)

    def test_off_by_default(self):
        """Defaults to None, so loading an array does not silently change under an upgrade."""
        kept, rep = self._run([_point(n_params=1_155_392, n_ne=106_816, loss=5.2)],
                              max_ceiling_frac=None)
        self.assertEqual(len(kept), 1)
        self.assertEqual(rep["n_dropped"], 0)

    def test_report_carries_the_column_even_when_the_screen_is_off(self):
        """Same contract as the other diagnostic columns: you learn that a ladder's bottom."""
        _, rep = self._run([_point(n_params=1_155_392, n_ne=106_816)], max_ceiling_frac=None)
        self.assertAlmostEqual(rep["ne_frac_min_before"], 106_816 / 1_155_392)
        self.assertEqual(rep["ne_frac_min_before"], rep["ne_frac_min_after"])

    def test_it_is_a_shape_cut_not_a_loss_cut(self):
        """The distinguishing property: two runs of the SAME architecture at different losses."""
        shape = dict(n_params=10_621_184, n_ne=6_426_880)
        kept, _ = self._run([_point(loss=3.3, **shape), _point(loss=5.9, **shape)],
                            max_ceiling_frac=None, min_ne_frac=0.25)
        self.assertEqual(len(kept), 2)

    def test_checked_before_the_duration_screen(self):
        """A d64 run that is ALSO undertrained is blamed on its SHAPE."""
        bad = _point(n_params=1_155_392, n_ne=106_816, max_iters=2, loss=5.2)   # tpp ~ 0.23
        _, rep = self._run([bad, _point(n_params=10_621_184, n_ne=6_426_880, loss=3.3)],
                           max_ceiling_frac=None, min_ne_frac=0.15, min_tpp=5)
        self.assertEqual(rep["by_reason"], {"min_ne_frac": 1})

    def test_rejects_out_of_range_thresholds(self):
        pts = [_point()]
        with self.assertRaises(ValueError):
            self._run(pts, min_ne_frac=1.0)      # no model is 100% non-embedding -> drops everything
        with self.assertRaises(ValueError):
            self._run(pts, min_ne_frac=0.0)
        with self.assertRaises(ValueError):
            self._run(pts, min_ne_frac=15)       # a percentage, not a fraction

    def test_raises_on_an_unpopulated_param_count(self):
        with self.assertRaises(ValueError) as cm:
            self._run([_point(n_params=0, n_ne=0)], max_ceiling_frac=None, min_ne_frac=0.15)
        self.assertIn("process_eval_arrays", str(cm.exception))

    def test_emptying_a_cohort_raises_and_names_the_screen(self):
        with self.assertRaises(ValueError) as cm:
            self._run([_point(n_params=1_155_392, n_ne=106_816)],
                      max_ceiling_frac=None, min_ne_frac=0.15)
        self.assertIn("min_ne_frac", str(cm.exception))

    def test_print_report_labels_the_criterion(self):
        import contextlib, io
        pts = [_point(n_params=1_155_392, n_ne=106_816, loss=5.2),
               _point(n_params=10_621_184, n_ne=6_426_880, loss=3.3)]
        _, report = dl.filter_cohorts({0.0: pts}, max_ceiling_frac=None, min_ne_frac=0.15)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            dl.print_filter_report(report, max_ceiling_frac=None, min_ne_frac=0.15)
        out = buf.getvalue()
        self.assertIn("non-emb params >= 0.15", out)
        self.assertIn("1 by min_ne_frac", out)
        self.assertIn("ne_frac min", out)
        self.assertIn("0.092 -> 0.605", out)


if __name__ == "__main__":
    unittest.main()
