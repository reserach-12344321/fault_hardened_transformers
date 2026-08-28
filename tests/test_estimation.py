"""Unit tests for the Chinchilla fit + pairs bootstrap in scaling_law/.

Pure numpy/scipy, no JAX and no disk, on synthetic data drawn from a known surface with
small multiplicative log-space noise.

Two tests are load-bearing beyond ordinary coverage and should not be weakened:
test_solver_tolerances_are_tight (scipy's defaults silently collapse the bootstrap CIs) and
test_matches_the_power_law_basis (the surface is parametrised in centred log-coordinates,
and reproducing the old basis where they overlap is the only thing keeping that honest).
"""
import unittest

import numpy as np

import scaling_law as est
from scaling_law import starts

# Amplitudes chosen so BOTH power-law terms are comparable to E across the grid (each spans ~0.1..2
# in loss), i.e. alpha AND beta are well identified -- otherwise a swamped term biases its exponent.
TRUE = dict(E=1.5, A=126.0, B=3170.0, alpha=0.30, beta=0.40)
# The same truth as the fit reports it: decay rates, positive, at the default pivots.
TRUE_SLOPE = dict(alpha1=TRUE["alpha"], beta1=TRUE["beta"])


def _synth(seed=0, sigma=0.01):
    """A designed (N, D) grid + known-surface losses with multiplicative log-noise -> (N, D, L, TRUE)."""
    rng = np.random.default_rng(seed)
    Ns = np.geomspace(1e6, 1e9, 8)
    Ds = np.geomspace(1e8, 1e11, 8)
    N, D = (a.ravel() for a in np.meshgrid(Ns, Ds))
    L = TRUE["E"] + TRUE["A"] / N ** TRUE["alpha"] + TRUE["B"] / D ** TRUE["beta"]
    L = L * np.exp(rng.normal(0.0, sigma, L.size))     # heteroskedastic in raw L, ~const in log space
    return N, D, L


# The COARSE centroid grid, lifted out of scaling_law.starts when it became test-only. It is
# the non-warm alternative these tests hand to `bootstrap(starts_boot=...)` -- deliberately
# cheap, since it is solved once per resample.
_COARSE = dict(alpha1s=(0.05, 0.5), beta1s=(0.05, 0.5), splits=(0.35, 0.65),
               e_shares=(0.1,), alpha2s=(0.0, 0.03), beta2s=(0.0, 0.03))


def _coarse_starts(u, v, logL, fix_E, fix_alpha2=0.0, fix_beta2=0.0):
    return est.centroid_starts(u, v, logL, fix_E, fix_alpha2, fix_beta2, **_COARSE)


def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    raise AssertionError(f"expected {exc.__name__}")


class EstimationTest(unittest.TestCase):
    def test_fit_recovers_exponents(self):
        """On low-noise synthetic data the joint fit should recover the true alpha, beta closely."""
        N, D, L = _synth()
        f = est.fit(N, D, L)
        assert abs(f.alpha1 - TRUE_SLOPE["alpha1"]) < 0.05, f.alpha1
        assert abs(f.beta1 - TRUE_SLOPE["beta1"]) < 0.05, f.beta1
        assert f.alpha2 == 0.0 and f.beta2 == 0.0    # the default surface is the plain power law


    def test_bootstrap_is_deterministic(self):
        """Same inputs + seed -> identical case indices -> identical parameter samples."""
        N, D, L = _synth()
        b1 = est.bootstrap(N, D, L, n_boot=32, seed=1, n_jobs=1)
        b2 = est.bootstrap(N, D, L, n_boot=32, seed=1, n_jobs=1)
        for k in est.PARAM_NAMES:
            assert np.array_equal(b1.samples[k], b2.samples[k]), k
        assert b1.n_boot == 32


    def test_ci_covers_truth(self):
        """The CI should bracket the true exponents on well-identified synthetic data."""
        N, D, L = _synth()
        b = est.bootstrap(N, D, L, n_boot=200, seed=0, n_jobs=1)
        for p in ("alpha1", "beta1"):
            lo, hi = b.ci(p)
            assert lo <= TRUE_SLOPE[p] <= hi, (p, lo, hi)


    def test_invalid_arguments_raise(self):
        N, D, L = _synth()
        # A curvature must be None (fit it) or a FINITE constant; nan/inf would poison every start.
        _raises(ValueError, est.fit, N, D, L, fix_alpha2=np.nan)
        _raises(ValueError, est.fit, N, D, L, fix_beta2=np.inf)


    def test_solver_tolerances_are_tight(self):
        """The tolerances are load-bearing, not cosmetic -- see solver.LBFGSB_OPTS."""
        assert est.LBFGSB_OPTS["gtol"] <= 1e-10, est.LBFGSB_OPTS
        assert est.LBFGSB_OPTS["ftol"] <= 1e-14, est.LBFGSB_OPTS


    def test_centroid_starts_pass_through_the_data_centroid(self):
        """Each start's surface carries its declared share of the loss at the centroid, which is what
        lets a and b be DERIVED rather than gridded.
        """
        N, D, L = _synth()
        u, v = est.centred(N, D)
        logL = np.log(L)
        mL, mu, mv = logL.mean(), u.mean(), v.mean()
        # Checked with each curvature held and freed: a freed one multiplies the grid but must not move
        # any start off the centroid, since the amplitude is solved for the curvature it carries.
        for fix_alpha2, fix_beta2, mult in ((0.0, 0.0, 1),
                                            (None, 0.0, len(starts.ALPHA2S)),
                                            (0.0, None, len(starts.BETA2S)),
                                            (None, None, len(starts.ALPHA2S) * len(starts.BETA2S))):
            grid = est.centroid_starts(u, v, logL, fix_E=0.0, fix_alpha2=fix_alpha2,
                                       fix_beta2=fix_beta2)
            for e, a, al1, al2, b, be1, be2 in grid:
                n_term = np.exp(a - al1 * mu - al2 * mu ** 2)
                d_term = np.exp(b - be1 * mv - be2 * mv ** 2)
                assert np.isclose(n_term + d_term, np.exp(mL)), (n_term, d_term, np.exp(mL))
                assert e == -np.inf                   # fixed E -> no floor level is generated at all
            # ...and with E fixed there are no duplicate starts.
            assert len({tuple(s) for s in grid}) == len(grid) == \
                len(starts.ALPHA1S) * len(starts.BETA1S) * len(starts.SPLITS) * mult


    def test_multistart_reports_convergence(self):
        """The diagnostic must count starts and hits, and most starts should reach the optimum."""
        N, D, L = _synth()
        f, diag = est.fit_report(N, D, L, fix_E=0.0)
        assert diag["n_starts"] == 1 + len(starts.ALPHA1S) * len(starts.BETA1S) * len(starts.SPLITS)
        assert diag["n_at_optimum"] >= 0.5 * diag["n_starts"], diag
        assert f.objective > 0


    def test_warm_started_bootstrap_does_not_collapse_the_interval(self):
        """THE regression test for the warm-start change."""
        N, D, L = _synth()
        warm = est.bootstrap(N, D, L, n_boot=60, seed=7, n_jobs=1, fix_E=0.0)
        grid = est.bootstrap(N, D, L, n_boot=60, seed=7, n_jobs=1, fix_E=0.0,
                             starts_boot=_coarse_starts(*est.centred(N, D), np.log(L), 0.0))
        assert warm.starts["resample_mode"] == "warm" and grid.starts["resample_mode"] == "grid"
        for p in ("alpha1", "beta1"):
            # Same seed -> same case indices -> sample i of each is the SAME resampled dataset solved
            # from two different start sets, so they are comparable one for one.
            rel = np.abs(warm.samples[p] - grid.samples[p]) / np.maximum(np.abs(grid.samples[p]), 1e-12)
            agree = int((rel <= 1e-6).sum())
            assert agree >= 0.9 * rel.size, (p, agree, rel.size, rel.max())

            wlo, whi = warm.ci(p)
            glo, ghi = grid.ci(p)
            ratio = (whi - wlo) / (ghi - glo)
            # The failure this guards is a COLLAPSE (an order-of-magnitude-too-tight interval),
            # which would show up here as a ratio near 0.05. The band is wide enough to tolerate the
            # tail-optimum disagreement measured above and far too tight to let a collapse through.
            assert 0.5 < ratio < 2.0, (p, ratio, (wlo, whi), (glo, ghi))


    def test_resamples_are_warm_started_whatever_is_free(self):
        """Warm-starting is no longer guarded on fix_E / fix_alpha2 -- see resample.bootstrap."""
        N, D, L = _synth()
        for fix_E, fix_alpha2, fix_beta2 in ((0.0, 0.0, 0.0), (None, 0.0, 0.0), (None, None, 0.0),
                                             (None, 0.0, None), (None, None, None)):
            b = est.bootstrap(N, D, L, n_boot=8, seed=0, n_jobs=1, fix_E=fix_E, fix_alpha2=fix_alpha2,
                              fix_beta2=fix_beta2)
            assert b.starts["resample_mode"] == "warm", (fix_E, fix_alpha2, fix_beta2, b.starts)
            assert "n_rescued" in b.starts
        forced = est.bootstrap(N, D, L, n_boot=8, seed=0, n_jobs=1, fix_E=None,
                               starts_boot=_coarse_starts(*est.centred(N, D), np.log(L), None))
        assert forced.starts["resample_mode"] == "grid", forced.starts


    def test_matches_the_power_law_basis(self):
        """The centred parametrisation must BE the old ``E + A/N^alpha + B/D^beta`` surface, exactly."""
        N, D, L = _synth()
        logL = np.log(L)
        old = dict(E=1.6, A=34.0, alpha=0.20, B=2600.0, beta=0.41)

        def old_predict(NN, DD):
            return old["E"] + old["A"] / NN ** old["alpha"] + old["B"] / DD ** old["beta"]

        # the surface, at three different gauges -- the pivots must not move a single prediction
        for N0, D0 in ((1.0, 1.0), (est.N0_DEFAULT, est.D0_DEFAULT), (3.3e8, 7e9)):
            f = est.ChinchillaFit.from_power_law(**old, N0=N0, D0=D0)
            assert np.allclose(f.predict(N, D), old_predict(N, D), rtol=1e-12), (N0, D0)
            assert np.isclose(f.alpha1, old["alpha"], rtol=1e-12)       # decay rates: same sign
            assert np.isclose(f.beta1, old["beta"], rtol=1e-12)
            assert f.alpha2 == 0.0 and f.beta2 == 0.0                   # no curvature in the old basis
            # ...and `a` is the N-term's value AT the pivot, which is the point of the basis
            assert np.isclose(f.amp_N, old["A"] / N0 ** old["alpha"], rtol=1e-12)

        # the OBJECTIVE is a function of the surface, so it is invariant too
        f0 = est.ChinchillaFit.from_power_law(**old, N0=est.N0_DEFAULT, D0=est.D0_DEFAULT)
        f1 = est.ChinchillaFit.from_power_law(**old, N0=1.0, D0=1.0)
        v0, _ = est.objective_and_grad(f0.as_theta(), *est.centred(N, D), logL, est.DEFAULT_HUBER)
        v1, _ = est.objective_and_grad(f1.as_theta(), *est.centred(N, D, 1.0, 1.0), logL,
                                       est.DEFAULT_HUBER)
        assert np.isclose(v0, v1, rtol=1e-12), (v0, v1)


    def test_fit_is_gauge_invariant(self):
        """Fitting at different pivots must give the SAME surface: only the coordinates move."""
        N, D, L = _synth()
        ref = est.fit(N, D, L, fix_E=None)
        for N0, D0 in ((1.0, 1.0), (3.3e8, 7e9)):
            f = est.fit(N, D, L, fix_E=None, N0=N0, D0=D0)
            assert np.allclose(f.predict(N, D), ref.predict(N, D), rtol=1e-5), (N0, D0)
            assert np.isclose(f.alpha1, ref.alpha1, atol=1e-4), (f.alpha1, ref.alpha1)
            assert np.isclose(f.beta1, ref.beta1, atol=1e-4), (f.beta1, ref.beta1)
            # a moves by exactly -alpha1 * log(N0_new / N0_ref): equating the N-term at the two gauges,
            # a' - alpha1*log(N/N0') = a - alpha1*log(N/N0)  ->  a' = a - alpha1*log(N0'/N0).
            assert np.isclose(f.a, ref.a - ref.alpha1 * np.log(N0 / ref.N0), atol=1e-3), (f.a, ref.a)


    def test_analytic_gradient_matches_finite_differences(self):
        """The gradient is handed to L-BFGS-B with jac=True, so an error in it is an error in every fit."""
        N, D, L = _synth()
        u, v = est.centred(N, D)
        logL = np.log(L)
        for theta in (np.array([np.log(1.6), 0.02, 0.20, 0.03, -0.9, 0.41, 0.05]),
                      np.array([np.log(1.6), 0.02, 0.20, -0.05, -0.9, 0.41, -0.04]),
                      np.array([-np.inf, 0.02, 0.20, -0.05, -0.9, 0.41, 0.02])):
            _, g = est.objective_and_grad(theta, u, v, logL, est.DEFAULT_HUBER)
            for i in range(7):
                if not np.isfinite(theta[i]):
                    assert g[i] == 0.0                      # pinned coordinate contributes nothing
                    continue
                h = 1e-6 * max(1.0, abs(theta[i]))
                tp, tm = theta.copy(), theta.copy()
                tp[i] += h
                tm[i] -= h
                num = (est.objective_and_grad(tp, u, v, logL, est.DEFAULT_HUBER)[0]
                       - est.objective_and_grad(tm, u, v, logL, est.DEFAULT_HUBER)[0]) / (2 * h)
                assert np.isclose(g[i], num, rtol=1e-4, atol=1e-10), (i, g[i], num)


    def test_alpha2_is_off_by_default_and_pinnable(self):
        """The curvature must default to exactly 0 (so nothing changes for callers that never."""
        N, D, L = _synth()
        assert est.fit(N, D, L).alpha2 == 0.0
        assert est.fit(N, D, L, fix_alpha2=0.02).alpha2 == 0.02
        free = est.fit(N, D, L, fix_alpha2=None)
        pinned = est.fit(N, D, L, fix_alpha2=0.0)
        assert free.objective <= pinned.objective * (1 + 1e-9)      # a superset of the model
        # on data generated WITHOUT curvature, the freed coefficient should stay small
        assert abs(free.alpha2) < 0.05, free.alpha2


    def test_beta2_is_off_by_default_and_pinnable(self):
        """The D-curvature is the exact mirror of alpha2: off by default, pinnable."""
        N, D, L = _synth()
        assert est.fit(N, D, L).beta2 == 0.0
        assert est.fit(N, D, L, fix_beta2=0.02).beta2 == 0.02
        free = est.fit(N, D, L, fix_beta2=None)
        pinned = est.fit(N, D, L, fix_beta2=0.0)
        assert free.objective <= pinned.objective * (1 + 1e-9)      # a superset of the model
        # on data generated WITHOUT curvature, the freed coefficient should stay small
        assert abs(free.beta2) < 0.05, free.beta2
        # ...and freeing it must not disturb the term it curves: beta1 is still the slope, now read at D0
        assert abs(free.beta1 - TRUE_SLOPE["beta1"]) < 0.05, free.beta1


def _curved_D_synth(sigma=0.004, seed=5):
    """A grid drawn from the FITTED form with a real D-curvature."""
    rng = np.random.default_rng(seed)
    Ns = np.geomspace(1e6, 1e9, 8)
    Ds = np.geomspace(1e8, 1e11, 8)
    N, D = (a.ravel() for a in np.meshgrid(Ns, Ds))
    truth = est.ChinchillaFit(E=1.5, a=np.log(0.9), alpha1=0.30, b=np.log(0.7), beta1=0.40,
                              objective=np.nan, beta2=0.06)
    L = truth.predict(N, D)
    if sigma:
        L = L * np.exp(rng.normal(0.0, sigma, L.size))
    return N, D, L, truth


class CurvedDTermTest(unittest.TestCase):
    def test_beta2_recovers_a_curved_D_term(self):
        """On data built WITH a D-curvature."""
        N, D, L, truth = _curved_D_synth(sigma=0.0)
        exact = est.fit(N, D, L, fix_beta2=None)
        for k in ("E", "a", "alpha1", "b", "beta1", "beta2"):
            assert np.isclose(getattr(exact, k), getattr(truth, k), rtol=1e-6, atol=1e-8), k

        N, D, L, truth = _curved_D_synth(sigma=0.004)
        free = est.fit(N, D, L, fix_beta2=None)
        assert np.allclose(free.predict(N, D), truth.predict(N, D), rtol=0.01)
        # Holding the curvature at 0 must fit WORSE -- if it did not, beta2 would be unidentifiable here
        # and even the noiseless recovery above would be reading a flat direction.
        held = est.fit(N, D, L, fix_beta2=0.0)
        assert free.objective < held.objective, (free.objective, held.objective)


    def test_beta2_bootstrap_is_conditioned_on_the_same_model(self):
        """A freed beta2 must reach the CIs: the resamples refit the SAME model as the point."""
        N, D, L = _synth()
        b = est.bootstrap(N, D, L, n_boot=40, seed=0, n_jobs=1, fix_beta2=None)
        assert b.samples["beta2"].std() > 0.0
        held = est.bootstrap(N, D, L, n_boot=8, seed=0, n_jobs=1)
        assert np.all(held.samples["beta2"] == 0.0)        # held: every resample carries the constant


    def test_beta2_widens_the_beta1_interval(self):
        """Freeing the D-curvature must COST precision on beta1 -- they trade off along a ridge."""
        N, D, L = _synth()
        held = est.bootstrap(N, D, L, n_boot=120, seed=0, n_jobs=1)
        free = est.bootstrap(N, D, L, n_boot=120, seed=0, n_jobs=1, fix_beta2=None)
        w_held = np.subtract(*reversed(held.ci("beta1")))
        w_free = np.subtract(*reversed(free.ci("beta1")))
        assert w_free > w_held, (w_held, w_free)


    def test_saturation_refuses_a_diverging_D_term(self):
        """beta2 < 0 makes the D-term grow without bound as D -> infinity, so there is no saturated loss."""
        ok = est.ChinchillaFit(E=1.5, a=0.0, alpha1=0.3, b=0.0, beta1=0.4, objective=0.0, beta2=0.05)
        assert np.isfinite(ok.saturated_loss(1e8))
        bad = est.ChinchillaFit(E=1.5, a=0.0, alpha1=0.3, b=0.0, beta1=0.4, objective=0.0, beta2=-0.05)
        _raises(ValueError, bad.saturated_loss, 1e8)


    def test_pivots_are_validated(self):
        N, D, L = _synth()
        _raises(ValueError, est.fit, N, D, L, N0=0.0)
        _raises(ValueError, est.fit, N, D, L, D0=-1.0)
        _raises(ValueError, est.fit, N, D, L, N0=np.inf)


if __name__ == "__main__":
    unittest.main()
