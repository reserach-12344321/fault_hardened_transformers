"""Uncertainty: the pairs (case) bootstrap.

Each resample draws n runs i.i.d. with replacement and refits the surface, warm-started from
the point estimate; `n_rescued` counts how often one of the spread alternatives beats that
warm start, which is what tells you the surface has gone multimodal on resamples.

Case resampling treats the rows as an i.i.d. sample, so the reported uncertainty includes
which DESIGN POINTS were drawn. This sweep's (N, D) grid is chosen, not sampled, so a
resample that drops the large-N anchors is not a plausible alternative experiment -- the
intervals here are conservative, mostly on the exponents.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from joblib import Parallel, delayed

from .solver import DEFAULT_HUBER, fit_report, multistart
from .starts import start_from_shares
from .surface import (D0_DEFAULT, N0_DEFAULT, PARAM_NAMES, ChinchillaFit, centred, check_fix_E,
                      check_fix_alpha2, check_fix_beta2, check_pivots, validate_arrays)

# Extra starts carried alongside the warm start, as (alpha1, beta1, s_A), spread across
# N-dominated / balanced / D-dominated surfaces so a resample that has genuinely moved to
# another basin is still found. They are also what makes `n_rescued` meaningful.
_WARM_EXTRA = ((0.05, 0.5, 0.35), (0.5, 0.05, 0.65), (0.2, 0.2, 0.5))


def warm_starts(point: ChinchillaFit, u, v, logL, fix_E, fix_alpha2=0.0,
                fix_beta2=0.0) -> list[np.ndarray]:
    """The point estimate as start #0, plus the _WARM_EXTRA centroid starts as insurance."""
    mL, mu, mv = float(np.mean(logL)), float(np.mean(u)), float(np.mean(v))
    s_E = 0.0 if fix_E is not None else 0.1
    a2 = point.alpha2 if fix_alpha2 is None else fix_alpha2
    b2 = point.beta2 if fix_beta2 is None else fix_beta2
    return [point.as_theta()] + [
        start_from_shares(mL, mu, mv, al, be, s * (1 - s_E), (1 - s) * (1 - s_E), s_E, a2, b2)
        for al, be, s in _WARM_EXTRA]


@dataclass
class BootstrapResult:
    """Point estimate + the per-parameter arrays of bootstrap-resampled estimates."""
    point: ChinchillaFit
    samples: dict            # {"E": (n_boot,) array, "A": ..., ...} -- one entry per PARAM_NAMES
    n_boot: int
    # Convergence witness: the point fit's multi-start counts, the resample mode, and
    # n_rescued -- resamples where an extra start beat the warm start. Persistently
    # non-zero means the surface is multimodal on resamples and warm starting is not safe.
    starts: dict

    def ci(self, param: str, lo: float = 2.5, hi: float = 97.5) -> tuple[float, float]:
        """Percentile confidence interval for a parameter from the bootstrap distribution."""
        v = self.samples[param]
        return float(np.percentile(v, lo)), float(np.percentile(v, hi))

    def stats(self, param: str) -> dict:
        """median / std / 2.5% / 97.5% / CV for one parameter (CV = std / |mean|)."""
        v = self.samples[param]
        lo, hi = self.ci(param)
        return {"point": getattr(self.point, param), "median": float(np.median(v)),
                "std": float(np.std(v)), "lo": lo, "hi": hi,
                "cv": float(np.std(v) / (abs(np.mean(v)) + 1e-12))}


def bootstrap(N, D, L, *, n_boot: int = 1000, seed: int = 0,
              n_jobs: int = -1, huber: float = DEFAULT_HUBER,
              starts_point: Sequence | None = None, starts_boot: Sequence | None = None,
              fix_E: float | None = None, fix_alpha2: float | None = 0.0,
              fix_beta2: float | None = 0.0,
              N0: float = N0_DEFAULT, D0: float = D0_DEFAULT) -> BootstrapResult:
    """Pairs-bootstrap the Chinchilla fit, in parallel over resamples."""
    N, D, L = validate_arrays(N, D, L)
    check_fix_E(fix_E, L)
    check_fix_alpha2(fix_alpha2)
    check_fix_beta2(fix_beta2)
    check_pivots(N0, D0)
    u, v = centred(N, D, N0, D0)
    logL = np.log(L)

    point, point_diag = fit_report(N, D, L, huber=huber, starts=starts_point, fix_E=fix_E,
                                   fix_alpha2=fix_alpha2, fix_beta2=fix_beta2, N0=N0, D0=D0)

    # A resample is a small perturbation of the surface just fitted, so the point estimate
    # is an excellent start and a full multi-start per resample is wasted work. The
    # insurance is `warm_starts`' three spread alternatives, and `n_rescued` counts the
    # resamples where one of them won -- a persistently non-zero count means the surface is
    # multimodal on resamples and these intervals are too narrow.
    warm = starts_boot is None
    starts = warm_starts(point, u, v, logL, fix_E, fix_alpha2, fix_beta2) if warm else starts_boot

    def fit_one(ru, rv, rL):
        """One resample -> (fit, was the warm start beaten by an alternative?)."""
        f, diag = multistart(ru, rv, rL, starts, huber=huber, fix_E=fix_E, fix_alpha2=fix_alpha2,
                             fix_beta2=fix_beta2, N0=N0, D0=D0)
        return f, (warm and not diag["first_at_optimum"])

    # Indices drawn up front, not inside the workers, so the bootstrap is deterministic
    # given the seed regardless of how joblib schedules the refits.
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, N.size, size=(n_boot, N.size))
    out = Parallel(n_jobs=n_jobs)(
        delayed(fit_one)(u[idx[i]], v[idx[i]], logL[idx[i]]) for i in range(n_boot))

    samples = {k: np.array([getattr(f, k) for f, _ in out]) for k in PARAM_NAMES}
    return BootstrapResult(
        point=point, samples=samples, n_boot=n_boot,
        starts={"point_starts": point_diag["n_starts"],
                "point_at_optimum": point_diag["n_at_optimum"],
                "resample_mode": "warm" if warm else "grid",
                "n_rescued": int(sum(rescued for _, rescued in out))})
