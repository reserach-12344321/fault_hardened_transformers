"""Minimising the Huber objective: multi-start bounded L-BFGS-B.

Holding a coordinate fixed is not a separate code path: theta is always
(e, a, alpha1, alpha2, b, beta1, beta2), and fix_E / fix_alpha2 / fix_beta2 remove one from
the free set. Both curvatures default to 0.0, which is the Chinchilla form exactly. `fit`
centres the data once, so the optimiser never sees N, D or a pivot.

TOLERANCES ARE EXPLICIT AND LOAD-BEARING. scipy's defaults are absolute thresholds and this
objective sits far below them, so they stop the solver early and only a handful of starts
reach the optimum -- invisible in the point estimate, but it collapses a warm-started
bootstrap's CI by an order of magnitude.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from .starts import default_starts
from .surface import (D0_DEFAULT, DEFAULT_HUBER, N0_DEFAULT, ChinchillaFit, centred, check_fix_E,
                      check_fix_alpha2, check_fix_beta2, check_pivots, objective_and_grad,
                      validate_arrays)

# Do NOT relax these to scipy's defaults -- see the module docstring.
LBFGSB_OPTS = dict(gtol=1e-12, ftol=1e-16, maxiter=20000)

# Per-coordinate bounds on theta = (e, a, alpha1, alpha2, b, beta1, beta2).
#   e       unconstrained below, with a per-fit upper cap (E < min L) from `bounds_for`
#   a, b    unconstrained log-amplitudes at the pivots; the data sets their scale
#   alpha1  decay rates at the pivots, strictly positive: more parameters, or more tokens,
#   beta1   cannot make the loss worse. The EFFECTIVE rate may still go negative away from
#           the pivot when a curvature is free, which is part of what curvature is for.
#   alpha2  finite but generous, and not sign-constrained. |curvature| > 1 would swing the
#   beta2   effective exponent by more than +-2 across a decade, which is not a scaling law.
PARAM_BOUNDS = ((None, None), (None, None), (1e-3, 5.0), (-1.0, 1.0), (None, None), (1e-3, 5.0),
                (-1.0, 1.0))
_I_E = 0                                      # index of e      in theta
_I_ALPHA2 = 3                                 # index of alpha2 in theta
_I_BETA2 = 6                                  # index of beta2  in theta


def bounds_for(free: tuple[int, ...], e_hi: float) -> list[tuple]:
    """Bounds for one block of free coordinates, with e capped at e_hi (E < min L)."""
    return [(PARAM_BOUNDS[i][0], e_hi) if i == _I_E else PARAM_BOUNDS[i] for i in free]


def free_coords(fix_E: float | None, fix_alpha2: float | None = 0.0,
                fix_beta2: float | None = 0.0) -> tuple[int, ...]:
    """Which coordinates of theta the optimizer varies."""
    free = ([_I_E] if fix_E is None else []) + [1, 2] \
        + ([_I_ALPHA2] if fix_alpha2 is None else []) + [4, 5] \
        + ([_I_BETA2] if fix_beta2 is None else [])
    return tuple(sorted(free))


def fixed_e(fix_E: float | None) -> float | None:
    """The constant e = log E when E is held, else None. fix_E == 0 -> -inf."""
    if fix_E is None:
        return None
    return -np.inf if fix_E == 0.0 else float(np.log(fix_E))


def expand_theta(x_free, free, e_const, alpha2_const: float | None = 0.0,
                 beta2_const: float | None = 0.0) -> np.ndarray:
    """A free sub-vector back to a full theta."""
    theta = np.empty(7)
    theta[list(free)] = x_free
    if e_const is not None:
        theta[_I_E] = e_const
    if alpha2_const is not None:
        theta[_I_ALPHA2] = alpha2_const
    if beta2_const is not None:
        theta[_I_BETA2] = beta2_const
    return theta


def clamp_start(theta, e_hi: float) -> np.ndarray:
    """A start vector moved strictly inside the feasible box, which L-BFGS-B requires."""
    theta = np.asarray(theta, dtype=float).copy()
    if theta.size != 7:
        raise ValueError(f"a start must be a 7-vector (e, a, alpha1, alpha2, b, beta1, beta2); "
                         f"got size {theta.size}. Starts in the 6-coordinate basis that predates "
                         f"beta2 pad with a trailing 0.0 (the Chinchilla D-term); starts in the old "
                         f"5-coordinate basis (a, b, e, alpha, beta) are NOT convertible by padding "
                         f"at all -- the exponents change sign and the amplitudes move to the "
                         f"pivots.")
    theta[_I_E] = e_hi - 1.0 if not np.isfinite(theta[_I_E]) else min(theta[_I_E], e_hi)
    for i in (2, 3, 5, 6):
        theta[i] = np.clip(theta[i], *PARAM_BOUNDS[i])
    return theta


def multistart(u, v, logL, starts: Sequence, *, huber: float = DEFAULT_HUBER,
               fix_E: float | None = None, fix_alpha2: float | None = 0.0,
               fix_beta2: float | None = 0.0,
               N0: float = N0_DEFAULT, D0: float = D0_DEFAULT) -> tuple[ChinchillaFit, dict]:
    """Fit from every start in `starts`; the lowest objective wins. -> (fit, diag)."""
    free = free_coords(fix_E, fix_alpha2, fix_beta2)
    e_const = fixed_e(fix_E)
    e_hi = float(np.min(logL))                        # E < min L
    bounds = bounds_for(free, e_hi)

    def objective(x_free):
        theta = expand_theta(x_free, free, e_const, fix_alpha2, fix_beta2)
        val, grad = objective_and_grad(theta, u, v, logL, huber)
        return val, grad[list(free)]

    results = [minimize(objective, clamp_start(start, e_hi)[list(free)], jac=True,
                        method="L-BFGS-B", bounds=bounds, options=LBFGSB_OPTS)
               for start in starts]

    best = min(results, key=lambda r: r.fun)
    fit = ChinchillaFit.from_theta(expand_theta(best.x, free, e_const, fix_alpha2, fix_beta2),
                                   best.fun, N0=N0, D0=D0)
    return fit, starts_diag([r.fun for r in results], best.fun)


def starts_diag(objectives, best: float) -> dict:
    """How many starts reached the winning objective, and whether the first did."""
    f = np.asarray(objectives, dtype=float)
    tol = abs(best) * 1e-9 + 1e-15
    return {"n_starts": int(f.size), "n_at_optimum": int((f <= best + tol).sum()),
            "first_at_optimum": bool(f[0] <= best + tol)}


def fit(N, D, L, *, huber: float = DEFAULT_HUBER, starts: Sequence | None = None,
        fix_E: float | None = None, fix_alpha2: float | None = 0.0,
        fix_beta2: float | None = 0.0,
        N0: float = N0_DEFAULT, D0: float = D0_DEFAULT) -> ChinchillaFit:
    """Point estimate of the surface on the full sample."""
    return fit_report(N, D, L, huber=huber, starts=starts, fix_E=fix_E, fix_alpha2=fix_alpha2,
                      fix_beta2=fix_beta2, N0=N0, D0=D0)[0]


def fit_report(N, D, L, *, huber: float = DEFAULT_HUBER, starts: Sequence | None = None,
               fix_E: float | None = None, fix_alpha2: float | None = 0.0,
               fix_beta2: float | None = 0.0,
               N0: float = N0_DEFAULT, D0: float = D0_DEFAULT) -> tuple[ChinchillaFit, dict]:
    """`fit` plus its multi-start diagnostic."""
    N, D, L = validate_arrays(N, D, L)
    check_fix_E(fix_E, L)
    check_fix_alpha2(fix_alpha2)
    check_fix_beta2(fix_beta2)
    check_pivots(N0, D0)
    u, v = centred(N, D, N0, D0)
    logL = np.log(L)
    if starts is None:
        starts = default_starts(u, v, logL, fix_E, fix_alpha2, fix_beta2)
    return multistart(u, v, logL, starts, huber=huber, fix_E=fix_E, fix_alpha2=fix_alpha2,
                      fix_beta2=fix_beta2, N0=N0, D0=D0)
