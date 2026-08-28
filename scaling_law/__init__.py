r"""Scaling-law analysis: fit a Chinchilla surface to a set of training runs.

    L(N, D) = E + exp(a - alpha1*u - alpha2*u**2) + exp(b - beta1*v - beta2*v**2)
    u = log(N/N0),  v = log(D/D0)             (Hoffmann et al. 2022 when alpha2 = beta2 = 0)

N is total parameters, D training tokens, L eval loss in nats. Every (N, D, L) row is a
separate training run -- there are no intermediate-checkpoint points.

The estimator is this package's top level, so the usual import is `import scaling_law as
est`. Start in surface.py -- the parametrisation is the least obvious part -- then solver.py
and resample.py. Two things not to "simplify": the solver tolerances, and the multi-start.
"""
from .resample import BootstrapResult, bootstrap, warm_starts
from .solver import (LBFGSB_OPTS, PARAM_BOUNDS, bounds_for, clamp_start, expand_theta, fit,
                     fit_report, fixed_e, free_coords, multistart)
from .starts import centroid_starts, default_starts, ols_start, start_from_shares
from .surface import (D0_DEFAULT, DEFAULT_HUBER, MIN_RUNS, N0_DEFAULT, PARAM_NAMES, ChinchillaFit,
                      centred, check_fix_E, check_fix_alpha2, check_fix_beta2, check_pivots,
                      default_pivot, diagnostics, log_pred_softmax, objective_and_grad,
                      validate_arrays)

__all__ = [
    # model
    "ChinchillaFit", "PARAM_NAMES", "DEFAULT_HUBER", "MIN_RUNS", "objective_and_grad",
    "log_pred_softmax", "validate_arrays", "check_fix_E", "check_fix_alpha2", "check_fix_beta2",
    "check_pivots", "centred", "default_pivot", "diagnostics", "N0_DEFAULT", "D0_DEFAULT",
    # starts
    "ols_start", "centroid_starts", "default_starts", "start_from_shares",
    # solver
    "fit", "fit_report", "multistart", "free_coords", "fixed_e", "expand_theta", "clamp_start",
    "bounds_for", "PARAM_BOUNDS", "LBFGSB_OPTS",
    # bootstrap
    "bootstrap", "BootstrapResult", "warm_starts",
]
