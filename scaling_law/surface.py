r"""The fitted surface and its objective.

    L(N, D) = E + exp(a - alpha1*u - alpha2*u**2) + exp(b - beta1*v - beta2*v**2)

with u = log(N/N0), v = log(D/D0) -- Hoffmann et al. 2022 "Approach 3" when the curvatures
are 0. Centred coordinates rather than A/N**alpha + B/D**beta so `a` is the N-term's value
AT THE PIVOT rather than extrapolated to a one-parameter model, which is what put bootstrap
draws of (log A, alpha) on a near-1:1 ridge. The minus signs are in the form, so the
exponents are decay rates and positive on real data.

alpha2/beta2 are descriptive knobs, off by default: they say the effective exponent varies
across the ladder, not that a mechanism produces a quadratic. Neither can represent the
N x D interaction -- this form is additively separable -- and a fitted beta2 < 0 makes the
loss turn back up at large D, so saturated_loss refuses it. The pivots are a GAUGE, fixed so
amplitudes stay comparable across fits. Fit in log space, Huber loss, via logsumexp.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

# The reported parameters. ``E`` is the floor in nats; the rest are the polynomial coefficients of
# the two terms in centred log-coordinates. This is NOT the internal theta order -- see `as_theta`.
PARAM_NAMES = ("E", "a", "alpha1", "alpha2", "b", "beta1", "beta2")

# The pivots, decided once and held CONSTANT -- not default_pivot(N), which would make a and
# b incomparable between two fits of the same cohort. N0 is the geometric mean of the fitted
# sizes: the N side has no usable decorrelating pivot, being a three-way tangle with alpha2.
# D0 is the decorrelating pivot, which on the D side is well defined and stable.
N0_DEFAULT = 4.0e7
D0_DEFAULT = 4.0e7

# Fewest runs a fit is allowed: 5 free parameters with both curvatures held, up to 7 with
# them free. The floor is the smaller one, since demanding 7 would reject data the default
# model fits perfectly well.
MIN_RUNS = 5

# Huber transition in log-loss units. Hoffmann et al.'s value; at this delta the objective
# is nearly L1 (only ~8% of a real cohort's residuals fall in the quadratic region), which
# is why the solver needs explicit tolerances.
DEFAULT_HUBER = 1e-3


@dataclass
class ChinchillaFit:
    """A fitted L = E + exp(a - alpha1*u - alpha2*u**2) + exp(b - beta1*v - beta2*v**2)."""
    E: float
    a: float                  # log of the N-term's value AT the pivot N0
    alpha1: float             # log-log decay rate of the N-term at N0   (positive on real data)
    b: float                  # log of the D-term's value AT the pivot D0
    beta1: float              # log-log decay rate of the D-term at D0   (positive on real data)
    objective: float          # Huber objective at the optimum (lower = better)
    alpha2: float = 0.0       # curvature of the N-term in log space; 0 = Chinchilla
    beta2: float = 0.0        # curvature of the D-term in log space; 0 = Chinchilla
    N0: float = N0_DEFAULT
    D0: float = D0_DEFAULT

    # ---- the terms, and the coordinates they live in ---------------------------------------------
    def u(self, N):
        """Centred log model size ``log(N/N0)``."""
        return np.log(np.asarray(N, dtype=float) / self.N0)

    def v(self, D):
        """Centred log token count ``log(D/D0)``."""
        return np.log(np.asarray(D, dtype=float) / self.D0)

    def n_term(self, N):
        """``exp(a - alpha1*u - alpha2*u**2)`` -- the whole N-dependence."""
        u = self.u(N)
        return np.exp(self.a - self.alpha1 * u - self.alpha2 * u ** 2)

    def d_term(self, D):
        """``exp(b - beta1*v - beta2*v**2)`` -- the whole D-dependence."""
        v = self.v(D)
        return np.exp(self.b - self.beta1 * v - self.beta2 * v ** 2)

    def predict(self, N, D):
        """Predicted loss at model size(s) ``N`` and token count(s) ``D`` (broadcasting)."""
        return self.E + self.n_term(N) + self.d_term(D)

    def saturated_loss(self, N):
        """Loss at saturation, D -> infinity."""
        if self.beta2 < 0:
            raise ValueError(
                f"this fit has beta2={self.beta2:g} < 0, so its D-term DIVERGES as D -> infinity "
                f"(loss turns back up with more data) and there is no saturation limit to report. "
                f"Refit with fix_beta2=0.0 (or a non-negative constant) if the saturated loss is "
                f"what you need.")
        return self.E + self.n_term(N)

    # ---- derived readings ------------------------------------------------------------------------
    @property
    def amp_N(self) -> float:
        """The N-term's value at the pivot, in nats: ``exp(a)``."""
        return float(np.exp(self.a))

    @property
    def amp_D(self) -> float:
        """The D-term's value at the pivot, in nats: ``exp(b)``."""
        return float(np.exp(self.b))

    def effective_alpha(self, N):
        """Local log-log decay rate of the N-term: alpha1 + 2*alpha2*u."""
        return self.alpha1 + 2.0 * self.alpha2 * self.u(N)

    def effective_beta(self, D):
        """The D-side twin of effective_alpha: beta1 + 2*beta2*v, equal to beta1 at D0."""
        return self.beta1 + 2.0 * self.beta2 * self.v(D)

    def as_dict(self) -> dict:
        return {"E": self.E, "a": self.a, "alpha1": self.alpha1, "alpha2": self.alpha2,
                "b": self.b, "beta1": self.beta1, "beta2": self.beta2,
                "objective": self.objective, "N0": self.N0, "D0": self.D0}

    def as_theta(self) -> np.ndarray:
        """This surface as the internal theta (e, a, alpha1, alpha2, b, beta1, beta2)."""
        return np.array([np.log(self.E) if self.E > 0 else -np.inf,
                         self.a, self.alpha1, self.alpha2, self.b, self.beta1, self.beta2])

    @classmethod
    def from_theta(cls, theta, objective: float, N0: float = N0_DEFAULT,
                   D0: float = D0_DEFAULT) -> "ChinchillaFit":
        e, a, alpha1, alpha2, b, beta1, beta2 = theta
        return cls(E=float(np.exp(e)), a=float(a), alpha1=float(alpha1), alpha2=float(alpha2),
                   b=float(b), beta1=float(beta1), beta2=float(beta2), objective=float(objective),
                   N0=float(N0), D0=float(D0))

    @classmethod
    def from_power_law(cls, E, A, alpha, B, beta, objective: float = float("nan"),
                       gamma: float = 0.0, delta: float = 0.0, N0: float = N0_DEFAULT,
                       D0: float = D0_DEFAULT) -> "ChinchillaFit":
        """Build from the old basis L = E + A*N**-alpha * exp(-gamma*ln(N/N0)**2) + B*D**-beta."""
        return cls(E=float(E), a=float(np.log(A) - alpha * np.log(N0)), alpha1=float(alpha),
                   alpha2=float(gamma), b=float(np.log(B) - beta * np.log(D0)),
                   beta1=float(beta), beta2=float(delta), objective=float(objective),
                   N0=float(N0), D0=float(D0))

    @property
    def compute_optimal_exponents(self) -> tuple[float, float]:
        """Compute-optimal exponents under C ~ 6ND: N_opt ~ C**a, D_opt ~ C**b with
        a = beta1/(alpha1+beta1) and b = alpha1/(alpha1+beta1), read AT THE PIVOTS.
        """
        p, q = self.alpha1, self.beta1
        if p <= 0 or q <= 0:
            return float("nan"), float("nan")
        s = p + q
        return q / s, p / s


def centred(N, D, N0: float = N0_DEFAULT, D0: float = D0_DEFAULT):
    """(u, v) = (log(N/N0), log(D/D0)), the coordinates every objective here is written in."""
    return (np.log(np.asarray(N, dtype=float) / float(N0)),
            np.log(np.asarray(D, dtype=float) / float(D0)))


def objective_and_grad(theta, u, v, logL, huber):
    r"""Value and analytic gradient of sum_i Huber(log L_pred_i - log L_i), in one pass.

    Handed to L-BFGS-B via jac=True. `u`/`v` are the CENTRED coordinates. In log space the
    model is logsumexp([e, a - alpha1*u - alpha2*u**2, b - beta1*v - beta2*v**2]); with p
    the softmax over those three rows,

        d(log L_pred)/d(e)  = p_E
        d(log L_pred)/d(a)  = p_A,   d/d(alpha1) = -p_A * u,   d/d(alpha2) = -p_A * u**2
        d(log L_pred)/d(b)  = p_B,   d/d(beta1)  = -p_B * v,   d/d(beta2)  = -p_B * v**2

    and the Huber derivative is clip(s, -huber, +huber). Gradient in theta order.
    """
    e, a, alpha1, alpha2, b, beta1, beta2 = theta
    T = np.vstack([
        np.full_like(u, e),                       # log E                (constant across runs)
        a - alpha1 * u - alpha2 * u ** 2,         # log(N-term)
        b - beta1 * v - beta2 * v ** 2,           # log(D-term)
    ])
    lp = logsumexp(T, axis=0)                     # log L_pred per run, numerically stable
    s = lp - logL                                 # log-residual
    absr = np.abs(s)
    quad = np.minimum(absr, huber)                # Huber: quadratic part is min(|s|, delta)
    val = np.sum(0.5 * quad ** 2 + huber * (absr - quad))

    ds = np.clip(s, -huber, huber)                # dHuber/ds_i
    p = np.exp(T - lp)                            # p[0]=p_E, p[1]=p_A, p[2]=p_B
    grad = np.array([
        np.sum(ds * p[0]),                        # d/de
        np.sum(ds * p[1]),                        # d/da
        np.sum(ds * p[1] * (-u)),                 # d/dalpha1
        np.sum(ds * p[1] * (-u ** 2)),            # d/dalpha2
        np.sum(ds * p[2]),                        # d/db
        np.sum(ds * p[2] * (-v)),                 # d/dbeta1
        np.sum(ds * p[2] * (-v ** 2)),            # d/dbeta2
    ])
    return val, grad


def log_pred_softmax(fit: ChinchillaFit, u, v):
    """(log L_pred, p) at a fitted surface, p being the softmax weights [p_E, p_A."""
    e = np.log(fit.E) if fit.E > 0 else -np.inf   # E == 0 (Kaplan no-floor): the E term drops (p_E=0)
    T = np.vstack([np.full_like(u, e),
                   fit.a - fit.alpha1 * u - fit.alpha2 * u ** 2,
                   fit.b - fit.beta1 * v - fit.beta2 * v ** 2])
    lp = logsumexp(T, axis=0)
    return lp, np.exp(T - lp)


def check_pivots(N0, D0) -> None:
    """Pivots must be finite and positive -- they are divided into N/D and then logged."""
    for name, x in (("N0", N0), ("D0", D0)):
        if not np.isfinite(x) or x <= 0:
            raise ValueError(f"{name} must be finite and > 0; got {x!r}")


def validate_arrays(N, D, L):
    """(N, D, L) as float arrays, checked to be usable for a log-space fit."""
    N = np.asarray(N, dtype=float)
    D = np.asarray(D, dtype=float)
    L = np.asarray(L, dtype=float)
    if not (N.shape == D.shape == L.shape) or N.ndim != 1:
        raise ValueError(f"N, D, L must be 1-D arrays of equal length; "
                         f"got {N.shape}, {D.shape}, {L.shape}")
    if N.size < MIN_RUNS:
        raise ValueError(f"need at least {MIN_RUNS} runs to identify the surface; got {N.size}")
    for name, arr in (("N", N), ("D", D), ("L", L)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")
        if np.any(arr <= 0):
            raise ValueError(f"{name} must be strictly positive (we fit in log space); "
                             f"got min {arr.min()}")
    return N, D, L


def diagnostics(fit: ChinchillaFit, N, D, L) -> dict:
    """Goodness-of-fit summary in log space (the space the fit is done in)."""
    N, D, L = validate_arrays(N, D, L)
    pred = fit.predict(N, D)
    log_res = np.log(pred) - np.log(L)
    logL = np.log(L)
    n_exp, d_exp = fit.compute_optimal_exponents
    return {"rmse_log": float(np.sqrt(np.mean(log_res ** 2))),
            "r2_log": float(1.0 - np.sum(log_res ** 2) / np.sum((logL - logL.mean()) ** 2)),
            "max_rel_err": float(np.max(np.abs(pred - L) / L)),
            "N_exponent": n_exp, "D_exponent": d_exp}


def check_fix_E(fix_E, L) -> None:
    """Validate fix_E: None (fit it), 0 (Kaplan no-floor), or 0 < E < min(L)."""
    if fix_E is None:
        return
    if fix_E < 0:
        raise ValueError(f"fix_E must be None, 0 (Kaplan no-floor), or >0; got {fix_E}")
    if fix_E > 0 and fix_E >= float(np.min(L)):
        raise ValueError(f"fix_E={fix_E} must be < min(L)={float(np.min(L)):.4g} (E is the loss floor)")


def check_fix_alpha2(fix_alpha2) -> None:
    """Validate fix_alpha2: None (fit it) or a finite constant (0.0 = Chinchilla)."""
    if fix_alpha2 is None:
        return
    if not np.isfinite(fix_alpha2):
        raise ValueError(f"fix_alpha2 must be None or finite; got {fix_alpha2}")


def check_fix_beta2(fix_beta2) -> None:
    """Validate fix_beta2: None (fit it) or a finite constant (0.0 = Chinchilla)."""
    if fix_beta2 is None:
        return
    if not np.isfinite(fix_beta2):
        raise ValueError(f"fix_beta2 must be None or finite; got {fix_beta2}")


def default_pivot(x) -> float:
    """The geometric mean of `x` -- the data-driven pivot, offered but not the default."""
    return float(np.exp(np.mean(np.log(np.asarray(x, dtype=float)))))
