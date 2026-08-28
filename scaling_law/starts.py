"""Where the multi-start begins: centroid-anchored initialisations.

The amplitudes are not free quantities to grid over -- their scale is set by the slopes and
by where the data sits. So grid the dimensionless ones (slopes, and the SPLIT of the loss
between terms at the data centroid) and derive the amplitudes by making the starting surface
pass through that centroid; every start is then a plausible surface.

Everything works in the CENTRED coordinates, and the pivots are fixed constants rather than
the data centroid, so mean(u) is generally non-zero and the amplitude formulas carry the
offset. Curvatures are crossed into the grid only when free.
"""
from __future__ import annotations

import itertools

import numpy as np

# Slope levels are log-spaced in magnitude: they span an order of magnitude in practice, so
# linear spacing would waste resolution at the shallow end where they live.
ALPHA1S = (0.05, 0.2, 0.8)
BETA1S = (0.05, 0.2, 0.8)
SPLITS = (0.25, 0.5, 0.75)      # s_A; s_B is the remainder (times 1 - s_E when E is free)
E_SHARES = (0.05, 0.35)         # free-E only -- how much of Lbar the floor starts out holding
# Curvature levels tried when alpha2 is free, 0 first so a fit that wants the Chinchilla
# shape finds it immediately.
ALPHA2S = (0.0, 0.03, -0.03)
# The same magnitudes on the D side. Deliberately the same: nothing has been measured about
# the size of a D-curvature, and a different bracket would imply a measurement that does
# not exist.
BETA2S = (0.0, 0.03, -0.03)

# The coarse set, for per-resample fits deliberately not warm-started -- pass it as
# bootstrap(starts_boot=...) when n_rescued says the warm start can no longer be trusted.


def start_from_shares(mL, mu, mv, alpha1, beta1, s_A, s_B, s_E, alpha2=0.0,
                      beta2=0.0) -> np.ndarray:
    """One start theta whose surface passes through the data centroid (mu, mv."""
    return np.array([np.log(s_E) + mL if s_E > 0 else -np.inf,
                     np.log(s_A) + mL + alpha1 * mu + alpha2 * mu ** 2,
                     alpha1,
                     alpha2,
                     np.log(s_B) + mL + beta1 * mv + beta2 * mv ** 2,
                     beta1,
                     beta2])


def _centroid(u, v, logL) -> tuple[float, float, float]:
    return float(np.mean(logL)), float(np.mean(u)), float(np.mean(v))


def _shares(fix_E, splits, e_shares) -> list[tuple[float, float, float]]:
    """(s_A, s_B, s_E) triples summing to 1."""
    if fix_E is not None:
        return [(s, 1.0 - s, 0.0) for s in splits]
    return [((1 - sE) * s, (1 - sE) * (1 - s), sE) for sE in e_shares for s in splits]


def ols_start(u, v, logL, fix_E, fix_alpha2=0.0, fix_beta2=0.0) -> np.ndarray:
    """A single start read straight off the data -- candidate #1, and free."""
    X = np.column_stack([np.ones_like(u), u, v])
    _c0, c1, c2 = np.linalg.lstsq(X, logL, rcond=None)[0]
    s_E = 0.0 if fix_E is not None else 0.1
    s_A = s_B = (1.0 - s_E) / 2.0
    alpha1 = float(np.clip(-c1 / s_A, 1e-3, 5.0))      # undo the softmax-weight shrinkage
    beta1 = float(np.clip(-c2 / s_B, 1e-3, 5.0))
    # A free curvature starts flat: the regression is linear in u and v, so it says nothing
    # about a quadratic.
    return start_from_shares(*_centroid(u, v, logL), alpha1, beta1, s_A, s_B, s_E,
                             0.0 if fix_alpha2 is None else fix_alpha2,
                             0.0 if fix_beta2 is None else fix_beta2)


def centroid_starts(u, v, logL, fix_E, fix_alpha2=0.0, fix_beta2=0.0, *, alpha1s=ALPHA1S,
                    beta1s=BETA1S, splits=SPLITS, e_shares=E_SHARES, alpha2s=ALPHA2S,
                    beta2s=BETA2S) -> list[np.ndarray]:
    """The centroid-anchored grid over (alpha1, beta1, split)."""
    mL, mu, mv = _centroid(u, v, logL)
    a2_levels = alpha2s if fix_alpha2 is None else (fix_alpha2,)
    b2_levels = beta2s if fix_beta2 is None else (fix_beta2,)
    return [start_from_shares(mL, mu, mv, al, be, s_A, s_B, s_E, a2, b2)
            for al, be, (s_A, s_B, s_E), a2, b2
            in itertools.product(alpha1s, beta1s, _shares(fix_E, splits, e_shares), a2_levels,
                                 b2_levels)]


def default_starts(u, v, logL, fix_E, fix_alpha2=0.0, fix_beta2=0.0) -> list[np.ndarray]:
    """The OLS start followed by the centroid grid -- the multi-start set for a point estimate."""
    return ([ols_start(u, v, logL, fix_E, fix_alpha2, fix_beta2)]
            + centroid_starts(u, v, logL, fix_E, fix_alpha2, fix_beta2))
