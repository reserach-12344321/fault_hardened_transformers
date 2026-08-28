r"""Load the fault-marginalised predictive distributions the logit-sample worker writes.

    p_bar(. | context) = E_chip[ softmax(z(context, chip)) ]     (softmax THEN average)

for a fixed context set at each eval fault (k, p). Exists to test whether the fault acts on
the predictive like a TEMPERATURE: fit_temperature finds the single global T minimising
KL(p_bar || softmax(z0/T)) and reports how much of the effect it explains.

One job holds base.npz, one point_<i>.npz per (k, p_eval), and a digest saying which points
are complete. load_array takes an experiment root and pools its waves. Group arrays are
large, so each point is reduced to p_bar and a noise floor ON LOAD. JAX-free.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

import numpy as np


RESULTS_DIR = "results"
RESULTS_BASE = os.path.join(RESULTS_DIR, "base.npz")
RESULTS_JSON = os.path.join(RESULTS_DIR, "logit_marginals.json")
DONE_MARKER = os.path.join(RESULTS_DIR, "DONE")

# ======================================================================================
# Model-shape helpers
# ======================================================================================
# Pure functions of a model_config dict.
def non_embedding_params(mc: dict) -> int:
    """Non-embedding parameter count from a model_config dict, exact for this Llama variant."""
    d, L, m = int(mc["n_embd"]), int(mc["n_layer"]), int(mc["multiple_of"])
    ffn_hidden = m * ((int(2 * (4 * d) / 3) + m - 1) // m)          # SwiGLU hidden (matches the model)
    return L * (4 * d * d + 3 * d * ffn_hidden + 2 * d) + d


def size_key(mc: dict) -> str:
    """``d{n_embd:04d}_L{n_layer:02d}`` -- the label EvalResult.size_key produces."""
    return f"d{int(mc['n_embd']):04d}_L{int(mc['n_layer']):02d}"


def point_path(job_dir: str, i: int) -> str:
    """Path to point i's npz. Mirrors logit_sample_worker.point_name -- keep the two in step."""
    return os.path.join(job_dir, RESULTS_DIR, f"point_{int(i):03d}.npz")


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def kl_rows(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """KL(p || q) per row (last axis), in nats. p, q are (..., V) probability arrays."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)



@dataclass
class MarginalPoint:
    """The fault-marginalised predictive at one ``(k, p_eval)`` over the fixed context set."""
    k: int
    p: float
    index: int
    p_bar: np.ndarray                    # (C, V)  the marginal predictive (mean of the K groups)
    noise_floor_kl: float                # split-group KL: the Monte-Carlo floor on any misfit
    digest: dict = field(repr=False)     # the worker's own summary for this point
    groups: np.ndarray | None = field(default=None, repr=False)   # (K, C, V) if keep_groups
    raw: np.ndarray | None = field(default=None, repr=False)      # (n_raw, V) single chips, one context


@dataclass
class MarginalRun:
    """One trained checkpoint's marginal predictives over the shared contexts, at every eval fault."""
    job: str
    size_key: str
    n_params: int
    n_non_embedding: int
    n_train_tokens: int
    p_train: float
    k_train: float
    n_chips: int
    n_groups: int
    context_seed: int
    context_stream: int
    clean_logits: np.ndarray             # (C, V) = z0
    targets: np.ndarray                  # (C,)
    points: dict                         # p_eval -> MarginalPoint
    header: dict = field(repr=False)

    @property
    def tokens_per_param(self) -> float:
        """Tokens per TOTAL parameter -- the unit the sweep was designed in."""
        return self.n_train_tokens / self.n_params

    @property
    def arm(self) -> str:
        return "clean" if self.p_train == 0.0 else f"p_train={self.p_train:g}"

    @property
    def clean_probs(self) -> np.ndarray:
        return softmax(self.clean_logits)

    @property
    def clean_loss(self) -> float:
        """L_0: the mean clean cross-entropy over the context set, in nats -- the natural per-run
        scale for the fault's KL, so dividing a KL curve by it is a pure rescale.
        """
        p = self.clean_probs[np.arange(self.targets.shape[0]), self.targets]
        return float(np.mean(-np.log(np.clip(p, 1e-12, 1.0))))

    @property
    def p_eval(self) -> list:
        return sorted(self.points)


def load_run(job_dir: str, *, keep_groups: bool = False, keep_raw: bool = True) -> MarginalRun:
    """Load one marginal-predictive job. Raises if it is missing or has no points yet."""
    base_path, json_path = os.path.join(job_dir, RESULTS_BASE), os.path.join(job_dir, RESULTS_JSON)
    for pth in (base_path, json_path):
        if not os.path.isfile(pth):
            raise FileNotFoundError(f"{job_dir}: missing {os.path.relpath(pth, job_dir)}")
    with open(json_path) as f:
        doc = json.load(f)
    header = {k: v for k, v in doc.items() if k != "points"}
    with open(os.path.join(job_dir, "final_model.json")) as f:
        mc = json.load(f)

    with np.load(base_path) as zb:
        clean_logits = np.asarray(zb["clean_logits"], dtype=np.float32)
        targets = np.asarray(zb["targets"])
    points: dict = {}
    for d in doc["points"]:
        i, p = int(d["index"]), float(d["p"])
        if d.get("is_clean"):            # p=0 stub (should not occur -- prep drops p=0) -> the clean pass
            continue
        pth = point_path(job_dir, i)
        if not os.path.isfile(pth):
            continue                     # digest without its file: a death between the two writes
        with np.load(pth) as z:
            g = np.asarray(z["groups"], dtype=np.float64)               # (K, C, V)
            raw = np.asarray(z["raw"], dtype=np.float32) if (keep_raw and "raw" in z.files) else None
        pbar = g.mean(axis=0).astype(np.float32)                        # (C, V)
        half = g.shape[0] // 2
        floor = float(np.mean(kl_rows(g[:half].mean(0), g[half:].mean(0)))) if half >= 1 else float("nan")
        points[p] = MarginalPoint(k=int(d["k"]), p=p, index=i, p_bar=pbar, noise_floor_kl=floor,
                                  digest=d, groups=(g.astype(np.float32) if keep_groups else None),
                                  raw=raw)
    if not points:
        raise ValueError(f"{job_dir}: no completed (k, p) points yet")

    return MarginalRun(
        job=os.path.basename(job_dir.rstrip("/")),
        size_key=size_key(mc),
        n_params=int(header["n_params"]),
        n_non_embedding=non_embedding_params(mc),
        n_train_tokens=int(header["n_train_tokens"]),
        p_train=float(header["p_train"]),
        k_train=float(header["k_train"]),
        n_chips=int(header["n_chips"]),
        n_groups=int(header["n_groups"]),
        context_seed=int(header["context_seed"]),
        context_stream=int(header["context_stream"]),
        clean_logits=clean_logits,
        targets=targets,
        points=points,
        header=header,
    )


def logit_array_dirs(paths) -> list:
    """Expand experiment root(s) into their waves; pass plain array dirs through unchanged."""
    from experiment_util.prepare_eval_array import wave_dirs   # local: keeps this module import-light
    out = []
    for path in ([paths] if isinstance(paths, str) else list(paths)):
        out.extend(wave_dirs(path) or [path])
    return out


def load_array(array_dirs, *, require_done: bool = False, **kw) -> list:
    """Load every job with at least one completed point, sorted by (N, D, p_train).

    `array_dirs` is an experiment root (pooling all its waves), a single wave or plain array
    dir, or an iterable of either.

    `require_done` defaults to False so a still-running array can be inspected as it fills in;
    a job with some points written is loaded with those points. When pooling a root that mixes
    finished models with partially-scored ones, pass True.

    Two things are re-checked when pooling, both silent corruptions otherwise:

      * DUPLICATE JOB NAMES. Callers key results by run.job, so two runs sharing a name would
        overwrite one another. Waves are disjoint by construction; this catches when they are not.
      * CONTEXT IDENTITY. Every run's p_bar is indexed by position in a fixed context set drawn
        from (context_seed, context_stream). Runs from different seeds are not comparable
        per-context, which is exactly what the temperature analysis does, so a mismatch raises.
    """
    dirs = logit_array_dirs(array_dirs)
    runs = []
    for array_dir in dirs:
        for d in sorted(glob.glob(os.path.join(array_dir, "*"))):
            if not (os.path.isdir(d) and os.path.isfile(os.path.join(d, "final_model.json"))):
                continue
            if require_done and not os.path.isfile(os.path.join(d, DONE_MARKER)):
                continue
            try:
                runs.append(load_run(d, **kw))
            except (FileNotFoundError, ValueError):
                continue                 # not started / no points yet -- skip silently while running
    if not runs:
        raise FileNotFoundError(
            f"no jobs with completed points under {dirs if len(dirs) > 1 else dirs[0]!r}"
            + (f" (expanded from {array_dirs!r})" if dirs != [array_dirs] else ""))

    seen = {}
    for r in runs:
        if r.job in seen:
            raise ValueError(
                f"job name {r.job!r} appears in more than one array dir; results are keyed by job "
                f"name, so pooling these would silently drop one of them")
        seen[r.job] = True

    ctx = {(r.context_seed, r.context_stream, int(r.clean_logits.shape[0])) for r in runs}
    if len(ctx) != 1:
        raise ValueError(
            f"the loaded runs do not share one context set -- (context_seed, context_stream, "
            f"n_contexts) takes {len(ctx)} distinct values {sorted(ctx)}. p_bar is indexed by "
            f"position in that set, so these runs are not comparable per-context. Load them "
            f"separately, or restage the odd wave under the campaign's spec.")

    runs.sort(key=lambda r: (r.n_params, r.n_train_tokens, r.p_train))
    return runs


# THE TEMPERATURE MODEL: is the fault z -> z/T ?  One GLOBAL T per (run, fault), minimising
#     T* = argmin_T  sum_c KL(p_bar_c || softmax(z0_c / T)).
# The joint model is a one-parameter exponential family in beta = 1/T, so T* is the exact
# moment-matching point sum_c E_{softmax(z0_c/T*)}[z0_c] = sum_c E_{p_bar_c}[z0_c], monotone
# in beta -- a 1-D solve. Everything downstream comes from that T and p_T = softmax(z0/T):
#     total_c     = KL(p_bar_c || clean_c)      the fault's whole effect on context c
#     residual_c  = KL(p_bar_c || p_T,c)        what the temperature model misses
#     explained_c = KL(p_T,c   || clean_c)      what it accounts for
# T* is the joint M-projection, so mean(total) = mean(residual) + mean(explained) EXACTLY,
# making the explained fraction a true decomposition.
DEFAULT_T_GRID = np.geomspace(0.25, 100.0, 28)      # coarse -- only a start for the Newton solve


def moment_curve(clean_logits, T_grid=DEFAULT_T_GRID):
    """(len(T_grid),): sum_c E_{softmax(z0_c/T)}[z0_c] for each T. Monotone in T."""
    z0 = np.asarray(clean_logits, np.float64)
    return np.array([(softmax(z0 / T) * z0).sum() for T in T_grid])


def fits_for_run(run, *, T_grid=DEFAULT_T_GRID, n_newton=8) -> dict:
    """Fit every eval fault of ONE run -> ``{p_eval: fit_temperature(...) dict}``."""
    z0 = np.asarray(run.clean_logits, np.float64)
    pclean = softmax(z0)
    curve = moment_curve(z0, T_grid)
    return {p: fit_temperature(run.points[p].p_bar, z0, T_grid=T_grid, curve=curve,
                               n_newton=n_newton, z0=z0, pclean=pclean)
            for p in run.p_eval}


def fit_temperature(p_bar, clean_logits, *, T_grid=DEFAULT_T_GRID, curve=None, n_newton=8,
                    z0=None, pclean=None):
    """Fit one global T for this (run, fault) and decompose the fault's effect with it.

    Returns a dict:
        T          the effective temperature for the whole run at this fault
        total      (C,) KL(p_bar_c || clean_c)   -- the fault's whole effect, per context
        residual   (C,) KL(p_bar_c || p_T,c)     -- what the global temperature misses
        explained  (C,) KL(p_T,c   || clean_c)   -- what it accounts for
        frac       sum(explained)/sum(total), the exact Pythagorean ratio
    All KL in nats, with mean(total) == mean(residual) + mean(explained).

    `curve`, `z0` and `pclean` are per-run invariants this otherwise rebuilds on every call.
    Passing them is bit-identical and simply skips the work, which is worth it across a run's
    many faults; fits_for_run hoists all three.
    """
    z0 = np.asarray(clean_logits, np.float64) if z0 is None else z0
    pb = np.asarray(p_bar, np.float64)
    if curve is None:
        curve = moment_curve(z0, T_grid)
    Tg = np.asarray(T_grid, np.float64)
    m = float((pb * z0).sum())                          # target: sum_c E_p_bar_c[z0_c]
    beta = 1.0 / float(np.interp(m, curve[::-1], Tg[::-1]))          # start from the curve inverse
    blo, bhi = 1.0 / Tg.max(), 1.0 / Tg.min()
    for _ in range(n_newton):                           # Newton on the summed moment (d/dbeta = sum Var)
        pbeta = softmax(beta * z0)
        Ec = (pbeta * z0).sum(1)
        Ez = float(Ec.sum())
        Vsum = float((pbeta * z0 * z0).sum() - (Ec * Ec).sum())
        prev, beta = beta, min(max(beta + (m - Ez) / max(Vsum, 1e-9), blo), bhi)
        # Exact fixpoint: the update is a pure function of beta, so every remaining iteration
        # would reproduce this one bit-for-bit. Breaking is not an approximation -- it returns
        # the identical T the full sweep would. Where Newton instead dithers in the last ulp
        # the test never fires and the full sweep runs.
        if beta == prev:
            break
    T = float(1.0 / beta)
    pT = softmax(z0 / T)
    pclean = softmax(z0) if pclean is None else pclean
    total = kl_rows(pb, pclean)
    residual = kl_rows(pb, pT)
    explained = kl_rows(pT, pclean)
    return dict(T=T, total=total, residual=residual, explained=explained,
                frac=float(explained.sum() / total.sum()) if total.sum() > 0 else float("nan"))
