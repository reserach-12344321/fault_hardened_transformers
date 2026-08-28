r"""Load scaling-law training data -- the heavy raw path and the light processed path.

RAW (needs JAX): load_sweep turns each completed run of a sweep dir into an EvalResult by
deserialising its checkpoint. process_training_runs runs it once to write the JSONs.

PROCESSED (light): three loaders read those back, each slicing the (train fault) x (eval
fault) grid differently -- load_eval_results by eval condition (pooling training recipes),
load_matched_results down the p_eval == p_train diagonal, load_fixed_ptrain_results along a
row of fixed p_train -- and eval_cohort_arrays turns a cohort into the (N, D, L) arrays the
fit consumes. One EvalResult carries several eval conditions, so cohort membership is per
MEASUREMENT, not per record. The loaders never drop anything; trimming is filter_cohorts.
"""
from __future__ import annotations

import contextlib
import glob
import io
import json
import math
import os

# CPU by default: counting params only needs the checkpoint in host memory, and the
# workstation GPU is shared with live training. setdefault, so a caller can force the GPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from nano_llama.eval_result import EvalPoint, EvalResult

# ============================================================================================
# PROCESSED EvalResult JSONs -> cohorts and fit arrays  (light: no checkpoints)
# ============================================================================================
def _gather_json(paths) -> list[str]:
    """A dir, a file, or an iterable of them -> a sorted list of *.json files."""
    if isinstance(paths, str):
        paths = [paths]
    files: list[str] = []
    for path in paths:
        if os.path.isdir(path):
            files.extend(glob.glob(os.path.join(path, "**", "*.json"), recursive=True))
        elif os.path.isfile(path):
            files.append(path)
        else:
            raise FileNotFoundError(f"no such file or directory: {path}")
    return sorted(files)


def load_eval_results(paths, *, sort: bool = True
                      ) -> dict[tuple[float, float], list[EvalPoint]]:
    """Load processed EvalResult JSONs, grouped by (k_eval, p_eval)."""
    cohorts: dict[tuple[float, float], list[EvalPoint]] = {}
    for fp in _gather_json(paths):
        for point in EvalResult.load(fp).points():
            cohorts.setdefault((point.k_eval, point.p_eval), []).append(point)
    if sort:
        for key in cohorts:
            cohorts[key].sort(key=lambda r: r.eval_loss)
    return cohorts


def load_matched_results(paths, *, k=None, sort: bool = True
                         ) -> dict[float, list[EvalPoint]]:
    """Load processed EvalResult JSONs keeping only each model's MATCHED condition."""
    matched, missing, ks = [], [], set()
    for fp in _gather_json(paths):
        result = EvalResult.load(fp)
        ks.add(float(result.k_train))
        hit = next((pt for pt in result.points()
                    if float(pt.k_eval) == float(result.k_train)
                    and float(pt.p_eval) == float(result.p_train)), None)
        (matched if hit is not None else missing).append(hit if hit is not None else fp)
    if missing:
        raise ValueError(
            f"{len(missing)} record(s) carry no (k_eval, p_eval) == (k_train, p_train) point, so they "
            f"were never scored at their own fault condition; first: {missing[0]}")
    if k is None:
        if len(ks) != 1:
            raise ValueError(
                f"data spans k_train values {sorted(ks)}; pass k=<value> to choose one. Pooling "
                f"different fault block sizes into one scaling-law fit is not meaningful.")
        k = next(iter(ks))

    cohorts: dict[float, list[EvalPoint]] = {}
    for pt in matched:
        if float(pt.k_train) == float(k):
            cohorts.setdefault(float(pt.p_train), []).append(pt)
    if not cohorts:
        raise ValueError(f"no records with k_train == {k}; available: {sorted(ks)}")
    if sort:
        for key in cohorts:
            cohorts[key].sort(key=lambda r: r.eval_loss)
    return cohorts


def load_fixed_ptrain_results(paths, *, p_train: float = 0.0, k=None, sort: bool = True
                              ) -> dict[float, list[EvalPoint]]:
    """Load processed EvalResult JSONs for ONE p_train, grouped by p_eval.

    A ROW of the eval grid where load_matched_results takes the diagonal: every cohort holds
    the SAME models, one training recipe, re-scored at a different eval fault. So this asks
    how a model trained at p_train degrades as the fault it is RUN UNDER grows.

    The cohorts are the same checkpoints scored repeatedly, so they are NOT independent --
    any cross-cohort uncertainty must resample MODELS, not each cohort's rows. They can also
    be RAGGED, since every job carries its own baselines; nothing is dropped, so check the
    sizes and ignore the thin ones.

    `k` applies to both sides. `sort=False` keeps record order, making the cohorts
    INDEX-ALIGNED when the row is rectangular, so cohorts[p][i] is the same model for every
    p -- any analysis indexing ACROSS cohorts needs it, since the loss ranking differs at
    every eval fault.
    """
    selected, all_p, ks = [], set(), set()
    for fp in _gather_json(paths):
        result = EvalResult.load(fp)
        all_p.add(float(result.p_train))
        if float(result.p_train) == float(p_train):
            ks.add(float(result.k_train))
            selected.append(result)
    if not selected:
        raise ValueError(f"no records with p_train == {p_train}; available: {sorted(all_p)}")
    if k is None:
        if len(ks) != 1:
            raise ValueError(
                f"records at p_train == {p_train} span k_train values {sorted(ks)}; pass k=<value> to "
                f"choose one. Pooling different fault block sizes into one scaling-law fit is not "
                f"meaningful.")
        k = next(iter(ks))

    cohorts: dict[float, list[EvalPoint]] = {}
    for result in selected:
        if float(result.k_train) != float(k):
            continue
        for pt in result.points():
            if float(pt.k_eval) == float(k):
                cohorts.setdefault(float(pt.p_eval), []).append(pt)
    if not cohorts:
        raise ValueError(f"no points with k_train == k_eval == {k} at p_train == {p_train}; "
                         f"k_train values present: {sorted(ks)}")
    if sort:
        for key in cohorts:
            cohorts[key].sort(key=lambda r: r.eval_loss)
    return cohorts


def _ne_frac(pt: EvalPoint) -> float:
    """The fraction of a run's parameters that are not the token embedding, in (0, 1]."""
    total = float(pt.total_n_params)
    if total <= 0:
        raise ValueError(
            f"min_ne_frac needs a positive total_n_params, but a record has {total} -- regenerate "
            f"the processed dir with process_eval_arrays.py so the param counts are populated")
    return float(pt.n_non_embedding_params) / total


def filter_cohorts(cohorts: dict, *, max_ceiling_frac: float | None = 0.70,
                   min_tpp: float | None = None,
                   min_ne_frac: float | None = None) -> tuple[dict, dict]:
    """Drop unusable runs from already-loaded cohorts, keeping list order.

    An analysis choice, deliberately kept out of the loaders, whose promise that nothing is
    dropped is what makes their cohort sizes a readable inventory. Operates on the cohort
    LISTS, not eval_cohort_arrays' output, because callers rely on cohorts[key][i] being the
    model behind arrays[...][i].

    TWO KINDS OF CRITERION. `max_ceiling_frac` cuts on the RESPONSE -- keep runs with
    eval_loss < frac * ln(vocab_size), the vocab read per run. Defensible only where it
    removes runs that are demonstrably unusable, and it must not be pushed further to tidy
    residuals: a heavier-fault cohort sits higher in loss at the same (N, D), so it would
    drop different runs from different cohorts and bias the cross-cohort comparison the
    matched diagonal exists for. `min_tpp` (tokens per TOTAL parameter) and `min_ne_frac`
    (non-embedding parameter share) cut on the DESIGN, which is a statement about which runs
    belong in the population. Hence the ceiling is on by default and the other two are not.

    TPP IS ALWAYS AGAINST TOTAL PARAMETERS: gen_sweep builds every horizon as
    D = tok_per_param * total params, so the rungs are defined in those units and no other.

    Every criterion uses the same threshold for every cohort, so there is no per-cohort
    override, and a run must pass all the active ones.

    CAVEAT: the fitted beta1 moves by up to 0.30 between reasonable screens, two to three
    times its bootstrap CI, so the screen dominates the D-term's uncertainty. Report which
    one was used, and treat a D-term conclusion that survives only one setting as unproven.

    Returns (filtered, report); `by_reason` blames the FIRST criterion a run failed.
    Reporting is not optional -- the drop is deliberately uneven across cohorts, and an
    unreported filter reads as "this is the whole array" when it is not. Raises if a filter
    empties a cohort outright, rather than silently losing a rung.
    """
    if max_ceiling_frac is not None and not 0.0 < float(max_ceiling_frac) <= 1.0:
        raise ValueError(f"max_ceiling_frac must be in (0, 1]; got {max_ceiling_frac!r}. It is a "
                         f"FRACTION of ln(vocab_size), not a loss in nats.")
    if min_ne_frac is not None and not 0.0 < float(min_ne_frac) < 1.0:
        raise ValueError(f"min_ne_frac must be in (0, 1); got {min_ne_frac!r}. It is the FRACTION of "
                         f"a model's parameters that are NOT the token embedding, not a parameter "
                         f"count -- 1.0 would drop every run (no model is 100% non-embedding).")
    if min_tpp is not None and float(min_tpp) <= 0.0:
        raise ValueError(f"min_tpp must be > 0; got {min_tpp!r}. It is a floor on tokens per TOTAL "
                         f"parameter.")

    def _tpp(pt: EvalPoint) -> float:
        """Tokens per TOTAL parameter."""
        n = float(pt.total_n_params)
        return float(pt.n_train_tokens) / n if n > 0 else float("nan")

    def _reject(pt: EvalPoint) -> str | None:
        """The FIRST criterion this run fails, or None if it passes them all."""
        if max_ceiling_frac is not None:
            vocab = int(pt.model_config.vocab_size)          # per run: a mixed-tokenizer array is fine
            if float(pt.eval_loss) >= float(max_ceiling_frac) * math.log(vocab):
                return "ceiling"
        # Before the duration cut: attributing a drop to "the model is mostly embedding" is
        # more informative than to a duration floor it also happens to fail.
        if min_ne_frac is not None and _ne_frac(pt) < float(min_ne_frac):
            return "min_ne_frac"
        if min_tpp is not None:
            n = float(pt.total_n_params)
            if n <= 0:
                raise ValueError(
                    f"a tpp screen needs a positive total_n_params, but a record has {n} -- "
                    f"regenerate the processed dir with process_eval_arrays.py so the param counts "
                    f"are populated")
            if float(pt.n_train_tokens) / n < float(min_tpp):
                return "min_tpp"
        return None

    filtered, report, emptied = {}, {}, []
    for key, points in cohorts.items():
        verdicts = [(pt, _reject(pt)) for pt in points]
        kept = [pt for pt, why in verdicts if why is None]   # list order preserved
        if points and not kept:
            emptied.append(key)
        filtered[key] = kept
        by_reason = {}
        for _, why in verdicts:
            if why is not None:
                by_reason[why] = by_reason.get(why, 0) + 1
        report[key] = {
            "n_before": len(points), "n_after": len(kept), "n_dropped": len(points) - len(kept),
            "by_reason": by_reason,
            "L_max_before": max((float(p.eval_loss) for p in points), default=float("nan")),
            "L_max_after": max((float(p.eval_loss) for p in kept), default=float("nan")),
            # Reported whether or not the tpp screen is active: this array trains different
            # architectures over different TPP windows, so the span is part of reading a fit.
            "tpp_min_before": min((_tpp(p) for p in points), default=float("nan")),
            "tpp_min_after": min((_tpp(p) for p in kept), default=float("nan")),
            "tpp_max_before": max((_tpp(p) for p in points), default=float("nan")),
            "tpp_max_after": max((_tpp(p) for p in kept), default=float("nan")),
            # Most embedding-dominated model in the cohort, reported whether or not the
            # screen is on: it is where N is least well defined.
            "ne_frac_min_before": min((_ne_frac(p) for p in points), default=float("nan")),
            "ne_frac_min_after": min((_ne_frac(p) for p in kept), default=float("nan")),
        }
    if emptied:
        raise ValueError(
            f"the filter (max_ceiling_frac={max_ceiling_frac}, min_tpp={min_tpp}, "
            f"min_ne_frac={min_ne_frac}) emptied cohort(s) "
            f"{sorted(emptied)}; loosen it rather than fitting a ladder that is silently missing a rung")
    return filtered, report


def print_filter_report(report: dict, *, max_ceiling_frac: float | None = 0.70,
                        min_tpp: float | None = None,
                        min_ne_frac: float | None = None) -> None:
    """Print what filter_cohorts removed, per cohort."""
    actives = (max_ceiling_frac, min_tpp, min_ne_frac)
    if all(a is None for a in actives):
        print("run filter: OFF (no ceiling / min_tpp / min_ne_frac) -- every loaded run is kept")
    else:
        crit = ", ".join(s for s in (
            f"L < {max_ceiling_frac:g} ln(vocab)" if max_ceiling_frac is not None else "",
            f"tokens/total-param >= {min_tpp:g}" if min_tpp is not None else "",
            f"non-emb params >= {min_ne_frac:g}" if min_ne_frac is not None else "") if s)
        total = sum(r["n_dropped"] for r in report.values())
        tally = {}
        for r in report.values():
            for why, n in r["by_reason"].items():
                tally[why] = tally.get(why, 0) + n
        detail = f"  ({', '.join(f'{n} by {w}' for w, n in sorted(tally.items()))})" if tally else ""
        print(f"run filter: keep {crit}  ->  dropped {total} run(s) across "
              f"{len(report)} cohort(s){detail}")

    hdr = (f"{'cohort':>9}{'kept':>7}{'dropped':>9}{'TPP range':>30}{'L max':>20}"
           f"{'ne_frac min':>20}")
    print(hdr); print("-" * len(hdr))
    for key in sorted(report):
        r = report[key]
        label = f"{key:g}" if isinstance(key, (int, float)) else str(key)
        t_col = (f"{r['tpp_min_before']:.3g}-{r['tpp_max_before']:.3g} -> "
                 f"{r['tpp_min_after']:.3g}-{r['tpp_max_after']:.3g}"
                 if "tpp_min_before" in r else "n/a")
        l_col = f"{r['L_max_before']:.3f} -> {r['L_max_after']:.3f}"
        # .get-style guard so a report built before this column existed still prints.
        n_col = (f"{r['ne_frac_min_before']:.3f} -> {r['ne_frac_min_after']:.3f}"
                 if "ne_frac_min_before" in r else "n/a")
        print(f"{label:>9}{r['n_after']:>7}{r['n_dropped']:>9}{t_col:>30}{l_col:>20}{n_col:>20}")


def eval_cohort_arrays(results: list[EvalPoint]) -> dict:
    """A cohort of EvalPoint -> the (N, D, L) arrays the fit consumes."""
    import numpy as np
    return {
        "N": np.array([r.total_n_params for r in results], dtype=float),
        "N_ne": np.array([r.n_non_embedding_params for r in results], dtype=float),
        "D": np.array([r.n_train_tokens for r in results], dtype=float),
        "L": np.array([r.eval_loss for r in results], dtype=float),
        "size": np.array([r.size_key for r in results]),
    }
