"""Fit the scaling laws (including uncertainty estimates) and dump to disk

Load the processed EvalResult JSONs, screen them, build each p_train cohort's (N, D, L)
arrays, fit and bootstrap.
"""
from __future__ import annotations

import os
import datetime

import numpy as np

import scaling_law as est
from scaling_law import data_loading as dl
from scaling_law import fit_store


def fit_matched(processed_dir, *, k=None, max_ceiling_frac=0.70,
                fix_alpha2=0.0, fix_beta2=0.0, base_fix_alpha2="inherit",
                N0=est.N0_DEFAULT, D0=est.D0_DEFAULT,
                min_tpp=None, min_ne_frac=None,
                fix_E=0.0, bootstrap=None, n_jobs=-1, verbose=True):
    """Load -> screen -> fit -> bootstrap. Returns (analyses, meta)."""
    # Validate BEFORE the load + fit: a bad spec should cost a millisecond, not ten minutes.
    boot_spec = (bootstrap or fit_store.BootstrapSpec()).validate()

    # ---- load the matched diagonal, then screen it -------------------------------------------------
    cohorts = dl.load_matched_results(processed_dir, k=k)
    k_used = float(cohorts[next(iter(cohorts))][0].k_train)
    # Applied to the cohort LISTS, before any array is derived, so cohorts[p][i] stays the
    # model behind arr[...][i] -- the alignment the eval_se columns below depend on.
    cohorts, filt = dl.filter_cohorts(cohorts, max_ceiling_frac=max_ceiling_frac,
                                      min_tpp=min_tpp, min_ne_frac=min_ne_frac)
    if verbose:
        dl.print_filter_report(filt, max_ceiling_frac=max_ceiling_frac, min_tpp=min_tpp,
                               min_ne_frac=min_ne_frac)
        print()

    # ---- per-cohort arrays ------------------------------------------------------------------------
    cohort_arr, inventory = {}, []
    for p in sorted(cohorts):
        # arr["N"] is the TOTAL parameter count, what the surface is fit against; arr["N_ne"]
        # rides along for the plots.
        arr = dl.eval_cohort_arrays(cohorts[p])
        # Each run's MEASURED standard error -- the noise floor the residuals get compared
        # against. Carried in the artifact because the plots will not have the EvalPoints.
        arr["eval_se"] = np.array([r.eval_se for r in cohorts[p]], dtype=float)
        cohort_arr[p] = arr
        inventory.append(dict(p_train=float(p), n=len(cohorts[p]),
                              n_distinct_N=len(set(arr["N"].tolist())),
                              n_distinct_D=len(set(arr["D"].tolist())),
                              L_min=float(arr["L"].min()), L_max=float(arr["L"].max()),
                              D_min=float(arr["D"].min()), D_max=float(arr["D"].max()),
                              eval_se_median=float(np.median(arr["eval_se"])),
                              eval_se_max=float(arr["eval_se"].max()),
                              n_short_of_se_target=int(sum(not r.reached_se_target
                                                           for r in cohorts[p]))))

    if verbose:
        _print_inventory(inventory, k_used)

    # ---- fit + bootstrap --------------------------------------------------------------------------
    analyses = {}
    if verbose:
        print(f"\nbootstrapping: {boot_spec.label()}")
    base_p = min(cohort_arr) if cohort_arr else None
    if base_fix_alpha2 != "inherit" and verbose:
        _a2 = ("free" if base_fix_alpha2 is None else f"{base_fix_alpha2:g}")
        print(f"\nbase cohort p_train = {base_p:g}: alpha2 overridden to {_a2} "
              f"(other cohorts: {'free' if fix_alpha2 is None else f'{fix_alpha2:g}'})")
    for i, p in enumerate(sorted(cohort_arr)):            # distinct seed per cohort -> independence
        a = cohort_arr[p]
        # The override applies to the reference cohort only, and to its bootstrap as well as
        # its point fit, or the CI would come from a different model class than the estimate.
        a2_p = base_fix_alpha2 if (p == base_p and base_fix_alpha2 != "inherit") else fix_alpha2
        boot = est.bootstrap(a["N"], a["D"], a["L"], seed=boot_spec.seed + i, n_jobs=n_jobs,
                             fix_E=fix_E, fix_alpha2=a2_p, fix_beta2=fix_beta2,
                             N0=N0, D0=D0, **boot_spec.kwargs())
        # boot.point IS the point estimate; est.bootstrap computes it with the same multistart
        # est.fit would. Calling est.fit separately would leave the plotted point and its CI
        # centre as two independently-computed objects, equal only by construction.
        fit = boot.point
        analyses[p] = dict(arr=a, fit=fit, boot=boot,
                           diag=est.diagnostics(fit, a["N"], a["D"], a["L"]))

    meta = dict(
        created=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        processed_dir=os.path.abspath(processed_dir),
        condition="matched",                 # (k_eval, p_eval) == (k_train, p_train); the eval diagonal
        k_train=k_used,
        fix_E=fix_E, fix_alpha2=fix_alpha2, fix_beta2=fix_beta2,
        # None is meaningful here (curvature free), so "inherit" is the sentinel for "no
        # override", and it round-trips through JSON as a string.
        base_fix_alpha2=base_fix_alpha2,
        base_p_train=(None if base_fix_alpha2 == "inherit" or base_p is None else float(base_p)),
        # The gauge a/b are reported in: the amplitudes are meaningless without it, and two
        # artifacts are only comparable if these agree.
        pivots=dict(N0=float(N0), D0=float(D0)),
        bootstrap=boot_spec.to_dict(),       # n_boot / seed

        filters=dict(max_ceiling_frac=max_ceiling_frac, min_tpp=min_tpp, min_ne_frac=min_ne_frac),
        filter_report={f"{k:g}": v for k, v in filt.items()},
        inventory=inventory,
    )
    if verbose:
        print()
        fit_store.print_fit_table(analyses, meta)   # the same table the plots notebook reprints
    return analyses, meta


def _print_inventory(inventory, k_used):
    n = sum(c["n"] for c in inventory)
    print(f"{n} models at the matched condition, k_train = {k_used:g}")
    print(f"{len(inventory)} p_train cohort(s):\n")
    hdr = f"{'p_train':>9}{'models':>8}{'distinct N':>12}{'distinct D':>12}{'L range':>18}"
    print(hdr); print("-" * len(hdr))
    for c in inventory:
        lrange = f"{c['L_min']:.3f} .. {c['L_max']:.3f}"
        print(f"{c['p_train']:>9g}{c['n']:>8}{c['n_distinct_N']:>12}{c['n_distinct_D']:>12}"
              f"{lrange:>18}")
    ses = [c["eval_se_median"] for c in inventory]
    print(f"\nper-model eval SE: median of cohort medians {np.median(ses):.5f} nats  "
          f"(max {max(c['eval_se_max'] for c in inventory):.5f})")
    short = sum(c["n_short_of_se_target"] for c in inventory)
    if short:
        print(f"!! {short} model(s) did NOT reach the SE target -- their loss is noisier than the rest")


if __name__ == "__main__":
    # ---- source: the processed EvalResult dir from process_eval_arrays.py ----------------- # <-- SET
    PROCESSED_DIR = "/mnt/storage/eval_summaries/training_2026-08-26-11-43-09"
    OUT_ROOT = "/media/trevor/data_flash/scaling_law_fits"
    OUT_DIR = os.path.join(OUT_ROOT, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

    # Which fault block size to analyze. None = require the data to hold exactly one and use it.
    K = None

    # the run screen, applied uniformly to every cohort by dl.filter_cohorts.
    # The ceiling cuts on the RESPONSE; the other two cut on the DESIGN.
    MAX_CEILING_FRAC = 1.0   # kill runs that didn't get below the entropy of the dataset
    MIN_TPP = 5              # None to disable; floor on tokens per TOTAL parameter
    MIN_NE_FRAC = 0.5        # None to disable; drops embedding-dominated bottom rungs

    #  the irreducible floor E. None fits it, 0.0 is a Kaplan pure power law, a
    # constant pins it.
    FIX_E = 1.7
    # the N-term's curvature alpha2. 0.0 is the plain Chinchilla power law, None
    # fits it, a constant pins it.
    FIX_ALPHA2 = None
    # Override for the BASE cohort only. 0.0 pins the fault-free reference's curvature at zero,
    # which keeps the capacity inversion a single division. This CONSTRAINS the reference fit
    # rather than relabelling it -- read the R2 of the p=0 row before relying on it.
    BASE_FIX_ALPHA2 = 0.0
    # curvature for the D term. Does not match the data
    FIX_BETA2 = 0.0
    # the pivots the amplitudes are reported at
    N0, D0 = est.N0_DEFAULT, est.D0_DEFAULT
    # the bootstrap
    BOOTSTRAP = fit_store.BootstrapSpec(n_boot=1500, seed=0)

    analyses, meta = fit_matched(PROCESSED_DIR, k=K, max_ceiling_frac=MAX_CEILING_FRAC,
                                 min_tpp=MIN_TPP, min_ne_frac=MIN_NE_FRAC, fix_E=FIX_E, fix_alpha2=FIX_ALPHA2, base_fix_alpha2=BASE_FIX_ALPHA2,
                                 fix_beta2=FIX_BETA2, N0=N0, D0=D0, bootstrap=BOOTSTRAP)
    print()
    fit_store.save_analyses(OUT_DIR, analyses, meta=meta)
    print(f"\nnext: point ARTIFACT_DIR in analysis/scaling_law_plots.ipynb at\n  {OUT_DIR}")
