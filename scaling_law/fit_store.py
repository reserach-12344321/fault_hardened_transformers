"""On-disk format for fitted scaling-law cohorts -- the fit script's handoff to the plots.

Not a pickle: the dataclasses round-trip through an .npz of arrays plus a JSON manifest of
scalars, so reading does not depend on this repo's class layout. load_analyses rebuilds them
and returns {p_train: {arr, fit, boot, diag}} -- the shape a plot cell consumes directly.

SCHEMA VERSIONS ARE REFUSED, NOT MIGRATED: each bump changed what a stored number MEANS, and
every one would be silently wrong in an artifact's most important numbers.
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, Tuple

import numpy as np

from .surface import ChinchillaFit
from .resample import BootstrapResult

__all__ = ["SCHEMA_VERSION", "MANIFEST_FILE", "ARRAYS_FILE", "BootstrapSpec", "save_analyses",
           "load_analyses", "print_fit_table"]

SCHEMA_VERSION = 6
MANIFEST_FILE = "fit_manifest.json"
ARRAYS_FILE = "cohorts.npz"


@dataclass(frozen=True)
class BootstrapSpec:
    """How an artifact's CIs were computed: the resample count and the seed."""
    n_boot: int = 1500
    seed: int = 0

    def validate(self) -> "BootstrapSpec":
        """Raise on an unusable spec, before the minutes-long fit rather than after."""
        if int(self.n_boot) < 1:
            raise ValueError(f"n_boot must be >= 1; got {self.n_boot!r}")
        return self

    def kwargs(self) -> dict:
        """The keyword arguments for scaling_law.bootstrap."""
        return dict(n_boot=int(self.n_boot))

    def label(self) -> str:
        """One-line description for a print-out / plot provenance line."""
        return f"pairs (case resampling), n_boot={self.n_boot}, seed={self.seed}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BootstrapSpec":
        """Rebuild from a manifest's `bootstrap` block."""
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown bootstrap fields {sorted(unknown)} in the manifest; expected "
                             f"{sorted(cls.__dataclass_fields__)}. An artifact carrying fields this "
                             f"schema does not define predates schema {SCHEMA_VERSION} -- re-run the "
                             f"fit script.")
        return cls(**d)


def _cohort_key(i: int) -> str:
    return f"c{i:02d}"


def save_analyses(out_dir: str, analyses: Dict[float, dict], *, meta: dict, verbose: bool = True) -> str:
    """Write `analyses` and `meta` to a fresh `out_dir`, and return it."""
    _check_bootstrap_agrees(analyses, meta)
    os.makedirs(out_dir, exist_ok=False)
    arrays: Dict[str, np.ndarray] = {}
    cohorts_meta = []
    for i, p in enumerate(sorted(analyses)):
        a = analyses[p]
        key = _cohort_key(i)
        for name, v in a["arr"].items():
            arrays[f"{key}/arr/{name}"] = np.asarray(v)
        for name, v in a["boot"].samples.items():
            arrays[f"{key}/boot/{name}"] = np.asarray(v, dtype=float)
        cohorts_meta.append(dict(
            key=key, p_train=float(p), n=int(np.size(a["arr"]["L"])),
            fit=a["fit"].as_dict(),
            # The bootstrap carries its own point fit, which IS `fit`. Stored rather than
            # assumed, so a mismatch shows up in the artifact instead of being defined away.
            boot_point=a["boot"].point.as_dict(),
            n_boot=int(a["boot"].n_boot),
            diag={k: float(v) for k, v in a["diag"].items()},
            # How the optimiser behaved, per cohort. A fit whose multi-start barely
            # converged is not obvious from its parameters, so the witness travels with the
            # artifact rather than scrolling past in a log.
            starts=dict(getattr(a["boot"], "starts", {}) or {}),
        ))
    np.savez_compressed(os.path.join(out_dir, ARRAYS_FILE), **arrays)
    manifest = dict(schema_version=SCHEMA_VERSION, **meta, cohorts=cohorts_meta)
    with open(os.path.join(out_dir, MANIFEST_FILE), "w") as f:
        json.dump(manifest, f, indent=2, default=_json_default)
    if verbose:
        n_runs = sum(c["n"] for c in cohorts_meta)
        print(f"wrote {len(cohorts_meta)} cohort(s) / {n_runs} runs to {out_dir}\n"
              f"  {ARRAYS_FILE}: {len(arrays)} arrays | {MANIFEST_FILE}: switches + provenance")
    return out_dir


def load_analyses(path: str) -> Tuple[Dict[float, dict], dict]:
    """Read an artifact dir back as ({p_train: {arr, fit, boot, diag}}, manifest)."""
    manifest = _read_manifest(path)
    with np.load(os.path.join(path, ARRAYS_FILE), allow_pickle=False) as z:
        names = list(z.files)
        analyses: Dict[float, dict] = {}
        for c in manifest["cohorts"]:
            key = c["key"]
            arr = {n.split("/")[-1]: z[n] for n in names if n.startswith(f"{key}/arr/")}
            samples = {n.split("/")[-1]: z[n] for n in names if n.startswith(f"{key}/boot/")}
            if not arr or not samples:
                raise ValueError(f"{path}: cohort {key} (p_train={c['p_train']}) has no "
                                 f"{'arrays' if not arr else 'bootstrap samples'} in {ARRAYS_FILE}")
            analyses[float(c["p_train"])] = dict(
                arr=arr,
                fit=ChinchillaFit(**c["fit"]),
                boot=BootstrapResult(point=ChinchillaFit(**c["boot_point"]), samples=samples,
                                     n_boot=int(c["n_boot"]), starts=dict(c["starts"])),
                diag=dict(c["diag"]),
            )
    return analyses, manifest


def _check_bootstrap_agrees(analyses: Dict[float, dict], meta: dict) -> None:
    """The manifest's bootstrap block must describe the bootstraps actually stored."""
    block = meta.get("bootstrap")
    if not block:
        raise ValueError("meta has no 'bootstrap' block; pass BootstrapSpec(...).to_dict() so the "
                         "artifact records how its CIs were produced")
    spec = BootstrapSpec.from_dict(dict(block))
    for p, a in analyses.items():
        b = a["boot"]
        if int(b.n_boot) != int(spec.n_boot):
            raise ValueError(
                f"cohort p_train={p:g} was bootstrapped with n_boot={b.n_boot}, but meta says "
                f"n_boot={spec.n_boot}")


def print_fit_table(analyses: Dict[float, dict], manifest: dict) -> None:
    """The per-cohort fit summary: a switch line plus E/alpha1/beta1/R2 per p_train."""
    fix_E = manifest["fix_E"]
    efit = ("E free" if fix_E is None else
            ("E fixed = 0 (Kaplan no-floor)" if fix_E == 0 else f"E fixed = {fix_E}"))
    a2 = manifest.get("fix_alpha2", 0.0)
    ecurv = ("N-curvature free" if a2 is None else
             ("alpha2 = 0 (plain power law)" if a2 == 0 else f"alpha2 fixed = {a2}"))
    b2 = manifest.get("fix_beta2", 0.0)
    dcurv = ("D-curvature free" if b2 is None else
             ("beta2 = 0 (plain power law)" if b2 == 0 else f"beta2 fixed = {b2}"))
    print(f"k_train = {manifest['k_train']:g}   |   N = total params   |   {efit}   |   "
          f"{ecurv}   |   {dcurv}")
    # A per-cohort override means the rows were not all produced by the switches above, so
    # it is said here rather than left in the manifest for a reader to notice.
    ba2 = manifest.get("base_fix_alpha2", "inherit")
    if ba2 != "inherit":
        print(f"base cohort p_train = {manifest.get('base_p_train')}: "
              + ("alpha2 free" if ba2 is None else f"alpha2 fixed = {ba2:g}")
              + "  -- the REFERENCE fit is constrained differently from the rest")
    piv = manifest.get("pivots")
    if piv:
        print(f"pivots: N0 = {piv['N0']:.4g}, D0 = {piv['D0']:.4g}  -- 'a' and 'b' are the two terms' "
              f"log-values there")
    # The CIs in this table are only as meaningful as the resampling behind them, so it is named here
    # rather than left to a reader to go find in the manifest.
    print(f"CIs: {BootstrapSpec.from_dict(dict(manifest['bootstrap'])).label()}\n")
    hdr = (f"{'cohort':<15}{'n':>5}{'E':>8}{'a':>8}{'alpha1 (95% CI)':>24}"
           f"{'beta1 (95% CI)':>24}{'R2_log':>8}")
    print(hdr); print("-" * len(hdr))
    for p in sorted(analyses):
        a = analyses[p]
        f, b = a["fit"], a["boot"]
        al_lo, al_hi = b.ci("alpha1")
        be_lo, be_hi = b.ci("beta1")
        print(f"{f'p_train={p:g}':<15}{np.size(a['arr']['L']):>5}{f.E:>8.3f}{f.a:>+8.3f}"
              f"{f'{f.alpha1:+.3f} [{al_lo:+.3f}, {al_hi:+.3f}]':>24}"
              f"{f'{f.beta1:+.3f} [{be_lo:+.3f}, {be_hi:+.3f}]':>24}{a['diag']['r2_log']:>8.3f}")
    # One curvature block per axis, printed only when that curvature is actually non-zero somewhere --
    # a fit with both held is the common case and gains nothing from two empty tables.
    _print_curvature(analyses, "alpha2", "N")
    _print_curvature(analyses, "beta2", "D")


def _print_curvature(analyses: Dict[float, dict], param: str, axis: str) -> None:
    """The per-cohort curvature table for one axis."""
    if not any(getattr(analyses[p]["fit"], param) for p in analyses):
        return
    eff = "effective_alpha" if axis == "N" else "effective_beta"
    print()
    print(f"{'cohort':<15}{f'{param} (95% CI)':>26}{'eff. slope at ladder ends':>30}")
    for p in sorted(analyses):
        f, b = analyses[p]["fit"], analyses[p]["boot"]
        lo, hi = b.ci(param)
        x = analyses[p]["arr"][axis]
        print(f"{f'p_train={p:g}':<15}{f'{getattr(f, param):+.4f} [{lo:+.4f}, {hi:+.4f}]':>26}"
              f"{f'{getattr(f, eff)(x.min()):+.3f} .. {getattr(f, eff)(x.max()):+.3f}':>30}")


def _read_manifest(path: str) -> dict:
    mpath = os.path.join(path, MANIFEST_FILE)
    if not os.path.isfile(mpath):
        raise FileNotFoundError(f"{path} is not a fit artifact dir (no {MANIFEST_FILE}); point at what "
                                f"analysis/fit_matched_scaling_law.py wrote")
    with open(mpath) as f:
        manifest = json.load(f)
    got = manifest.get("schema_version")
    if got != SCHEMA_VERSION:
        raise ValueError(f"{path}: fit artifact schema_version {got}, this reader expects "
                         f"{SCHEMA_VERSION}; re-run analysis/fit_matched_scaling_law.py. Each bump "
                         f"changed what a stored number MEANS, so an older artifact would not fail "
                         f"to read -- it would be quietly wrong, which is why this is a refusal.")
    return manifest


def _json_default(o: Any):
    """numpy scalars/arrays -> JSON, which json.dump would otherwise refuse."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")
