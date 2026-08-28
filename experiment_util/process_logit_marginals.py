#!/usr/bin/env python
#!/usr/bin/env python
r"""Reduce a logit-marginal job array to a small artifact the paper figures can ship with.

The raw campaign is ~1.4 GB per job, none of which the figures draw. Two passes, because
they cost wildly different amounts:

  materialize        the digest: read every point of every job and fit a temperature at
                     each, keeping the per-(run, fault) scalars. Incremental and parallel
                     over jobs, which are independent. Not a GPU job -- the solve is a
                     moment-matching Newton iteration in float64.
  materialize_clouds the scatter clouds: one job's base.npz plus ONE point npz, so
                     re-selecting the figure's panels is seconds rather than a full pass.

The digest FIXES the temperature model; a different one needs the raw array again. `moment`
is stored per point so a T can be re-derived and checked without re-reading the groups.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np
from joblib import Parallel, delayed

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    # repo root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiment_util import logit_marginals as lm

# Bumped whenever a stored field changes meaning. load_digest REFUSES an older artifact
# outright rather than reading a field that has moved underneath it -- the same contract
# fit_store applies to a fit artifact, and for the same reason: a silently
# mis-read digest is wrong numbers in a paper figure with nothing to notice it.
SCHEMA_VERSION = 1
MANIFEST_FILE = "logit_digest.json"      # the campaign-level record; per-job files sit beside it
CLOUD_DIR = "clouds"                     # scatter clouds, one npz per (job, p_eval)
DIGEST_QUANTILES = (10.0, 50.0, 90.0)    # of the per-context KL vectors, before they are dropped


# ======================================================================================
# Writing: one job -> its digest record
# ======================================================================================
def run_digest(run) -> dict:
    """Reduce ONE loaded :class:`logit_marginals.MarginalRun` to the scalars the figures read."""
    z0 = np.asarray(run.clean_logits, np.float64)
    fits = lm.fits_for_run(run)          # hoists the per-run invariants; bit-identical, ~20% cheaper
    points = []
    for p in run.p_eval:
        pt = run.points[p]
        d = fits[p]
        rec = dict(p_eval=float(p), k_eval=int(pt.k), index=int(pt.index),
                   T=float(d["T"]),
                   # The temperature solve's sufficient statistic: T is the unique value with
                   # sum_c E_{softmax(z0/T)}[z0] == this. Stored so T stays checkable without groups.
                   moment=float((np.asarray(pt.p_bar, np.float64) * z0).sum()),
                   frac=float(d["frac"]),
                   noise_floor_kl=float(pt.noise_floor_kl))
        for name in ("total", "residual", "explained"):
            v = np.asarray(d[name], np.float64)
            rec[f"{name}_mean"] = float(v.mean())
            for q, x in zip(DIGEST_QUANTILES, np.percentile(v, DIGEST_QUANTILES)):
                rec[f"{name}_q{q:g}"] = float(x)
        points.append(rec)

    return dict(
        schema_version=SCHEMA_VERSION,
        job=run.job, size_key=run.size_key,
        n_params=int(run.n_params), n_non_embedding=int(run.n_non_embedding),
        n_train_tokens=int(run.n_train_tokens),
        p_train=float(run.p_train), k_train=float(run.k_train),
        n_chips=int(run.n_chips), n_groups=int(run.n_groups),
        context_seed=int(run.context_seed), context_stream=int(run.context_stream),
        n_contexts=int(z0.shape[0]), vocab_size=int(z0.shape[1]),
        clean_loss=float(run.clean_loss),
        points=points,
    )


def _source_point_count(job_dir: str) -> int:
    """How many points the SOURCE job has written."""
    try:
        with open(os.path.join(job_dir, lm.RESULTS_JSON)) as f:
            return sum(1 for d in json.load(f).get("points", []) if not d.get("is_clean"))
    except (OSError, ValueError):
        return -1


def _needs_work(job_dir: str, dest: str) -> bool:
    """Does this job need (re)processing? Cheap: reads only the small per-job JSONs."""
    if not os.path.isfile(dest):
        return True
    try:
        with open(dest) as f:
            have = len(json.load(f)["points"])
    except (OSError, ValueError, KeyError):
        return True                       # unreadable digest -> rewrite it
    return have < _source_point_count(job_dir)


def _process_one(job_dir: str, dest: str) -> tuple:
    """Reduce ONE job and write its record. Module-level so joblib can ship it to a worker."""
    name = os.path.basename(job_dir.rstrip("/"))
    try:
        rec = run_digest(lm.load_run(job_dir, keep_raw=False))
    except (FileNotFoundError, ValueError) as e:
        return ("failed", name, str(e))
    tmp = dest + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f)
    os.replace(tmp, dest)                 # atomic: a kill mid-write never leaves half a record
    return ("written", name, "")


def materialize(array_dirs, out_dir: str, *, require_done: bool = False,
                skip_existing: bool = True, n_jobs: int = -1, verbose: bool = True) -> tuple:
    """Reduce every job under `array_dirs` into `out_dir`, one <job>.json per job.

    Jobs are independent -- a run's temperature fit reads only its own logits -- so the
    parallel result is bit-identical to the serial one. `n_jobs=-1` takes the whole box; on a
    workstation shared with live training, set it smaller.

    `skip_existing` reprocesses a job only if it has no digest yet or has written more points
    than its digest covers, which is the case while an array is still draining.

    Returns (written, skipped, failed) as lists of job names; failed carries the reason.
    """
    os.makedirs(out_dir, exist_ok=True)
    dirs = lm.logit_array_dirs(array_dirs)
    jobs = [d for a in dirs for d in sorted(_job_dirs(a, require_done=require_done))]

    todo, skipped = [], []
    for job_dir in jobs:
        name = os.path.basename(job_dir.rstrip("/"))
        dest = os.path.join(out_dir, f"{name}.json")
        if skip_existing and not _needs_work(job_dir, dest):
            skipped.append(name)
        else:
            todo.append((job_dir, dest))

    if verbose:
        print(f"{len(jobs)} job(s) across {len(dirs)} array dir(s) -> {out_dir}")
        print(f"  {len(todo)} to process, {len(skipped)} already up to date"
              + (f" | n_jobs={n_jobs}" if todo else ""))
    out = Parallel(n_jobs=n_jobs, verbose=(5 if verbose else 0))(
        delayed(_process_one)(jd, dest) for jd, dest in todo) if todo else []

    written = [name for st, name, _ in out if st == "written"]
    failed = [(name, why) for st, name, why in out if st == "failed"]
    if verbose:
        print(f"\ndigest: {len(written)} written, {len(skipped)} up to date, {len(failed)} failed")
        for name, why in failed[:5]:
            print(f"  !! {name}: {why[:100]}")
    return written, skipped, failed


def _job_dirs(array_dir: str, *, require_done: bool = False) -> list:
    """Job subdirs of one array dir -- the same test :func:`logit_marginals.load_array` applies."""
    import glob
    out = []
    for d in sorted(glob.glob(os.path.join(array_dir, "*"))):
        if not (os.path.isdir(d) and os.path.isfile(os.path.join(d, "final_model.json"))):
            continue
        if require_done and not os.path.isfile(os.path.join(d, lm.DONE_MARKER)):
            continue
        out.append(d)
    return out


# ======================================================================================
# Writing: the scatter clouds  (cheap -- one point file per panel, not a whole job)
# ======================================================================================
def cloud_for(job_dir: str, p_eval: float, *, n_sample: int = 250_000, dead_floor: float = 1e-6,
              seed: int = 0) -> dict:
    """The centred (z0, log p_bar) cloud for one (job, p_eval), subsampled."""
    with open(os.path.join(job_dir, lm.RESULTS_JSON)) as f:
        doc = json.load(f)
    match = [d for d in doc["points"]
             if not d.get("is_clean") and np.isclose(float(d["p"]), float(p_eval))]
    if not match:
        raise KeyError(f"{os.path.basename(job_dir)}: no point at p_eval={p_eval:g} "
                       f"(has {sorted(float(d['p']) for d in doc['points'] if not d.get('is_clean'))})")
    d = match[0]
    with np.load(os.path.join(job_dir, lm.RESULTS_BASE)) as zb:
        z0 = np.asarray(zb["clean_logits"], np.float64)                 # (C, V)
    ppath = lm.point_path(job_dir, int(d["index"]))
    with np.load(ppath) as z:
        p_bar = np.asarray(z["groups"], np.float64).mean(axis=0)        # (C, V) -- the marginal

    live = lm.softmax(z0).max(0) > dead_floor
    zl = z0[:, live]
    zf = np.log(np.clip(p_bar[:, live], 1e-20, 1.0))
    x = (zl - zl.mean(1, keepdims=True)).ravel()
    y = (zf - zf.mean(1, keepdims=True)).ravel()
    n_all = int(x.size)
    if n_sample and n_sample < n_all:
        idx = np.random.default_rng(seed).choice(n_all, n_sample, replace=False)
        x, y = x[idx], y[idx]
    return dict(z0=x.astype(np.float32), z_fault=y.astype(np.float32),
                job=os.path.basename(job_dir.rstrip("/")), p_eval=float(d["p"]), k_eval=int(d["k"]),
                n_live=int(live.sum()), n_vocab=int(z0.shape[1]), n_contexts=int(z0.shape[0]),
                n_all=n_all, n_sample=int(x.size), dead_floor=float(dead_floor), seed=int(seed))


def _cloud_name(job: str, p_eval: float) -> str:
    return f"{job}__p{p_eval:.6g}.npz"


def materialize_clouds(array_dirs, out_dir: str, selections, *, n_sample: int = 250_000,
                       dead_floor: float = 1e-6, seed: int = 0, verbose: bool = True) -> list:
    """Write one cloud npz per ``(job, p_eval)`` in ``selections`` into ``out_dir/clouds/``."""
    dest_dir = os.path.join(out_dir, CLOUD_DIR)
    os.makedirs(dest_dir, exist_ok=True)
    by_name = {os.path.basename(d.rstrip("/")): d
               for a in lm.logit_array_dirs(array_dirs) for d in _job_dirs(a)}
    written = []
    for job, p in selections:
        if job not in by_name:
            raise KeyError(f"job {job!r} is not under {array_dirs!r}")
        c = cloud_for(by_name[job], p, n_sample=n_sample, dead_floor=dead_floor, seed=seed)
        path = os.path.join(dest_dir, _cloud_name(job, c["p_eval"]))
        np.savez_compressed(path, **{k: v for k, v in c.items() if k != "job"}, job=np.array(job))
        written.append(path)
        if verbose:
            print(f"  cloud {job[-34:]}  p={c['p_eval']:.4g}  "
                  f"{c['n_sample']:,}/{c['n_all']:,} pts  {os.path.getsize(path)/1e6:.1f} MB")
    return written


def write_manifest(out_dir: str, *, source, extra: dict | None = None) -> str:
    """Record what the artifact is and where it came from, beside the per-job records."""
    jobs = sorted(f[:-5] for f in os.listdir(out_dir) if f.endswith(".json") and f != MANIFEST_FILE)
    clouds = sorted(os.listdir(os.path.join(out_dir, CLOUD_DIR))) if \
        os.path.isdir(os.path.join(out_dir, CLOUD_DIR)) else []
    man = dict(schema_version=SCHEMA_VERSION,
               source=[source] if isinstance(source, str) else list(source),
               n_jobs=len(jobs), n_clouds=len(clouds), clouds=clouds, **(extra or {}))
    path = os.path.join(out_dir, MANIFEST_FILE)
    with open(path, "w") as f:
        json.dump(man, f, indent=1)
    return path


# Reading: what the notebook loads instead of the array. The reader lives beside the writer
# on purpose -- the format IS the contract between the expensive pass and the figures, and
# splitting the two ends across modules is how they drift.
@dataclass(frozen=True)
class DigestRun:
    """One model, as the figures see it."""
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
    n_contexts: int
    vocab_size: int
    clean_loss: float
    p_eval: tuple

    @property
    def tokens_per_param(self) -> float:
        """Tokens per TOTAL parameter -- the unit the sweep was designed in."""
        return self.n_train_tokens / self.n_params

    @property
    def arm(self) -> str:
        return "clean" if self.p_train == 0.0 else f"p_train={self.p_train:g}"


@dataclass
class LogitDigest:
    """A whole campaign, reduced: the runs, their per-fault scalars, and any stored scatter clouds."""
    runs: list                               # [DigestRun], sorted (N, D, p_train)
    fit: dict                                # (job, p_eval) -> the scalars dict
    clouds: dict = field(default_factory=dict, repr=False)   # (job, p_eval) -> npz path
    manifest: dict = field(default_factory=dict, repr=False)

    def cloud(self, job: str, p_eval: float) -> dict:
        """Load one stored scatter cloud -> dict with ``z0`` / ``z_fault`` (centred."""
        key = self._cloud_key(job, p_eval)
        if key is None:
            raise KeyError(
                f"no stored cloud for ({job!r}, p_eval={p_eval:g}). The digest carries scalars for "
                f"every run, but clouds only for the selections materialize_clouds was given; stored: "
                f"{sorted((j, float(f'{p:g}')) for j, p in self.clouds)}. Re-run the cloud pass -- it "
                f"reads one point file per panel, so it takes seconds.")
        with np.load(self.clouds[key], allow_pickle=False) as z:
            out = {k: z[k] for k in z.files}
        out["job"] = str(out["job"])
        return out

    def _cloud_key(self, job: str, p_eval: float):
        for (j, p) in self.clouds:
            if j == job and np.isclose(p, p_eval):
                return (j, p)
        return None

    def p_eval_grid(self) -> list:
        """Every eval fault present on any run, ascending."""
        return sorted({p for (_j, p) in self.fit})


def load_digest(out_dir: str) -> LogitDigest:
    """Read an artifact written by :func:`materialize` (+ :func:`materialize_clouds`)."""
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"{out_dir} is not a directory")
    files = sorted(f for f in os.listdir(out_dir) if f.endswith(".json") and f != MANIFEST_FILE)
    if not files:
        raise FileNotFoundError(
            f"{out_dir} holds no per-job digests; point at what process_logit_marginals.materialize "
            f"wrote")
    runs, fit = [], {}
    for f in files:
        with open(os.path.join(out_dir, f)) as fh:
            rec = json.load(fh)
        got = rec.get("schema_version")
        if got != SCHEMA_VERSION:
            raise ValueError(f"{f}: digest schema_version {got}, this reader expects "
                             f"{SCHEMA_VERSION}; re-run process_logit_marginals.py")
        runs.append(DigestRun(
            job=rec["job"], size_key=rec["size_key"], n_params=rec["n_params"],
            n_non_embedding=rec["n_non_embedding"], n_train_tokens=rec["n_train_tokens"],
            p_train=rec["p_train"], k_train=rec["k_train"], n_chips=rec["n_chips"],
            n_groups=rec["n_groups"], context_seed=rec["context_seed"],
            context_stream=rec["context_stream"], n_contexts=rec["n_contexts"],
            vocab_size=rec["vocab_size"], clean_loss=rec["clean_loss"],
            p_eval=tuple(sorted(p["p_eval"] for p in rec["points"]))))
        for p in rec["points"]:
            fit[(rec["job"], p["p_eval"])] = dict(p, job=rec["job"])

    # The same context-identity guard load_array applies: p_bar rows are positions in a fixed context
    # set, so runs drawn from different seeds are not comparable and must not land in one figure.
    ctx = {(r.context_seed, r.context_stream, r.n_contexts) for r in runs}
    if len(ctx) != 1:
        raise ValueError(f"the digest pools runs from {len(ctx)} different context sets {sorted(ctx)}; "
                         f"they are not comparable per-context")

    clouds = {}
    cdir = os.path.join(out_dir, CLOUD_DIR)
    if os.path.isdir(cdir):
        for f in sorted(os.listdir(cdir)):
            if not f.endswith(".npz"):
                continue
            job, _, ptxt = f[:-4].rpartition("__p")
            clouds[(job, float(ptxt))] = os.path.join(cdir, f)

    man = {}
    if os.path.isfile(os.path.join(out_dir, MANIFEST_FILE)):
        with open(os.path.join(out_dir, MANIFEST_FILE)) as fh:
            man = json.load(fh)

    runs.sort(key=lambda r: (r.n_params, r.n_train_tokens, r.p_train))
    return LogitDigest(runs=runs, fit=fit, clouds=clouds, manifest=man)


def select_panels(digest: LogitDigest, tpp_rows, p_show: float, arms=None) -> tuple:
    """Resolve the figure's (tokens/param row) x (training arm) panel grid against a digest."""
    arms = sorted({r.p_train for r in digest.runs}) if arms is None else list(arms)
    missing = [a for a in arms if a not in {r.p_train for r in digest.runs}]
    if missing:
        raise KeyError(f"no runs at p_train {missing}; digest has "
                       f"{sorted({r.p_train for r in digest.runs})}")
    chosen = []
    for tpp in tpp_rows:
        for arm in arms:
            hits = [r for r in digest.runs
                    if round(r.tokens_per_param) == tpp and r.p_train == arm]
            if not hits:
                have = sorted({round(x.tokens_per_param) for x in digest.runs})
                near = min(have, key=lambda t: abs(np.log(t / tpp))) if have else None
                raise KeyError(
                    f"no run at {tpp} tok/param, p_train={arm:g}. This digest has tokens/param "
                    f"{have}; nearest to {tpp} is {near}. TPP_ROWS is a DESIGN choice (which two "
                    f"durations the figure's rows are), so it is not snapped for you -- set it to a "
                    f"value this array actually has, in BOTH this script and the notebook.")
            chosen.append(hits[0])
    common = sorted(set.intersection(*(set(r.p_eval) for r in chosen)))
    if not common:
        raise ValueError("the selected runs share no eval fault")
    p_draw = min(common, key=lambda p: abs(np.log(p) - np.log(p_show)))
    return [(r.job, p_draw) for r in chosen], p_draw


if __name__ == "__main__":
    # ---- source array + destination ------------------------------------------- # <-- SET
    ARRAY = "/media/trevor/data_flash/job_arrays/logit_marginals_d512_tpp"
    OUT_DIR = "/mnt/storage/logit_digests/d512_tpp"
    REQUIRE_DONE = True        # True drops jobs still being scored (they would carry a short grid)

    # ---- the scatter panels to keep clouds for (the 2x2 figure's grid) --------- # <-- SET
    # The 2x2 scatter is clean vs the arm trained at the fault it is drawn at. THIS array has six
    # arms (0, 0.02, 0.04, 0.06, 0.08, 0.12), so the pair has to be named; the figure grid is 2x2.
    ARMS_SHOW = [0.0, 0.04]
    TPP_ROWS = [80, 640]       # must match analysis/logit_temperature_d512.ipynb. Tokens per TOTAL
                               # parameter, the unit the sweep was designed in: this array's ladder is
                               # [5, 10, 20, 40, 80, 160, 320, 640]. A stale value fails loudly in
                               # select_panels rather than drawing a wrong row.
    P_SHOW = 0.04
    N_SAMPLE = 250_000         # stored per panel; the figure draws 30k of them

    written, skipped, failed = materialize(ARRAY, OUT_DIR, require_done=REQUIRE_DONE, n_jobs=16)

    dg = load_digest(OUT_DIR)
    print(f"\ndigest: {len(dg.runs)} runs, {len(dg.fit)} (run, p_eval) points")
    sel, p_draw = select_panels(dg, TPP_ROWS, P_SHOW, arms=ARMS_SHOW)
    print(f"scatter panels at p_eval = {p_draw:g}:")
    materialize_clouds(ARRAY, OUT_DIR, sel, n_sample=N_SAMPLE)

    write_manifest(OUT_DIR, source=ARRAY,
                   extra=dict(require_done=REQUIRE_DONE, tpp_rows=list(TPP_ROWS),
                              arms_show=list(ARMS_SHOW), p_show=P_SHOW, p_draw=p_draw,
                              n_sample=N_SAMPLE))
    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(OUT_DIR) for f in fs)
    print(f"\nartifact: {OUT_DIR}  ({size/1e6:.1f} MB)")
