"""Stage fault-eval job arrays, in waves, for every completed run of a training sweep.

Each wave is a normal job array holding the runs that finished since the last one; a model
is staged exactly once, keyed on its job name. The eval worker scores each run's FINAL
checkpoint at the (k, p) pairs in its eval_config.json.

    <root>/experiment.json          the eval spec + the sweeps seen
    <root>/wave_<ts>/wave_spec.json marker + the spec that wave was staged under
    <root>/wave_<ts>/<job>/         final_model.{eqx,json}, checkpoint_meta.json, configs

Waves are built under a hidden .staging-<ts> and renamed in only once complete; the weights
are hardlinked, not copied. Edit the constants at the top of main() and re-run as the sweep
drains.
"""
import os
import glob
import json
import math
import errno
import shutil
import hashlib
import datetime
import collections

import numpy as np

from cluster_orchestrator import worker_api

from nano_llama.fault import FaultConfig
from nano_llama.fault_eval import EvalConfig
from nano_llama.metrics import final_val_loss

SPEC_FILE = "experiment.json"        # at the experiment root: the eval spec every wave shares
WAVE_MARKER = "wave_spec.json"       # in a wave dir: marks it as a wave (and records its spec)
WAVE_PREFIX = "wave_"
STAGING_PREFIX = ".staging-"         # a wave under construction; renamed into place when complete

# One candidate gridpoint: a run that finished training and is ready to be evaluated.  ``fault`` is
# its TRAIN FaultConfig, read during the scan so staging does not re-open it.
Candidate = collections.namedtuple("Candidate", "job_name sweep_name gridpoint_dir fault")
# Per-sweep tallies for the report.  ``diverged`` holds job names, not a count -- see _diverged.
SweepStats = collections.namedtuple("SweepStats", "name n_gridpoints n_ready n_incomplete diverged")


def _is_done(gridpoint_dir: str) -> bool:
    """Did this run train to completion? Asks cluster_orchestrator's DONE marker directly."""
    return worker_api.is_done(os.path.join(gridpoint_dir, "results"))


def _diverged(gridpoint_dir: str) -> bool:
    """Did this run diverge? True iff its last metrics row has a non-finite val loss."""
    try:
        with open(os.path.join(gridpoint_dir, "results", "metrics.json")) as fh:
            metrics = json.load(fh)
        if not metrics:
            return False
        v = final_val_loss(metrics)
        return v is not None and not math.isfinite(float(v))
    except (OSError, ValueError, KeyError):
        return False


def job_kp_pairs(k_train: float, p_train: float, extra_kp_pairs) -> tuple:
    """The eval conditions for one run: its two own-fault baselines, then the shared extras."""
    pairs, seen = [], set()
    for k, p in [(k_train, 0.0), (k_train, p_train), *extra_kp_pairs]:
        kp = (int(k), float(p))
        if kp not in seen:
            seen.add(kp)
            pairs.append(kp)
    return tuple(pairs)


def job_seed(base_seed: int, job_name: str) -> int:
    """This model's eval seed: a function of its NAME, not of its position in an array."""
    digest = hashlib.blake2b(f"{int(base_seed)}:{job_name}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2 ** 31)


def _link_or_copy(src: str, dst: str, hardlink: bool) -> None:
    """Place src at dst. hardlink=True links (no extra bytes."""
    if not hardlink:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError as e:
        if e.errno != errno.EXDEV:            # EXDEV = cross-device link; anything else is a real error
            raise
        shutil.copy2(src, dst)                # different filesystem -> a copy is the only option


def _write_json(path: str, doc) -> None:
    """Write a small JSON atomically (tmp + rename)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------
# The experiment root: its spec, its waves, and what has already been staged into them
# ---------------------------------------------------------------------------------------------

def build_spec(extra_kp_pairs, target_se: float, min_evals: int, max_evals: int, batch_size,
               seed: int) -> dict:
    """The eval spec as a JSON-normalised dict: what every wave of an experiment must share."""
    return {
        "extra_kp_pairs": [[int(k), float(p)] for k, p in extra_kp_pairs],
        "target_se": float(target_se),
        "min_evals": int(min_evals),
        "max_evals": int(max_evals),
        "batch_size": None if batch_size is None else int(batch_size),
        "seed": int(seed),
    }


def load_or_create_spec(root: str, spec: dict, *, sweep_dirs, allow_change: bool = False) -> tuple:
    """Read the experiment's spec, creating it on the first run -> (spec, created, changes)."""
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, SPEC_FILE)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    sweep_dirs = [os.path.abspath(d.rstrip("/")) for d in sweep_dirs]

    if not os.path.isfile(path):
        _write_json(path, {"created_at": now, "spec": spec, "sweep_dirs": sweep_dirs,
                           "superseded": []})
        return spec, True, []

    with open(path) as fh:
        doc = json.load(fh)
    old = doc["spec"]
    changes = [(key, old.get(key), spec[key]) for key in sorted(spec) if old.get(key) != spec[key]]
    if changes and not allow_change:
        detail = "\n".join(f"    {key}: {was!r} -> {now_!r}" for key, was, now_ in changes)
        raise SystemExit(
            f"ABORT: the eval spec differs from {path}:\n{detail}\n"
            f"  Models staged in earlier waves were scored under the OLD spec; pooling them with new\n"
            f"  ones would mix measurement conditions inside one cohort. Restore the constants above,\n"
            f"  start a new experiment root, or set ALLOW_SPEC_CHANGE = True if the mixture is\n"
            f"  intended (the old spec is then recorded under 'superseded').")
    if changes:
        doc.setdefault("superseded", []).append({"changed_at": now, "spec": old})
        doc["spec"] = spec
    doc["sweep_dirs"] = sorted(set(doc.get("sweep_dirs", [])) | set(sweep_dirs))
    _write_json(path, doc)
    return spec, False, changes


def wave_dirs(root: str) -> list:
    """Every wave in the experiment, oldest first."""
    if not os.path.isdir(root):
        return []
    return sorted(os.path.join(root, name) for name in os.listdir(root)
                  if os.path.isfile(os.path.join(root, name, WAVE_MARKER)))


def wave_jobs(wave_dir: str) -> list:
    """The job names in one wave (its subdirs; wave_spec.json / status.json are files)."""
    return sorted(name for name in os.listdir(wave_dir)
                  if os.path.isdir(os.path.join(wave_dir, name)))


def staged_job_names(root: str) -> dict:
    """{job_name: wave_dir} over every wave -- the models this experiment has already staged."""
    staged = {}
    for wave in wave_dirs(root):
        for name in wave_jobs(wave):
            staged.setdefault(name, wave)
    return staged


def wave_progress(wave_dir: str) -> tuple:
    """(n_done, n_jobs) for one wave, by the same DONE marker the orchestrator reads."""
    names = wave_jobs(wave_dir)
    done = sum(1 for n in names if worker_api.is_done(os.path.join(wave_dir, n, "results")))
    return done, len(names)


# ---------------------------------------------------------------------------------------------
# Scanning the training sweeps, and staging a wave
# ---------------------------------------------------------------------------------------------

def scan_candidates(sweep_dirs) -> tuple:
    """Every gridpoint across the sweeps ready to be evaluated -> (candidates, stats)."""
    candidates, stats = [], []
    for sweep_dir in sweep_dirs:
        sweep_name = os.path.basename(sweep_dir.rstrip("/"))
        n_grid = n_ready = n_incomplete = 0
        diverged = []
        for fault_path in sorted(glob.glob(os.path.join(sweep_dir, "*", "fault_config.json"))):
            gridpoint_dir = os.path.dirname(fault_path)
            name = os.path.basename(gridpoint_dir)
            n_grid += 1
            # The FINAL checkpoint, not best_model. It carries no config of its own, so it is
            # paired with the gridpoint's model_config.json into the pair Llama.deserialize
            # expects; meta.json comes along for the actual final step (-> D).
            ckpt_dir = os.path.join(gridpoint_dir, "results", "checkpoint")
            needed = (os.path.join(ckpt_dir, "model.eqx"), os.path.join(ckpt_dir, "meta.json"),
                      os.path.join(gridpoint_dir, "model_config.json"))
            if not _is_done(gridpoint_dir) or not all(map(os.path.isfile, needed)):
                n_incomplete += 1
                continue
            # DONE but blown up: the training worker marks a diverged run done on purpose, so this is
            # the only thing standing between a NaN checkpoint and a scored eval job.
            if _diverged(gridpoint_dir):
                diverged.append(f"{sweep_name}__{name}")
                continue
            candidates.append(Candidate(job_name=f"{sweep_name}__{name}", sweep_name=sweep_name,
                                        gridpoint_dir=gridpoint_dir, fault=FaultConfig.load(fault_path)))
            n_ready += 1
        stats.append(SweepStats(sweep_name, n_grid, n_ready, n_incomplete, diverged))

    names = [c.job_name for c in candidates]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise SystemExit(f"ABORT: {len(dupes)} job name(s) produced by more than one gridpoint -- the "
                         f"name is this experiment's only key, so one of each pair would be silently "
                         f"dropped as 'already staged'. Usually two sweep dirs sharing a basename:\n"
                         + "\n".join(f"    {n}" for n in dupes))
    return candidates, stats


def stage_wave(root: str, candidates, spec: dict, *, sweep_dirs, hardlink: bool = True) -> str:
    """Write one wave -- a complete, launchable job array -- and return its path."""
    # Named for the wall clock, disambiguated if two waves land in the same second (a re-run right
    # after a wave that staged nothing new, or a test) -- a collision must never silently merge two
    # waves into one dir.
    stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    suffix, n = stamp, 1
    while os.path.exists(os.path.join(root, WAVE_PREFIX + suffix)) or \
            os.path.exists(os.path.join(root, STAGING_PREFIX + suffix)):
        n += 1
        suffix = f"{stamp}_{n}"
    staging = os.path.join(root, STAGING_PREFIX + suffix)
    wave_dir = os.path.join(root, WAVE_PREFIX + suffix)
    os.makedirs(staging, exist_ok=False)

    extras = [tuple(kp) for kp in spec["extra_kp_pairs"]]
    for cand in candidates:
        job = os.path.join(staging, cand.job_name)
        os.makedirs(job)
        ckpt_dir = os.path.join(cand.gridpoint_dir, "results", "checkpoint")
        _link_or_copy(os.path.join(ckpt_dir, "model.eqx"),
                      os.path.join(job, "final_model.eqx"), hardlink)                  # the big one
        shutil.copy2(os.path.join(cand.gridpoint_dir, "model_config.json"),
                     os.path.join(job, "final_model.json"))
        shutil.copy2(os.path.join(ckpt_dir, "meta.json"), os.path.join(job, "checkpoint_meta.json"))
        shutil.copy2(os.path.join(cand.gridpoint_dir, "train_config.json"), job)
        shutil.copy2(os.path.join(cand.gridpoint_dir, "fault_config.json"), job)
        EvalConfig(kp_pairs=job_kp_pairs(cand.fault.k, cand.fault.p, extras),
                   target_se=spec["target_se"], min_evals=spec["min_evals"],
                   max_evals=spec["max_evals"], batch_size=spec["batch_size"],
                   seed=job_seed(spec["seed"], cand.job_name),          # per-model, name-derived
                   ).save(os.path.join(job, "eval_config.json"))

    _write_json(os.path.join(staging, WAVE_MARKER), {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "spec": spec,
        "sweep_dirs": [os.path.abspath(d.rstrip("/")) for d in sweep_dirs],
        "n_jobs": len(candidates),
    })
    os.rename(staging, wave_dir)      # atomic publish: the wave becomes visible complete or not at all
    return wave_dir


def main() -> None:

    # The EXPERIMENT ROOT: one dir per eval campaign, holding every wave. Re-running this script with
    # the same root stages whatever has finished training since the last wave; point it somewhere new
    # to start a fresh campaign.
    root = "/media/trevor/data_flash/job_arrays/eval_pgrid_512"

    sweep_dirs = ["/media/trevor/data_flash/job_arrays/d512_tpp_range_sweep_2026-08-21-10-43-45",]

    # The spec below must stay FIXED for the life of an experiment root -- earlier waves' models were
    # measured under it and are pooled with later ones. A change aborts unless this is set (which
    # records the old spec under "superseded" in experiment.json, and each wave keeps its own copy).
    ALLOW_SPEC_CHANGE = False

    p_grid = [2 * float(x) for x in  np.logspace(start=-3, stop=-1, num=80)]

    # Extra eval conditions, scored IN ADDITION to each run's own two baselines, which
    # job_kp_pairs prepends. A cross product here, but the format is a flat pair list, so a
    # dense p-grid on one k and a sparse one on another is equally expressible.
    k_values = [4]

    extra_kp_pairs = [(k, p) for k in k_values for p in p_grid]

    hardlink = True

    # A precision target rather than a fixed budget, absolute so every model on the ladder is
    # measured to the same accuracy. Each p draws its OWN sequences, so a difference like
    # L(p)-L(0) carries sqrt(2)x this error: 0.005 resolves the high-p end of the curve but
    # not the low-p end. Cost scales as (1/target_se)^2.
    target_se = 0.005
    min_evals = 8
    max_evals = 65536

    seed = 2323          # BASE seed; each model's own seed is derived from its name (job_seed)

    batch_size = None  # adaptively chosen per-GPU on the node (like training); set an int to pin it

    spec = build_spec(extra_kp_pairs, target_se, min_evals, max_evals, batch_size, seed)
    spec, created, changes = load_or_create_spec(root, spec, sweep_dirs=sweep_dirs,
                                                 allow_change=ALLOW_SPEC_CHANGE)

    print(f"fault-eval experiment: {root}" + ("   (NEW -- wrote experiment.json)" if created else ""))
    print(f"  extras: k={k_values} x {len(p_grid)} p in [{min(p_grid):.2g}, {max(p_grid):.2g}] "
          f"= {len(extra_kp_pairs)} pairs, PLUS each run's (k_train, 0) and (k_train, p_train)")
    print(f"  target_se={target_se:g} nats min_evals={min_evals} max_evals={max_evals} | "
          f"batch_size={batch_size or 'auto'} | base seed={seed} | hardlink={hardlink}")
    for key, was, is_ in changes:
        print(f"  !! SPEC CHANGED (allowed): {key}: {was!r} -> {is_!r} -- earlier waves used the old value")

    # ---- what this experiment has already staged, and how far those waves have got ----
    waves = wave_dirs(root)
    staged = staged_job_names(root)
    if waves:
        print(f"\nexisting waves ({len(waves)}):")
        for wave in waves:
            n_done, n_jobs = wave_progress(wave)
            flag = "" if n_done == n_jobs else "   <- still owed a launch/resume"
            print(f"  {os.path.basename(wave):<28} {n_done:>4}/{n_jobs:<4} jobs done{flag}")
        print(f"  {len(staged)} model(s) staged so far")
    else:
        print(f"\nno waves yet -- this is the first")

    # ---- what the sweeps have ready NOW ----
    candidates, stats = scan_candidates(sweep_dirs)
    print(f"\nscanning {len(sweep_dirs)} sweep(s):")
    for st in stats:
        print(f"  {st.name}: {st.n_gridpoints} gridpoints | {st.n_ready} ready, "
              f"{st.n_incomplete} not finished training, {len(st.diverged)} diverged")

    new = [c for c in candidates if c.job_name not in staged]
    print(f"\n{len(candidates)} ready model(s), {len(candidates) - len(new)} already staged "
          f"-> {len(new)} new")

    diverged = [name for st in stats for name in st.diverged]
    if diverged:
        # Named, not just counted, on EVERY wave: these are DONE-marked by the training worker, so they
        # will be re-scanned (and re-excluded) forever, and each one is a training-recipe failure.
        print(f"\n  !! {len(diverged)} DIVERGED run(s) excluded (non-finite final val loss) --")
        print(f"     these were marked DONE by the training worker on purpose, so only the divergence")
        print(f"     check keeps their NaN checkpoints out of the eval array:")
        for name in diverged:
            print(f"       {name}")

    if not new:
        print(f"\nnothing new to stage -- no wave written."
              f"\n  Re-run once more of the sweeps has finished training.")
        return

    wave_dir = stage_wave(root, new, spec, sweep_dirs=sweep_dirs, hardlink=hardlink)
    print(f"\nwrote {len(new)} eval job(s) -> {wave_dir}")
    print(f"\nNEXT: python -c \"from scripts.inject_resources import inject_resources; "
          f"inject_resources('{wave_dir}', 4, 24, 1)\"")
    print(f"      then launch with the eval worker (orchestrator_hooks/eval_worker.py) on THIS wave")
    print(f"      (earlier waves are separate arrays -- launch/resume each on its own)")


if __name__ == "__main__":
    main()
