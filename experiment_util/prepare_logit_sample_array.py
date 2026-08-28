"""Stage marginal-predictive job arrays, in waves, for every usable run of a sweep.

The sibling of prepare_eval_array: those jobs score fresh text under one chip and reduce
each sequence to a loss, these hold a FIXED context set and score it under many chips,
keeping the full next-token probability vector to build p_bar = E_chip[softmax(z)].

THE SPEC IS FROZEN PER ROOT. Every wave's models are pooled into one cohort, and the fitted
temperatures are compared across arms, durations and waves -- only meaningful if every model
saw the same context set and chip budget. A change aborts unless ALLOW_SPEC_CHANGE.

Cost is GPU-hours per job and the outputs are large; both are reported per wave and both
have a ceiling that aborts before anything is staged.

Usage: edit the constants at the top of main() and run it; re-run as the sweep drains.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")   # config generation only -- never touches a GPU

import glob
import json
import shutil
import datetime
import collections

import numpy as np

from nano_llama.fault import FaultConfig
from nano_llama.fault_eval import LogitSampleConfig

# The WAVE scheme (root spec + drift guard, wave discovery, name-derived seeds, the hardlink) is
# shared with prepare_eval_array rather than restated -- none of it is specific to what a wave's jobs
# then compute. `_is_done` / `_diverged` back the gridpoint screen defined below.
from experiment_util.prepare_eval_array import (_link_or_copy, _write_json, _is_done, _diverged,
                                                job_kp_pairs, job_seed,
                                                load_or_create_spec, wave_dirs, staged_job_names, wave_progress,
                                                WAVE_MARKER, WAVE_PREFIX, STAGING_PREFIX)
# `non_embedding_params` / `size_key` have ONE definition, in the marginal-predictive analysis
# module; seq_flops (below) is built on the first.
from experiment_util.logit_marginals import non_embedding_params, size_key


# Gridpoint screen + forward-cost arithmetic.
def seq_flops(mc: dict) -> float:
    """Forward FLOPs for ONE sequence under this config (matmuls + attention's T x T products)."""
    d, L, T = int(mc["n_embd"]), int(mc["n_layer"]), int(mc["block_size"])
    return 2.0 * non_embedding_params(mc) * T + 4.0 * L * T * T * d


def _read_gridpoint(gridpoint_dir: str):
    """Load one gridpoint's configs, or return (None."""
    if not os.path.isdir(gridpoint_dir):
        return None, "no such directory"
    paths = {n: os.path.join(gridpoint_dir, n)
             for n in ("model_config.json", "train_config.json", "fault_config.json")}
    missing = [n for n, p in paths.items() if not os.path.isfile(p)]
    if missing:
        return None, f"missing {', '.join(missing)} (is this a gridpoint dir?)"
    ckpt_dir = os.path.join(gridpoint_dir, "results", "checkpoint")
    for n in ("model.eqx", "meta.json"):
        if not os.path.isfile(os.path.join(ckpt_dir, n)):
            return None, f"missing results/checkpoint/{n} (run never checkpointed)"
    if not _is_done(gridpoint_dir):
        return None, "not DONE (still training, or died before finishing)"
    # DONE but blown up: the training worker marks a diverged run done on purpose, so this check is the
    # only thing standing between a NaN checkpoint and a sampled job.
    if _diverged(gridpoint_dir):
        return None, "DIVERGED (non-finite final val loss)"
    with open(paths["model_config.json"]) as f:
        mc = json.load(f)
    with open(paths["train_config.json"]) as f:
        tc = json.load(f)
    fc = FaultConfig.load(paths["fault_config.json"])
    return (os.path.basename(os.path.dirname(gridpoint_dir.rstrip("/"))),   # sweep name = parent dir
            os.path.basename(gridpoint_dir.rstrip("/")), gridpoint_dir, mc, fc, tc), None

# Matches launch_eval_everything's calibration, tuned up for the >~1e8-param regime this
# array lives in.
ANCHOR_FLOPS_PER_S = 8.5e13

# One candidate: a run that finished training and can be sampled. `mc` is its raw model_config dict
# (enough for the FLOP/storage arithmetic) and `fault` its TRAIN FaultConfig, both read during the
# scan so staging does not re-open them.
Candidate = collections.namedtuple("Candidate", "job_name sweep_name gridpoint_dir fault mc")
SweepStats = collections.namedtuple("SweepStats", "name n_gridpoints n_ready n_incomplete diverged")


def estimate_runtime(cands, kp_of, n_contexts, n_chips, flops_per_s=ANCHOR_FLOPS_PER_S):
    """Seconds per candidate and total."""
    per_job, by_arch = [], collections.defaultdict(lambda: [0, 0.0])
    for c in cands:
        n_fault = sum(1 for _, p in kp_of(c.fault) if p > 0)
        secs = n_contexts * (n_chips * n_fault + 1) * seq_flops(c.mc) / flops_per_s
        per_job.append(secs)
        by_arch[size_key(c.mc)][0] += 1
        by_arch[size_key(c.mc)][1] += secs
    return per_job, dict(by_arch), float(sum(per_job))


def estimate_storage(cands, kp_of, n_groups, n_contexts, n_raw):
    """Bytes per candidate and total, for what the WORKER writes back."""
    per_job = []
    for c in cands:
        V = int(c.mc["vocab_size"])
        n_fault = sum(1 for _, p in kp_of(c.fault) if p > 0)
        groups = n_groups * n_contexts * V * 4 * n_fault
        raw = max(n_raw, 0) * V * 4 * n_fault
        base = n_contexts * V * 4 + n_contexts * int(c.mc["block_size"]) * 4    # clean_logits + ctx
        per_job.append(float(groups + raw + base))
    return per_job, float(sum(per_job))


def build_spec(extra_kp_pairs, n_contexts, n_chips, n_groups, n_raw, raw_context,
               context_seed, context_stream, micro_batch, seed) -> dict:
    """The sampling spec as a JSON-normalised dict: what every wave of an experiment shares."""
    return {
        "extra_kp_pairs": [[int(k), float(p)] for k, p in extra_kp_pairs],
        "n_contexts": int(n_contexts), "n_chips": int(n_chips), "n_groups": int(n_groups),
        "n_raw": int(n_raw), "raw_context": int(raw_context),
        "context_seed": int(context_seed), "context_stream": int(context_stream),
        "micro_batch": None if micro_batch is None else int(micro_batch),
        "seed": int(seed),
    }


def scan_candidates(sweep_dirs) -> tuple:
    """Every gridpoint across the sweeps ready to be sampled -> (candidates, stats)."""
    candidates, stats = [], []
    for sweep_dir in sweep_dirs:
        sweep_name = os.path.basename(sweep_dir.rstrip("/"))
        n_grid = n_ready = n_incomplete = 0
        diverged = []
        for fault_path in sorted(glob.glob(os.path.join(sweep_dir, "*", "fault_config.json"))):
            gp = os.path.dirname(fault_path)
            name = os.path.basename(gp)
            n_grid += 1
            job, reason = _read_gridpoint(gp)          # DONE + not diverged + files present
            if job is None:
                if reason and reason.startswith("DIVERGED"):
                    diverged.append(f"{sweep_name}__{name}")
                else:
                    n_incomplete += 1
                continue
            candidates.append(Candidate(job_name=f"{sweep_name}__{name}", sweep_name=sweep_name,
                                        gridpoint_dir=gp, fault=FaultConfig.load(fault_path),
                                        mc=job[3]))
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
        ckpt = os.path.join(cand.gridpoint_dir, "results", "checkpoint")
        _link_or_copy(os.path.join(ckpt, "model.eqx"), os.path.join(job, "final_model.eqx"), hardlink)
        shutil.copy2(os.path.join(cand.gridpoint_dir, "model_config.json"),
                     os.path.join(job, "final_model.json"))
        shutil.copy2(os.path.join(ckpt, "meta.json"), os.path.join(job, "checkpoint_meta.json"))
        shutil.copy2(os.path.join(cand.gridpoint_dir, "train_config.json"), job)
        shutil.copy2(os.path.join(cand.gridpoint_dir, "fault_config.json"), job)
        kp = tuple((k, p) for k, p in job_kp_pairs(cand.fault.k, cand.fault.p, extras) if p > 0.0)
        LogitSampleConfig(kp_pairs=kp, n_contexts=spec["n_contexts"], n_chips=spec["n_chips"],
                          n_groups=spec["n_groups"], n_raw=spec["n_raw"],
                          raw_context=spec["raw_context"], context_seed=spec["context_seed"],
                          context_stream=spec["context_stream"], micro_batch=spec["micro_batch"],
                          seed=job_seed(spec["seed"], cand.job_name),   # per-model, name-derived
                          ).save(os.path.join(job, "logit_sample_config.json"))

    _write_json(os.path.join(staging, WAVE_MARKER), {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "spec": spec,
        "sweep_dirs": [os.path.abspath(d.rstrip("/")) for d in sweep_dirs],
        "n_jobs": len(candidates),
    })
    os.rename(staging, wave_dir)      # atomic publish: visible complete or not at all
    return wave_dir


def main() -> None:

    # The EXPERIMENT ROOT: one dir per sampling campaign, holding every wave. Re-running with the same
    # root stages whatever has finished training since the last wave. Keep it on the SAME filesystem
    # as the sweeps so the weights hardlink instead of copying (a cross-mount root copies a full
    # checkpoint per job), and note the results pulled back here are the big artifact -- see the
    # storage line in the cost report.
    root = "/media/trevor/data_flash/job_arrays/logit_marginals_d512_tpp"

    sweep_dirs = ["/media/trevor/data_flash/job_arrays/d512_tpp_range_sweep_2026-08-21-10-43-45"]

    # The spec below must stay FIXED for the life of an experiment root -- earlier waves' models were
    # sampled under it and are pooled with later ones. A change aborts unless this is set (which
    # records the old spec under "superseded" in experiment.json; each wave also keeps its own copy).
    ALLOW_SPEC_CHANGE = False

    # EXTRA eval fault conditions, scored IN ADDITION to each run's own (k_train, p_train) baseline
    # (job_kp_pairs prepends it per run; its (k_train, 0.0) partner is dropped as deterministic).
    # Cost and OUTPUT SIZE are both LINEAR in the number of points here -- see the cost report.
    p_grid = [2 * float(x) for x in np.logspace(start=-3, stop=-1, num=80)]
    k_values = [4]
    extra_kp_pairs = [(k, p) for k in k_values for p in p_grid]

    n_contexts = 256            # C: fixed contexts (final token of each of 256 shared sequences)
    n_chips = 1000              # M: chips marginalised per context (fixed budget)
    n_groups = 2                # K: running sub-averages, for the split-group KL noise floor.
                                #    THE STORAGE KNOB: the npz is K * C * V * 4 bytes PER POINT.
    n_raw = 100                 # single-chip distributions kept for ONE context (per-trajectory probe)
    raw_context = 0
    context_seed = 4242         # defines the SHARED context set; keep identical across arrays for reuse
    context_stream = 777
    micro_batch = None          # None -> chosen on-node for whatever GPU the job lands on

    # Ceilings, applied to the WAVE about to be staged (the unit you actually launch), with the
    # campaign total reported alongside. Both abort before anything is written.
    WARN_GPU_HOURS = 3000.0
    WARN_TB = 1.0

    hardlink = True
    seed = 20250724             # BASE seed; each model's own is derived from its name (job_seed)

    spec = build_spec(extra_kp_pairs, n_contexts, n_chips, n_groups, n_raw, raw_context,
                      context_seed, context_stream, micro_batch, seed)
    spec, created, changes = load_or_create_spec(root, spec, sweep_dirs=sweep_dirs,
                                                 allow_change=ALLOW_SPEC_CHANGE)

    print(f"marginal-predictive experiment: {root}" + ("   (NEW -- wrote experiment.json)" if created else ""))
    print(f"  sample : C={n_contexts} contexts, M={n_chips} chips, K={n_groups} groups, n_raw={n_raw}")
    print(f"  extras : k={k_values} x {len(p_grid)} p in [{min(p_grid):.3g}, {max(p_grid):.3g}] "
          f"= {len(extra_kp_pairs)} pairs, PLUS each run's (k_train, p_train) [p=0 dropped]")
    print(f"  context: seed={context_seed} stream={context_stream} | base seed={seed} | "
          f"hardlink={hardlink}")
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

    diverged = [n for st in stats for n in st.diverged]
    if diverged:
        # Named, not just counted, on EVERY wave: these are DONE-marked by the training worker, so
        # they are re-scanned (and re-excluded) forever, and each is a training-recipe failure.
        print(f"\n  !! {len(diverged)} DIVERGED run(s) excluded (non-finite final val loss):")
        for n in diverged[:10]:
            print(f"       {n}")
        if len(diverged) > 10:
            print(f"       ... and {len(diverged) - 10} more")
    if not new:
        print(f"\nnothing new to stage -- no wave written."
              f"\n  Re-run once more of the sweep has finished training.")
        return

    # ---- cost + size of THIS wave, and of the campaign it joins ----
    kp_of = lambda fc: job_kp_pairs(fc.k, fc.p, [tuple(kp) for kp in spec["extra_kp_pairs"]])
    _per, by_arch, wave_s = estimate_runtime(new, kp_of, n_contexts, n_chips)
    _perb, wave_b = estimate_storage(new, kp_of, n_groups, n_contexts, n_raw)
    _p2, _b2, all_s = estimate_runtime(candidates, kp_of, n_contexts, n_chips)
    _p3, all_b = estimate_storage(candidates, kp_of, n_groups, n_contexts, n_raw)
    print(f"\nthis wave ({len(new)} job(s)), forward arithmetic at {ANCHOR_FLOPS_PER_S:.2g} FLOP/s:")
    hdr = f"{'architecture':>14}{'jobs':>7}{'per job':>12}{'subtotal':>12}{'npz/job':>12}"
    print(hdr); print("-" * len(hdr))
    for arch in sorted(by_arch):
        n, secs = by_arch[arch]
        gb = max(b for c, b in zip(new, _perb) if size_key(c.mc) == arch) / 1e9
        print(f"{arch:>14}{n:>7}{secs / n / 3600:>10.2f} h{secs / 3600:>10.1f} h{gb:>10.2f} GB")
    print(f"  wave    : {wave_s / 3600:,.0f} GPU-hours | {wave_b / 1e12:.2f} TB of results")
    print(f"  campaign: {all_s / 3600:,.0f} GPU-hours | {all_b / 1e12:.2f} TB if every ready model "
          f"is eventually staged")
    try:
        free_tb = shutil.disk_usage(root).free / 1e12
        print(f"  disk    : {free_tb:.2f} TB free on the root's filesystem")
    except OSError:
        free_tb = None

    if wave_s / 3600 > WARN_GPU_HOURS:
        raise SystemExit(f"\nABORT: this wave is {wave_s / 3600:,.0f} GPU-hours, over WARN_GPU_HOURS="
                         f"{WARN_GPU_HOURS:g}. Nothing staged. Cut n_chips / n_contexts / the p-grid, "
                         f"or raise the constant.")
    if wave_b / 1e12 > WARN_TB:
        raise SystemExit(f"\nABORT: this wave would write {wave_b / 1e12:.2f} TB of results, over "
                         f"WARN_TB={WARN_TB:g}. Nothing staged. n_groups is the biggest lever (the npz "
                         f"is n_groups * n_contexts * vocab * 4 PER POINT), then the p-grid length and "
                         f"n_contexts. Or raise the constant if the disk really has room.")

    wave_dir = stage_wave(root, new, spec, sweep_dirs=sweep_dirs, hardlink=hardlink)
    print(f"\nwrote {len(new)} marginal-predictive job(s) -> {wave_dir}")
    print(f"\nNEXT: python -c \"from scripts.inject_resources import inject_resources; "
          f"inject_resources('{wave_dir}', 4, 24, 1)\"")
    print(f"      python orchestrator_hooks/scripts/launch_logit_sample_everything.py {wave_dir}")
    print(f"      (earlier waves are separate arrays -- launch/resume each on its own)")


if __name__ == "__main__":
    main()
