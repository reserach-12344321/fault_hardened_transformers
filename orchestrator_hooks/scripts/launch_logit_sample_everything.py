"""Run a marginal-predictive job array across both clusters, end to end.

    python experiment_util/prepare_logit_sample_array.py
    python -c "from experiment_util.inject_resources import inject_resources; \
               inject_resources('<array>', 4, 24, 1)"
    PYTHONPATH=<nano_llama>:<cluster_orchestrator> \
        python orchestrator_hooks/scripts/launch_logit_sample_everything.py <array_dir> [--local]

Re-running resumes, at (k, p) POINT granularity within a job.

TWO DIFFERENCES FROM THE EVAL ARRAY, BOTH OF WHICH BITE SILENTLY:

1. buffer_gb IS PART OF THE SCIENCE HERE, not an I/O knob. This array pins one shared context
   set, and SlidingLoader's geometry is a function of the buffer -- so the buffer selects
   WHICH sequences it returns. Marginals collected under different buffers are not poolable
   and nothing in the outputs would say so.
2. A POINT IS THE RESUME UNIT and can outlast a short walltime. The worker only checkpoints
   between points, so a point longer than the partition's limit is killed and restarted every
   allocation -- a livelock. The banner warns against cannon's preemptible-lane limit.
"""
import os
import json
import glob
import argparse

from cluster_orchestrator import ClusterRun, initialize_job_array, orchestrate, load_status
from orchestrator_hooks.cannon_config import cannon
from orchestrator_hooks.engaging_config import engaging
from orchestrator_hooks.local_config import local
# Reuse the eval launcher's split fingerprint and guard rather than restating them: a second
# copy of _SPLIT_BYTES is a constant that can drift out of step with the maintained one.
from orchestrator_hooks.scripts.launch_eval_everything import _assert_same_split, _SPLIT_BYTES
from nano_llama.train_core import choose_loader_buffer_for_file

# Paths ON each cluster (tmccourt's checkouts / scratch) -- must match the launcher .job files.
cannon_worker = "/n/home00/tmccourt/nano_llama/orchestrator_hooks/logit_sample_worker.py"
engaging_worker = "/home/tmccourt/nano_llama/orchestrator_hooks/logit_sample_worker.py"
local_worker = "/home/trevor/nano_llama/orchestrator_hooks/logit_sample_worker.py"

# The .bin the CONTEXT SET is drawn from: the chunk-shuffled IID held-out split. These three
# paths are byte-identical -- only the directory naming differs, because engaging's pool had
# no room for a second copy and took the shuffled corpus in place. Identical bytes is what
# lets a job land on either cluster and still draw the same contexts.
cannon_data = "/n/netscratch/iaifi_lab/Lab/tmccourt/data/fineweb_shuf/tok8192/val.bin"
engaging_data = "/orcd/pool/007/tmccourt/data/fineweb/tok8192/val.bin"
local_data = "/home/trevor/data/fineweb_shuf/tok8192/val_iid.bin"

# The context-set buffer -- see note 1 in the header. None (auto) resolves to whole-file
# resident for all three paths, since they are the same file, so it is identical on every
# cluster and matches what the local launcher uses. Change it only to another
# whole-in-RAM-equivalent value, and only for every deployment at once.
CONTEXT_BUFFER_GB = None

# Throughput anchor for the per-point walltime estimate below, shared with the prep script
# so the two agree by construction.
from experiment_util.prepare_logit_sample_array import ANCHOR_FLOPS_PER_S, seq_flops   # noqa: E402


def _requeue_seconds() -> tuple:
    """cannon's preemptible-lane walltime as (seconds, literal)."""
    try:
        from orchestrator_hooks.cannon_config import _REQUEUE_TIME
        spec = _REQUEUE_TIME.split("=", 1)[1]                       # "--time=1:00:00" -> "1:00:00"
    except Exception:
        return None, None
    days, _, rest = spec.rpartition("-")
    parts = [int(x) for x in rest.split(":")]
    while len(parts) < 3:
        parts.append(0)
    secs = parts[0] * 3600 + parts[1] * 60 + parts[2] + (int(days) * 86400 if days else 0)
    return secs, spec


def _worst_point_seconds(array_dir: str) -> tuple:
    """(seconds for the array's most expensive single (k, p) point, its job name)."""
    worst, who = 0.0, None
    for cfg_path in sorted(glob.glob(os.path.join(array_dir, "*", "logit_sample_config.json"))):
        job = os.path.dirname(cfg_path)
        try:
            with open(cfg_path) as f:
                sc = json.load(f)
            with open(os.path.join(job, "final_model.json")) as f:
                mc = json.load(f)
        except (OSError, ValueError):
            continue
        secs = int(sc["n_contexts"]) * int(sc["n_chips"]) * seq_flops(mc) / ANCHOR_FLOPS_PER_S
        if secs > worst:
            worst, who = secs, os.path.basename(job)
    return worst, who


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("array_dir", help="LOCAL marginal-predictive job-array directory (already stamped "
                                      "with resources)")
    ap.add_argument("--max-seconds", type=float, default=0,
                    help="worker wall-time budget for self-exit + resume (0 = none). NOTE the worker "
                         "only tests this BETWEEN points, so it cannot rescue a job whose single "
                         "point outlasts the partition walltime -- see the header.")
    ap.add_argument("--poll", type=float, default=15, help="orchestrator poll interval (s)")
    ap.add_argument("--local", action="store_true",
                    help="ALSO schedule onto the workstation's own SLURM (one 4090, shared with your "
                         "interactive use -- see orchestrator_hooks/local_config.py). Off by default: "
                         "this array is GPU-hours per job and would sit on your interactive card.")
    args = ap.parse_args()

    # Same split, byte for byte, on every path -- a per-cluster data difference would change the
    # context set itself here, not merely the sample. Local-only (the cluster paths cannot be stat'd
    # from here); the remotes were verified by sha256 on 2026-08-25, recorded above.
    _assert_same_split({"cannon_data": cannon_data, "engaging_data": engaging_data,
                        "local_data": local_data})

    static = lambda path: ({"data": path} if CONTEXT_BUFFER_GB is None
                           else {"data": path, "buffer_gb": CONTEXT_BUFFER_GB})
    run_map = {cannon: ClusterRun(worker=cannon_worker, static_data=static(cannon_data)),
               engaging: ClusterRun(worker=engaging_worker, static_data=static(engaging_data))}
    if args.local:
        run_map[local] = ClusterRun(worker=local_worker, static_data=static(local_data))

    # ---- banner: the two things that bite silently (header notes 1 and 2) ----
    resolved = choose_loader_buffer_for_file(local_data) if CONTEXT_BUFFER_GB is None else CONTEXT_BUFFER_GB
    n_jobs = len(glob.glob(os.path.join(args.array_dir, "*", "logit_sample_config.json")))
    print(f"logit-sample array: {args.array_dir}  ({n_jobs} job(s))")
    print(f"  split   : {_SPLIT_BYTES:,} B, identical on all 3 paths (sha256-verified 2026-08-25)")
    print(f"  contexts: buffer_gb={CONTEXT_BUFFER_GB} -> resolves to "
          f"{'whole-in-RAM' if resolved is None else f'{resolved} GB sliding'} on every cluster "
          f"(the context set is a FUNCTION of this -- see the header)")
    print(f"  clusters: cannon + engaging" + (" + local(4090)" if args.local else ""))

    worst, who = _worst_point_seconds(args.array_dir)
    limit, spec = _requeue_seconds()
    if worst:
        print(f"  cost    : worst single (k,p) point ~{worst / 3600:.2f} h  ({who})")
    if worst and limit:
        if worst > limit:
            print(f"\n  !! WALLTIME: cannon's preemptible lanes are capped at {spec} "
                  f"({limit / 3600:.2f} h) by cannon_config._REQUEUE_TIME, but the worst point needs "
                  f"~{worst / 3600:.2f} h.\n"
                  f"     A point is the resume unit, so such a job is killed mid-point EVERY "
                  f"allocation, restarts the same\n"
                  f"     point, and never completes one -- it will make no progress at all, quietly. "
                  f"Raise _REQUEUE_TIME\n"
                  f"     (or drop the offending model) before launching on those lanes.")
        else:
            print(f"  walltime: OK -- worst point {worst / 3600:.2f} h fits cannon's preemptible "
                  f"cap of {spec} ({limit / 3600:.2f} h)")

    if not os.path.exists(os.path.join(args.array_dir, "status.json")):
        initialize_job_array(args.array_dir)
    else:
        print(f"already initialized -> resuming {args.array_dir}")

    orchestrate(args.array_dir, run_map,
                poll_interval=args.poll, worker_max_seconds=args.max_seconds)

    status = load_status(args.array_dir)
    n_done = sum(1 for j in status["jobs"].values() if j["state"] == "done")
    print(f"\nlaunch_logit_sample_everything: {n_done}/{len(status['jobs'])} marginal jobs done")


if __name__ == "__main__":
    main()
