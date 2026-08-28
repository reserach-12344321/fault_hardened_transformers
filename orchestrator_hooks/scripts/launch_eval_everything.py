"""Run a nano_llama fault-eval job array across both clusters, end to end.

The eval-worker counterpart of launch_everything_fineweb.py: same two-cluster run_map, but
each ClusterRun drives eval_worker.py. Every job re-scores one trained checkpoint at a grid
of fault p and writes a single eval_results.json.

Unlike training there is no seed-injection step (prepare_eval_array bakes a per-model seed
into each eval_config.json) -- just stamp resources first:

    python -c "from experiment_util.inject_resources import inject_resources; \
               inject_resources('<array>', <cpus>, <mem_gb>, <gpus>)"
    PYTHONPATH=<nano_llama>:<cluster_orchestrator> \
        python orchestrator_hooks/scripts/launch_eval_everything.py <array_dir> [--max-seconds N] [--poll N]

Re-running on an already-initialized array resumes it.
"""
import os
import argparse

from cluster_orchestrator import ClusterRun, initialize_job_array, orchestrate, load_status
from orchestrator_hooks.cannon_config import cannon
from orchestrator_hooks.engaging_config import engaging
from orchestrator_hooks.local_config import local



# Paths ON each cluster (tmccourt's checkouts / scratch) -- must match the launcher .job files.
cannon_worker = "/n/home00/tmccourt/nano_llama/orchestrator_hooks/eval_worker.py"
# An eval .bin FILE, not a dir: the held-out split, so the final fault curves are unbiased by
# the val split used for best-model selection during training.
cannon_data = "/n/netscratch/iaifi_lab/Lab/tmccourt/data/fineweb_shuf/tok8192/val.bin"
engaging_worker = "/home/tmccourt/nano_llama/orchestrator_hooks/eval_worker.py"
engaging_data = "/orcd/pool/007/tmccourt/data/fineweb/tok8192/val.bin"
# The workstation's own single-node SLURM (see local_config): real capacity, but ONE 4090 that is also
# your interactive card, so it is opt-in via --local rather than always in the pool.
local_worker = "/home/trevor/nano_llama/orchestrator_hooks/eval_worker.py"
# The CHUNK-SHUFFLED copy, matching what the two clusters serve. The unshuffled original is
# still on this box, and pointing here at THAT would silently corrupt any mixed array: jobs
# on the 4090 would score a different corpus from jobs on cannon/engaging with nothing to
# signal it. The two differ ONLY in size, which is what _assert_same_split checks.
local_data = "/home/trevor/data/fineweb_shuf/tok8192/val_iid.bin"

# Every path in this launcher must be the SAME split, byte for byte: an eval array exists to compare
# models to each other, so a per-cluster data difference is common-mode error that does not cancel and
# is invisible in the results. Sizes are cheap and sufficient (a shuffle preserves length only when it
# is the same file; the shuffle changed it by the dropped tail).
_SPLIT_BYTES = 4_000_317_440          # chunk-shuffled val_iid, seed 1337


def _assert_same_split(paths: dict) -> None:
    """Fail BEFORE launching if any reachable path disagrees on size."""
    for name, p in paths.items():
        if not os.path.isfile(p):
            continue                                   # remote paths, or a box without this file
        n = os.path.getsize(p)
        if n != _SPLIT_BYTES:
            raise SystemExit(
                f"{name} = {p}\n  is {n:,} bytes, expected {_SPLIT_BYTES:,} (chunk-shuffled val_iid).\n"
                f"  4,000,698,588 means this is the PRE-SHUFFLE original -- scoring an array partly on\n"
                f"  it and partly on the shuffled split would silently corrupt every model comparison.")

# SlidingLoader buffer (GB) for the eval file. Without it the worker auto-selects
# whole-in-RAM, so every job cold-reads the whole multi-GB split before its first batch --
# and this FS has a long enough tail that one such read can cost tens of minutes.
#
# Do not shrink it far without re-checking: a buffered job reads exactly ONE megablock and
# `perm` is seeded per-job, so each MODEL scores a different slice. That region-to-region
# spread is between-model error the per-point SE does not capture, shrinking as 1/sqrt(buffer).
EVAL_BUFFER_GB = 10.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("array_dir", help="LOCAL fault-eval job-array directory (already stamped with resources)")
    ap.add_argument("--max-seconds", type=float, default=0,
                    help="worker wall-time budget for self-exit + resume (0 = none)")
    ap.add_argument("--poll", type=float, default=15, help="orchestrator poll interval (s)")
    ap.add_argument("--local", action="store_true", default=False,
                    help="also schedule onto the workstation's own SLURM (one 4090, shared with "
                         "your interactive use -- see orchestrator_hooks/local_config.py)")
    args = ap.parse_args()

    _assert_same_split({"cannon_data": cannon_data, "engaging_data": engaging_data,
                        "local_data": local_data})
    static = lambda path: {"data": path, "buffer_gb": EVAL_BUFFER_GB}
    run_map = {#cannon: ClusterRun(worker=cannon_worker, static_data=static(cannon_data)),
               engaging: ClusterRun(worker=engaging_worker, static_data=static(engaging_data))}
    if args.local:
        run_map[local] = ClusterRun(worker=local_worker, static_data=static(local_data))
        print(f"local cluster ENABLED: 1 job at a time on the 4090 | data {local_data}")

    if not os.path.exists(os.path.join(args.array_dir, "status.json")):
        initialize_job_array(args.array_dir)
    else:
        print(f"already initialized -> resuming {args.array_dir}")

    orchestrate(args.array_dir, run_map,
                poll_interval=args.poll, worker_max_seconds=args.max_seconds)

    status = load_status(args.array_dir)
    n_done = sum(1 for j in status["jobs"].values() if j["state"] == "done")
    print(f"\nlaunch_eval_everything: {n_done}/{len(status['jobs'])} eval jobs done")


if __name__ == "__main__":
    main()
