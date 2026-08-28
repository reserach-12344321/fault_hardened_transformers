"""Launch a nano_llama fault-sweep job array on cannon and engaging, end to end.

The array lives locally; the orchestrator pushes each job to the cluster, sbatches the
worker, and pulls results back. The array must already carry its per-job seeds (meta.json,
written by experiment_util/gen_full_sweep.py) and resources:

    python -c "from experiment_util.inject_resources import inject_resources; \
               inject_resources('<array>', <cpus>, <mem_gb>, <gpus>)"
    PYTHONPATH=<nano_llama>:<cluster_orchestrator> \
        python orchestrator_hooks/scripts/launch_everything_fineweb.py <array_dir>

Needs a live ssh control-master socket and both repos checked out on the cluster at the
PYTHONPATH paths in the launcher .job. Re-running an initialized array resumes it, so this
is also the crash-recovery entrypoint.
"""
import os
import argparse

from cluster_orchestrator import ClusterRun, initialize_job_array, orchestrate, load_status
from orchestrator_hooks import RESUME_PATHS
from orchestrator_hooks.cannon_config import cannon
from orchestrator_hooks.engaging_config import engaging

# Paths ON CANNON (tmccourt's checkouts / netscratch) -- must match cannon_launcher.job.
cannon_worker = "/n/home00/tmccourt/nano_llama/orchestrator_hooks/worker.py"
cannon_data = "/n/netscratch/iaifi_lab/Lab/tmccourt/data/fineweb_shuf/tok8192"
engaging_worker = "/home/tmccourt/nano_llama/orchestrator_hooks/worker.py"
engaging_data = "/orcd/pool/007/tmccourt/data/fineweb/tok8192"  # HDD pool: frees scratch flash for
                                                                # outputs, and the loader reads it
                                                                # in large sequential megablocks



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("array_dir", help="LOCAL job-array directory (already stamped with resources + seeds)")
    ap.add_argument("--max-seconds", type=float, default=0,
                    help="worker wall-time budget for self-exit + resume (0 = none)")
    ap.add_argument("--poll", type=float, default=15, help="orchestrator poll interval (s)")
    args = ap.parse_args()

    run_map = {cannon: ClusterRun(worker=cannon_worker, resume_paths=RESUME_PATHS,
                                  static_data={"data": cannon_data, "save_best": False}),
               engaging: ClusterRun(worker=engaging_worker, resume_paths=RESUME_PATHS,
                                    static_data={"data": engaging_data, "save_best": False})}

    if not os.path.exists(os.path.join(args.array_dir, "status.json")):
        initialize_job_array(args.array_dir)
    else:
        print(f"already initialized -> resuming {args.array_dir}")

    orchestrate(args.array_dir, run_map,
                poll_interval=args.poll, worker_max_seconds=args.max_seconds)

    status = load_status(args.array_dir)
    n_done = sum(1 for j in status["jobs"].values() if j["state"] == "done")
    print(f"\nlaunch_cannon: {n_done}/{len(status['jobs'])} jobs done")


if __name__ == "__main__":
    main()
