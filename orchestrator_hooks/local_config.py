"""The workstation's own single-node SLURM, as a Cluster the orchestrator can schedule onto.

One node, reached over `ssh localhost` exactly like a real cluster. Not enabled anywhere by
default -- read the caveats first, because this box is not a cluster node:
  * one GPU, the same card you use interactively;
  * `experiment_dir` stages job inputs on `/`, which is nearly full;
  * 24 GB of VRAM against an A100's 80, so a training rung sized for cannon will not fit;
  * queue depth is counted from `squeue -u $USER`, which here also sees cluster_orchestrator's
    own test suite submitting to this same partition.
"""
from cluster_orchestrator.cluster import Cluster, PhysicalPartition, LogicalPartition

# `debug` has no time limit, so a worker is never killed mid-point.
#
# n_gpus_possible=(0, 1) is required, not decoration: placement gates on
# `gres_count in partition.n_gpus_possible` and every worker job carries gres gpu:1, so at the
# (0,) default this lane accepts nothing and its jobs sit pending forever with no error.
local_debug = LogicalPartition(
    name="debug",
    physical=PhysicalPartition("debug",
                               max_in_flight=100,   # one 4090, shared with interactive use
                               allow_n_pending=1),
    enabled=True, n_gpus_possible=(0, 1))

local = Cluster(name="local", ssh_host="localhost",
                experiment_dir="/home/trevor/slurm_experiment",
                job_file="/home/trevor/nano_llama/orchestrator_hooks/scripts/local_launcher.job",
                partitions=[local_debug])
