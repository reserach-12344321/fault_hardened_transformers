from cluster_orchestrator.cluster import Cluster, PhysicalPartition, LogicalPartition


_inf = 1000

cannon_iaifi_gpu = LogicalPartition(
    name="iaifi_gpu",
    physical=PhysicalPartition("iaifi_gpu", max_in_flight=_inf, allow_n_pending=2),
    sbatch_args=['--account=iaifi_lab'],
    enabled=True, n_gpus_possible=(0, 1, 2, 4))

_REQUEUE_TIME = "--time=6:00:00"

cannon_iaifi_requeue = LogicalPartition(
    name="iaifi_gpu_requeue",
    physical=PhysicalPartition("iaifi_gpu_requeue", max_in_flight=_inf, allow_n_pending=2),
    sbatch_args=['--account=iaifi_lab', '--no-requeue', _REQUEUE_TIME],
    enabled=True, n_gpus_possible=(0, 1))

cannon_gpu_requeue = LogicalPartition(
    name="gpu_requeue",
    physical=PhysicalPartition("gpu_requeue", max_in_flight=_inf, allow_n_pending=2),
    sbatch_args=['--account=iaifi_lab',
                 "--exclude=holygpu8a31105,holygpu8a19102,holygpu7c1713,holygpu7c1734",
                 "--no-requeue",
                 "--constraint='a100|h100|h200|rtx6000pro'", _REQUEUE_TIME],  # --no-requeue
    enabled=True, n_gpus_possible=(0, 1))


_cannon_name = "cannon"
_cannon_host = "cannon"
_cannon_exp = "/n/netscratch/iaifi_lab/Lab/tmccourt/orchestrator/training_experiments"
_cannon_job ="/n/home00/tmccourt/nano_llama/orchestrator_hooks/scripts/cannon_launcher.job"


cannon = Cluster(name=_cannon_name, ssh_host=_cannon_host,
                 experiment_dir=_cannon_exp,
                 job_file=_cannon_job,
                 partitions=[cannon_iaifi_gpu, cannon_iaifi_requeue, cannon_gpu_requeue])

