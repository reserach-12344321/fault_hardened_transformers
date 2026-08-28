from dataclasses import dataclass, replace

from cluster_orchestrator.cluster import Cluster, PhysicalPartition, LogicalPartition
from cluster_orchestrator.resources import Resources, Gres


_inf = 1000


@dataclass
class GpuTypePartition(LogicalPartition):
    """A LogicalPartition that pins a job's generic gpu request to one specific card type."""
    gpu_type: str = ""

    def convert_resources(self, resources: Resources) -> Resources:
        """Gres("gpu", n) -> Gres("gpu:<gpu_type>", n). Anything else passes through."""
        gres = resources.gres
        if gres is None or gres.name != "gpu":
            return resources
        return replace(resources, gres=Gres(f"{gres.name}:{self.gpu_type}", gres.count))


@dataclass
class FixedGpuCountPartition(LogicalPartition):
    """A LogicalPartition that forces every gpu job to a fixed gres count."""
    n_gpus: int = 1

    def convert_resources(self, resources: Resources) -> Resources:
        """Gres("gpu", n) -> Gres("gpu", self.n_gpus)."""
        gres = resources.gres
        if gres is None or gres.name.split(":")[0] != "gpu":
            return resources
        if gres.count == self.n_gpus:
            return resources
        return replace(resources, gres=Gres(gres.name, self.n_gpus))


mit_normal_gpu = PhysicalPartition("mit_normal_gpu", max_in_flight=_inf, allow_n_pending=8)

engaging_resv_h200 = GpuTypePartition(name="engaging_resv_h200", physical=mit_normal_gpu,
                                      sbatch_args=['--account=rres_acc_tmccourt_2026-08-07_wfhwxix1',
                                                   '--qos=rres_qos_tmccourt_2026-08-07_wfhwxix1',
                                                   '--reservation=rres_tmccourt_2026-08-07_wfhwxix1',
                                                   '--time=12:00:00'],
                                      enabled=True, n_gpus_possible=(0, 1, 2,),
                                      gpu_type="h200")

engaging_resv_l40s = FixedGpuCountPartition(
    name="engaging_resv_l40s", physical=mit_normal_gpu,
    sbatch_args=['--account=rres_acc_tmccourt_2026-08-24_meihiwjf',
                 '--qos=rres_qos_tmccourt_2026-08-24_meihiwjf',
                 '--reservation=rres_tmccourt_2026-08-24_meihiwjf',
                 '--time=12:00:00'],
    enabled=True, n_gpus_possible=(1, 2, 4), n_gpus=2)

engaging_normal = LogicalPartition(name="engaging_normal", physical=mit_normal_gpu,
                                   sbatch_args=['--account=mit_amf_advanced_gpu',
                                                '--qos=mit_amf_advanced_gpu'],
                                   enabled=True, n_gpus_possible=(1,))

engaging_preemtible = LogicalPartition(
    name="mit_preemptable",
    physical=PhysicalPartition("mit_preemptable", max_in_flight=_inf, allow_n_pending=2),
    enabled=True, n_gpus_possible=(0, 1))

engaging = Cluster(name="engaging", ssh_host="engaging",
                   experiment_dir="/orcd/scratch/orcd/009/tmccourt/orchestrator/training_experiments",
                   job_file="/home/tmccourt/nano_llama/orchestrator_hooks/scripts/engaging_launcher.job",
                   partitions=[engaging_resv_l40s, engaging_preemtible, engaging_normal])
                   # engaging_resv_h200 -- expired 2026-07-29, kept above for reference
