"""Stamp a uniform resources.json onto every job folder of an array.

Every job must carry its cpu/mem/gres request before the array can be initialized.

    inject_resources.py /data/job_arrays/my_run -n 4 -m 48 -k 1

mem_gb / gpus of 0 mean "don't request it". Overwrites by default.
"""
import os
import sys
import argparse

from cluster_orchestrator import Resources, Gres
from cluster_orchestrator.resources import RESOURCES_FILE



def inject_resources(array_dir: str, n_cpus: int, mem_gb: int, n_gpus: int,
                     overwrite: bool = True) -> Resources:
    """Write the same Resources(n_cpus, mem, gres) into every immediate subdir of array_dir."""
    job_names = sorted(d for d in os.listdir(array_dir)
                       if os.path.isdir(os.path.join(array_dir, d)))
    assert job_names, f"no job subdirs in {array_dir}"

    res = Resources(
        n_cpus=n_cpus,
        mem=f"{mem_gb}G" if mem_gb > 0 else None,
        gres=Gres("gpu", n_gpus) if n_gpus > 0 else None,
    )
    print(f"injecting {res.to_sbatch_args()} into {len(job_names)} jobs under {array_dir}")

    n_written = 0
    for name in job_names:
        path = os.path.join(array_dir, name, RESOURCES_FILE)
        if os.path.exists(path) and not overwrite:
            print(f"  skip {name} ({RESOURCES_FILE} exists)")
            continue
        res.save(path)
        n_written += 1
    print(f"wrote {RESOURCES_FILE} to {n_written}/{len(job_names)} jobs")
    return res


if __name__ == "__main__":
    in_dir = "/media/trevor/data_flash/job_arrays/logit_marginals_d512_tpp/wave_2026-08-26-11-59-51"
    n_cpus = 1
    mem_gb = 48
    n_gpus = 1

    inject_resources(in_dir, n_cpus, mem_gb, n_gpus, overwrite=False)
