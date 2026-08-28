"""Assign each job its proportional share of the fleet, so nothing sticks out at the tail.

GPU-hours per job w_i are invariant to parallelism, so W = sum(w_i) is fixed however we
split it, and with G GPUs the ideal makespan is T = W / G. A job absorbs at most k_i GPUs,
so its floor wall-clock is w_i / k_i and it straggles exactly when w_i / k_i > T. Hence
k_i = the smallest allowed level >= w_i * G / W; a job that cannot reach T even at the
maximum level is flagged.

These are preemptable, randomly-ordered jobs, so G is the throughput denominator and not a
simultaneous-packing budget. Dry run by default; WRITE=True stamps resources.json.
"""
from __future__ import annotations

import os

os.environ['JAX_PLATFORMS'] = 'cpu'

import jax

jax.config.update("jax_platform_name", "cpu")

import os
import io
import glob
import contextlib
from collections import Counter
from typing import List, Sequence, Tuple

import jax

from nano_llama.llama import LlamaConfig, Llama
from nano_llama.train_core import TrainConfig

A100_TFLOP = 312e12 * 0.30    # bf16 dense peak * MFU (conservative; ranking is insensitive to this)


def _fwd_and_params(mc: LlamaConfig, cache: dict) -> Tuple[float, int]:
    k = (mc.n_embd, mc.n_layer, mc.vocab_size, mc.block_size, mc.multiple_of, mc.n_head)
    if k not in cache:
        with contextlib.redirect_stdout(io.StringIO()):
            m = Llama(mc, jax.random.PRNGKey(0))
        cache[k] = (m.forward_flops_per_token(), m.count_params())
    return cache[k]


def job_runtimes(sweep_dir: str) -> List[dict]:
    """Per-job estimated 1-GPU wall-clock (hours) = 3 * fwd_flops_per_token * D / (A100 peak*MFU)."""
    cache: dict = {}
    out = []
    for d in sorted(glob.glob(os.path.join(sweep_dir, "*"))):
        if not os.path.isfile(os.path.join(d, "model_config.json")):
            continue
        mc = LlamaConfig.load(os.path.join(d, "model_config.json"))
        tc = TrainConfig.load(os.path.join(d, "train_config.json"))
        f, N = _fwd_and_params(mc, cache)
        D = tc.max_iters * tc.batch_size * mc.block_size
        hrs = 3.0 * f * D / A100_TFLOP / 3600.0
        out.append(dict(dir=d, name=os.path.basename(d), n_embd=mc.n_embd, n_layer=mc.n_layer,
                        N=N, D=D, hrs_1gpu=hrs))
    return out


def assign_k(w: float, target_hours: float, allowed: Sequence[int]) -> int:
    """Smallest allowed GPU count k such that the job's floor wall-clock w/k <= target_hours."""
    for k in allowed:                       # allowed is sorted ascending
        if w / k <= target_hours:
            return k
    return allowed[-1]


def plan(sweep_dir: str, gpus_available: int, allowed_gpus: Sequence[int], write: bool,
         slack: float = 1.0, n_cpus: int = 4, mem_gb: int = 48) -> None:
    allowed = sorted(set(allowed_gpus))
    jobs = job_runtimes(sweep_dir)

    total_gpuh = sum(j["hrs_1gpu"] for j in jobs)
    target_hours = slack * total_gpuh / gpus_available           # T = slack * W / G

    for j in jobs:
        j["k"] = assign_k(j["hrs_1gpu"], target_hours, allowed)
        j["hrs_kgpu"] = j["hrs_1gpu"] / j["k"]
        j["sticks_out"] = j["hrs_kgpu"] > target_hours * 1.0001  # capped-out & still over T
    jobs.sort(key=lambda j: -j["hrs_1gpu"])

    makespan_1 = max(j["hrs_1gpu"] for j in jobs)
    makespan_k = max(j["hrs_kgpu"] for j in jobs)
    kdist = Counter(j["k"] for j in jobs)
    n_multi = sum(1 for j in jobs if j["k"] > 1)
    stragglers = [j for j in jobs if j["sticks_out"]]

    print(f"{len(jobs)} jobs | {total_gpuh:.0f} A100-GPU-h total (W, unchanged by DP)")
    print(f"fleet G = {gpus_available} GPU | allowed per-job {allowed} (cap {allowed[-1]}) | slack {slack:g}")
    print(f"target T = W/G{'' if slack == 1 else f' x {slack:g}'} = {target_hours:.1f}h "
          f"(the 'most jobs done' time; jobs above this stick out)")
    print(f"longest-job makespan:  {makespan_1:.0f}h (all 1-GPU)  ->  {makespan_k:.0f}h (by-share)  "
          f"[{makespan_1/makespan_k:.1f}x shorter tail]")
    print(f"multi-GPU jobs: {n_multi}/{len(jobs)} ({100*n_multi/len(jobs):.0f}%)  |  k-distribution: "
          f"{dict(sorted(kdist.items()))}")
    if stragglers:
        print(f"!! {len(stragglers)} job(s) still exceed T even at k={allowed[-1]} (unavoidable "
              f"stragglers; raise allowed cap or G to fix)")

    print(f"\n  {'run':>30} {'N(M)':>6} {'D(B)':>7} {'1GPU h':>8} {'k':>3} {'kGPU h':>8}  {'':>3}")
    for j in jobs[:14]:
        print(f"  {j['name']:>30} {j['N']/1e6:>5.0f}M {j['D']/1e9:>6.1f}B "
              f"{j['hrs_1gpu']:>8.1f} {j['k']:>3} {j['hrs_kgpu']:>8.1f}  {'<-!' if j['sticks_out'] else '':>3}")

    if write:
        from cluster_orchestrator import Resources, Gres
        from cluster_orchestrator.resources import RESOURCES_FILE
        for j in jobs:
            Resources(n_cpus=n_cpus, mem=f"{mem_gb}G", gres=Gres("gpu", j["k"])) \
                .save(os.path.join(j["dir"], RESOURCES_FILE))
        print(f"\nWROTE resources.json ({n_multi} multi-GPU) to {len(jobs)} jobs")
    else:
        print("\n(DRY RUN -- set WRITE=True to stamp resources.json)")


if __name__ == "__main__":
    INPUT_DIR = "/media/trevor/data_flash/job_arrays/d512_tpp_range_sweep_2026-08-21-10-43-45"
    GPUS_AVAILABLE = 20            # G: GPUs the sweep has access to right now (throughput denominator)
    ALLOWED_GPUS = [1, 2, 4]      # per-job options; max is the node/gang-scheduling cap
    SLACK = 1.0                   # target T = SLACK * W / G ; >1 loosens the tail, <1 tightens
    WRITE = False
    plan(INPUT_DIR, GPUS_AVAILABLE, ALLOWED_GPUS, WRITE, slack=SLACK)
