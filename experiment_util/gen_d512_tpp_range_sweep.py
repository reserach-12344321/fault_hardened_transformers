"""One mid-ladder rung (d512) over a wide TPP range at many seeds: the duration axis at fixed N.

The full sweep buys breadth in N and pays for it in D-resolution and seeds. This is the
complement: hold the model at rung 15 and sweep the horizon over ~2 decades of TPP at many
replicas, so every D has a real seed distribution. That gives a well-resolved L(D) at fixed
N, and the seed spread as a function of D and p -- the error model the pooled fit needs.

A thin FAMILIES/REPLICAS wrapper over gen_full_sweep.write_full_sweep, so the recipe, grid
rule, eval policy, naming, seeding and manifest are all the full sweep's. At d512 the
sqrt-width peak LR is one fixed number for the whole array, which is what this sweep needs.

COST: the expensive one. Read the printed "train cost" line before launching.
"""
from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"          # config generation only -- never touches a GPU

import datetime

from nano_llama.fault import FaultConfig
from experiment_util.gen_full_sweep import write_full_sweep


if __name__ == "__main__":
    # ---- output ------------------------------------------------------------------------- # <-- SET
    OUT_DIR = "/media/trevor/data_flash/job_arrays/d512_tpp_range_sweep_" + \
              datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # ---- the single rung + its TPP ladder ------------------------------------------------ # <-- SET
    RUNG = 15                # 15 = d512_L13 (~49.5M). The other d512 rung is 16 = d512_L16 (~59.0M).
    # ~2x-spaced over ~2.1 decades. The last two points are ~3/4 of the array's cost; drop
    # them for a third of the bill, at the price of the overtrain end.
    TPPS = [5, 10, 20, 40, 80, 160, 320, 640]
    FAMILIES = [("d512_tpp", [RUNG], TPPS)]

    # ---- fault grid: run in batches of ~6 p-values ---------------------------------------- # <-- SET
    # DISTINCT p values only: write_full_sweep keys cells on (shape, max_iters, k, p) and drops
    # repeats as duplicates. Extra draws at one p are what REPLICAS is for.
    FAULTS = [FaultConfig(p=0.0,  k=4),
              FaultConfig(p=0.02, k=4),
              FaultConfig(p=0.04, k=4),
              FaultConfig(p=0.06, k=4),
              FaultConfig(p=0.08, k=4),
              FaultConfig(p=0.12, k=4)]


    REPLICAS = 8             # seed-copies per cell -- the point of this array (distinct _rK dirs)
    BASE_SEED = 77420916338  # sorted job i -> BASE_SEED + i; distinct from every other array's BASE_SEED
                             # so replicas here are independent draws, not repeats of existing runs

    # ---- eval cadence: identical to the full sweep (see its EVAL_SEQS comment) ------------ # <-- SET
    TARGET_EVALS_  = 50      # at most this many evenly-spaced evals per run (None -> end-only eval)
    MAX_EVAL_FRAC_ = 0.02    # cap eval compute at this fraction of training
    EVAL_SEQS_     = 1024    # uniform: the seed spread this array measures must not be contaminated by
                             # a D-dependent eval error bar
    # ======================================================================================

    write_full_sweep(OUT_DIR, FAMILIES, FAULTS, replicas=REPLICAS, base_seed=BASE_SEED,
                     target_evals=TARGET_EVALS_, max_eval_frac=MAX_EVAL_FRAC_, eval_seqs=EVAL_SEQS_,
                     title="d512 TPP-range x seeds sweep")
