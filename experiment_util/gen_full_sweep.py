"""Generate the full scaling + fault sweep from (name, rung numbers, TPP ladder) families.

Every job gets the one locked recipe from gen_sweep, whose only shape-dependent knob is the
peak LR. A cell is one (rung, TPP, fault); rungs appearing in several families at disjoint
horizons do not collide, and a genuine duplicate is dropped and reported.

Resources are NOT written -- this sweep spans ~0.1M to ~0.9B params, so stamp them per-rung.
IDE-driven (edit the `# <-- SET` constants); run under scienv.
"""
from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"          # config generation only -- never touches a GPU

import json
import datetime
from typing import List, Sequence, Tuple

from nano_llama.fault import FaultConfig
from experiment_util.standard_models import MODELS
from experiment_util.gen_sweep import (
    write_job_array, spec_dirname, n_params, train_config_for, eval_overhead_frac,
    BASE_LR, BASE_WIDTH, MIN_LR_FRAC, WEIGHT_DECAY, BETA1, BETA2, ADAM_EPS, GRAD_CLIP, INIT_STD,
    GLOBAL_BATCH, TARGET_EVALS, MAX_EVAL_FRAC, EVAL_SEQS, WARMUP_FRAC,
)


def full_sweep_specs(families, faults, *, batch_size: int = GLOBAL_BATCH,
                     target_evals=TARGET_EVALS, max_eval_frac: float = MAX_EVAL_FRAC,
                     eval_seqs: int = EVAL_SEQS, n_params_fn=n_params):
    """(specs, rung_of_spec, dups) for the sweep."""
    seen: dict = {}
    specs: List[Tuple] = []
    rung_of_spec: List[int] = []
    dups: List[dict] = []
    npar_cache: dict = {}

    def npar(mc):                                  # memoized so overlapping families don't recount
        key = (mc.n_embd, mc.n_layer)
        if key not in npar_cache:
            npar_cache[key] = n_params_fn(mc)
        return npar_cache[key]

    for name, rung_numbers, tpps in families:
        bad = [r for r in rung_numbers if not (1 <= r <= len(MODELS))]
        if bad:
            raise ValueError(f"family {name!r}: rung numbers {bad} out of range 1..{len(MODELS)}")
        for r in rung_numbers:
            mc = MODELS[r - 1]
            N = npar(mc)
            for tpp in tpps:
                tc = train_config_for(mc, tpp, batch_size=batch_size, n_total=N,
                                      target_evals=target_evals, max_eval_frac=max_eval_frac,
                                      eval_seqs=eval_seqs)
                for fc in faults:
                    key = (mc.n_embd, mc.n_layer, tc.max_iters, fc.k, fc.p)
                    if key in seen:
                        dups.append(dict(family=name, first_seen=seen[key],
                                         shape=(mc.n_embd, mc.n_layer), max_iters=tc.max_iters,
                                         k=fc.k, p=fc.p))
                        continue
                    seen[key] = name
                    specs.append((mc, tc, fc))
                    rung_of_spec.append(r)
    return specs, rung_of_spec, dups, npar_cache


def write_full_sweep(out_dir, families, faults, *, replicas: int = 1, base_seed: int = 0,
                     batch_size: int = GLOBAL_BATCH, target_evals=TARGET_EVALS,
                     max_eval_frac: float = MAX_EVAL_FRAC, eval_seqs: int = EVAL_SEQS,
                     title: str = "Full sweep (standard parametrization)"):
    """Generate ONE (rung x TPP x fault) job array: build the specs, print the summary."""
    specs, _rungs, dups, npar_cache = full_sweep_specs(
        families, faults, batch_size=batch_size, target_evals=target_evals,
        max_eval_frac=max_eval_frac, eval_seqs=eval_seqs)

    # ---- summary -------------------------------------------------------------------------------------
    print(f"{title} -> {out_dir}")
    print(f"  parametrization: standard (llama2.c) -- no width/depth/duration scaling anywhere")
    print(f"  schedule/grid  : warmup {WARMUP_FRAC:.0%} of horizon, cosine to max_iters, "
          f"eval policy below")
    print(f"  hparams: peak LR {BASE_LR:.2e}*sqrt({BASE_WIDTH}/d) (min_lr = lr/{1/MIN_LR_FRAC:.0f}) | "
          f"wd {WEIGHT_DECAY:g} | betas ({BETA1:g}, {BETA2:g}) | eps {ADAM_EPS:g} | "
          f"clip {GRAD_CLIP:g} | init_std {INIT_STD:g} | batch {batch_size}")
    lrs = sorted({tc.learning_rate for _mc, tc, _fc in specs})
    print(f"  peak LR span across the ladder: {lrs[0]:.3e} .. {lrs[-1]:.3e}")
    for name, rungs, tpps in families:
        print(f"  {name:9s}: rungs {rungs[0]}-{rungs[-1]} ({len(rungs)}) x {len(tpps)} TPP {tpps} x "
              f"{len(faults)} faults = {len(rungs) * len(tpps) * len(faults)} jobs")
    npars = sorted(npar_cache.values())
    print(f"  unique jobs {len(specs)} (dropped {len(dups)} cross-family dup(s)) x {replicas} replicas "
          f"= {len(specs) * replicas} | n_params {npars[0]/1e6:.1f}M .. {npars[-1]/1e6:.0f}M")
    # training COST: what this array will actually burn, so an innocuous-looking TPP entry can't hide a 10x.
    tokens = [tc.max_iters * tc.batch_size * mc.block_size for mc, tc, _fc in specs]
    print(f"  train cost: {replicas * sum(tokens)/1e12:.2f}T tokens total | per-run "
          f"{min(tokens)/1e9:.3f}B .. {max(tokens)/1e9:.1f}B")
    # eval cadence: n_evals per run = intermediate on-grid evals + the guaranteed endpoint eval.
    n_evals = [(tc.max_iters - 1) // tc.eval_interval + 1 for _mc, tc, _fc in specs]
    fracs = [eval_overhead_frac(tc.eval_seqs, tc.batch_size, tc.eval_interval) for _mc, tc, _fc in specs]
    n_seqs = {tc.eval_seqs for _mc, tc, _fc in specs}
    assert n_seqs == {eval_seqs}, f"eval_seqs must be uniform across the ladder, got {sorted(n_seqs)}"
    print(f"  eval policy: {eval_seqs} seq/eval (uniform), target {target_evals} evals/run, cap "
          f"{max_eval_frac:.0%} compute -> {min(n_evals)}..{max(n_evals)} evals/run | "
          f"overhead {min(fracs):.2%}..{max(fracs):.2%}")
    if dups:
        ex = [f"{d['shape']} it{d['max_iters']} p{d['p']} ({d['family']} vs {d['first_seen']})"
              for d in dups[:3]]
        print(f"  ! dropped {len(dups)} cross-family duplicate(s), e.g. {ex}")

    # ---- replicas, then write the array --------------------------------------------------------------
    n_unique = len(specs)                       # BEFORE replication -- distinct (shape, horizon, fault)
    if replicas > 1:
        specs = [s for s in specs for _ in range(replicas)]
    written = write_job_array(specs, out_dir, name_fn=spec_dirname)   # d{d}_L{L}_it{it}_k{k}_p{p} (+_rK)

    # ---- meta.json seeds (resources are NOT written -- inject them per-rung, see the header) ---------
    for i, name in enumerate(sorted(d for d in os.listdir(out_dir)
                                    if os.path.isdir(os.path.join(out_dir, d)))):
        with open(os.path.join(out_dir, name, "meta.json"), "w") as f:
            json.dump({"seed": base_seed + i}, f, indent=2)

    # ---- manifest (provenance) -----------------------------------------------------------------------
    with open(os.path.join(out_dir, "full_sweep_manifest.json"), "w") as f:
        json.dump(dict(parametrization="standard",
                       recipe=dict(base_lr=BASE_LR, base_width=BASE_WIDTH, lr_rule="base_lr*sqrt(base_width/n_embd)",
                                   min_lr_frac=MIN_LR_FRAC, weight_decay=WEIGHT_DECAY,
                                   beta1=BETA1, beta2=BETA2, adam_eps=ADAM_EPS,
                                   grad_clip=GRAD_CLIP, init_std=INIT_STD, batch_size=batch_size),
                       replicas=replicas, base_seed=base_seed,
                       target_evals=target_evals, max_eval_frac=max_eval_frac, eval_seqs=eval_seqs,
                       families=[dict(name=n, rungs=list(r), tpps=list(t)) for n, r, t in families],
                       faults=[dict(p=fc.p, k=fc.k) for fc in faults],
                       # n_unique_jobs counts distinct (shape, horizon, fault) cells; n_jobs counts the
                       # dirs actually written, i.e. n_unique_jobs * replicas. Kept separate because
                       # conflating them reads as "192 unique x 2 replicas" for a 192-dir array.
                       n_unique_jobs=n_unique, n_jobs=len(written)), f, indent=2)

    print(f"\nwrote {len(written)} jobs to {out_dir} (model/train/fault + seeds; provenance in "
          f"full_sweep_manifest.json)")
    print(f"NEXT: python scripts/inject_resources.py {out_dir} ...   then launch_everything.py")
    return written


if __name__ == "__main__":
    # ---- output ------------------------------------------------------------------------ # <-- SET
    OUT_DIR = os.path.join("/media/trevor/data_flash/job_arrays",
                           "full_sweep_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

    # ---- families: (name, 1-based rung numbers, TPP ladder) ----------------------------- # <-- SET
    # FAMILIES = [
    #     ("standard", list(range(1, 19)),  [10, 20, 80, 320]),   # rungs 1-18
    #     ("large",    list(range(19, 25)), [2, 5, 10, 20]),      # rungs 19-24 (through d1344/L42)
    #     # ("extD",   list(range(1, 13)),  [2560, 5120, 10240]), # rungs 1-12 at high D
    # ]

    FAMILIES = [
        ("standard", list(range(7, 19)), [10, 20, 80, 320]),  # rungs 1-18
        ("large", list(range(19, 25)), [ 5, 10, 20]),  # rungs 19-24 (through d1344/L42)
        # ("extD",   list(range(1, 13)),  [2560, 5120, 10240]), # rungs 1-12 at high D
    ]

    # ---- fault grid --------------------------------------------------------------------- # <-- SET
    #FAULTS = [FaultConfig(p=0.0, k=4)]
    FAULTS = [FaultConfig(p=0.0, k=4),
              FaultConfig(p=0.005, k=4),
              FaultConfig(p=0.01, k=4),
              FaultConfig(p=0.015, k=4),
              FaultConfig(p=0.025, k=4),
              FaultConfig(p=0.03, k=4),
              FaultConfig(p=0.035, k=4),
              FaultConfig(p=0.04, k=4),
              FaultConfig(p=0.06, k=4),
              FaultConfig(p=0.08, k=4),
              FaultConfig(p=0.1, k=4),
              FaultConfig(p=0.12, k=4),
              FaultConfig(p=0.14, k=4),
              FaultConfig(p=0.16, k=4),
              FaultConfig(p=0.18, k=4),
              FaultConfig(p=0.2, k=4),
              ]

    REPLICAS  = 2               # seed-copies of every job (distinct _rK dirs, consecutive seeds)
    BASE_SEED = 2345258 #982779234 #1234212326 #424242314213

    # ---- eval cadence -------------------------------------------------------------------- # <-- SET
    TARGET_EVALS_  = 50         # at most this many evenly-spaced evals per run (None -> end-only)
    MAX_EVAL_FRAC_ = 0.02       # and never more than this fraction of training compute
    EVAL_SEQS_     = 1024       # sequences per eval, uniform across the ladder
    # ======================================================================================

    write_full_sweep(OUT_DIR, FAMILIES, FAULTS, replicas=REPLICAS, base_seed=BASE_SEED,
                     target_evals=TARGET_EVALS_, max_eval_frac=MAX_EVAL_FRAC_, eval_seqs=EVAL_SEQS_)
