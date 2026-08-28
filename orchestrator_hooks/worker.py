"""Cluster-side nano_llama training worker for cluster_orchestrator.

    python worker.py <inputs_dir> <results_dir> <static_data_json> <max_seconds>

Reads the job's spec and seed from inputs_dir and the data dir from the static-data map,
resumes from results_dir/checkpoint if one exists, and trains to the target or until the
wall-time budget elapses -- self-exiting before SLURM kills it so a later allocation resumes
cleanly. results_dir is the only directory that round-trips to the orchestrator; on reaching
the target the worker calls worker_api.mark_done, the marker it reads on pull-back.
"""
import os
import sys
import json
import math
import subprocess

import jax
import equinox as eqx

from nano_llama.llama import LlamaConfig, Llama
from nano_llama.initializations import build_model
from nano_llama.fault import FaultConfig
from nano_llama.metrics import load_metrics, is_diverged
from nano_llama.train_core import (TrainConfig, build_optimizer, choose_micro_batch,
                                   load_checkpoint, run_training, choose_loader_buffer,
                                   resolve_eval_seqs)
from cluster_orchestrator import worker_api

DATA_KEY = "data"


def _tee_console(path: str):
    """Mirror stdout+stderr into `path` at the FILE-DESCRIPTOR level."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sys.stdout.flush(); sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))                        # keep originals to restore on close
    tee = subprocess.Popen(["tee", "-a", path], stdin=subprocess.PIPE)   # tee's stdout = original fd 1
    os.dup2(tee.stdin.fileno(), 1)                        # fd 1 -> tee pipe (writes go to SLURM log AND path)
    os.dup2(tee.stdin.fileno(), 2)
    tee.stdin.close()                                     # now fds 1/2 are the only pipe write-ends
    return tee, saved


def _close_tee(tee, saved) -> None:
    """Restore stdout/stderr so tee gets EOF and exits."""
    sys.stdout.flush(); sys.stderr.flush()
    os.dup2(saved[0], 1); os.dup2(saved[1], 2)
    os.close(saved[0]); os.close(saved[1])
    try:
        tee.wait(timeout=15)
    except Exception:
        tee.kill()


def main() -> None:
    inputs_dir, results_dir, static_data_json, max_seconds_arg = sys.argv[1:5]
    os.makedirs(results_dir, exist_ok=True)
    tee, saved = _tee_console(os.path.join(results_dir, "console.log"))  # full console -> results/ (append)
    try:
        _run(inputs_dir, results_dir, static_data_json, float(max_seconds_arg))
    finally:
        _close_tee(tee, saved)


def _run(inputs_dir: str, results_dir: str, static_data_json: str, max_seconds: float) -> None:
    # ---- spec + seed from the (read-only) inputs ----
    mc = LlamaConfig.load(os.path.join(inputs_dir, "model_config.json"))
    tc = TrainConfig.load(os.path.join(inputs_dir, "train_config.json"))
    fc = FaultConfig.load(os.path.join(inputs_dir, "fault_config.json"))
    with open(os.path.join(inputs_dir, "meta.json")) as f:
        seed = int(json.load(f)["seed"])
    # micro_batch_size is optional (TrainConfig drops it); honor a pinned value, else auto.
    with open(os.path.join(inputs_dir, "train_config.json")) as f:
        micro_batch_cfg = json.load(f).get("micro_batch_size")

    with open(static_data_json) as f:
        static = json.load(f)
    data_dir = static[DATA_KEY]
    # Optional loader override: buffer_gb pins the SlidingLoader buffer in GB, absent means
    # auto-select by train.bin size. Lets a deployment force the sliding path on an otherwise
    # RAM-sized corpus without touching the data.
    buffer_override = static.get("buffer_gb")
    # save_best=False stops the run writing results/best_model on an improving eval. A
    # DEPLOYMENT choice: it changes what a run leaves on disk, not what it computes, since
    # best_val is still tracked and recorded in the checkpoint meta. Worth ~4 B/param per run,
    # a second copy of the weights the analysis path never reads -- prepare_eval_array scores
    # the checkpoint's weights at the LAST step, deliberately not the best-scoring snapshot.
    save_best = bool(static.get("save_best", True))
    # Data parallelism over every GPU in the allocation: one mesh axis, one replica per device.
    devices = jax.local_devices()
    n_dev = len(devices)
    mesh = jax.make_mesh((n_dev,), ("data",))
    device = devices[0]   # representative device for the memory-based micro-batch heuristic

    # ---- build the model/optimizer/fault-spec skeletons from the config + seed ----
    key = jax.random.PRNGKey(seed)
    key, model_key = jax.random.split(key)
    model = build_model(mc, model_key, init_std=tc.init_std)          # skeleton + standard llama2.c init
    optim = build_optimizer(tc, model)             # single global AdamW; no shape-dependent scaling
    opt_state = optim.init(eqx.partition(model, eqx.is_array)[0])
    spec = fc.to_spec()

    # ---- resume from results/checkpoint if it exists, else start fresh (step 0) ----
    ckpt_dir = os.path.join(results_dir, "checkpoint")
    start, best = 0, math.inf
    loaded = load_checkpoint(ckpt_dir, model, opt_state)
    if loaded is not None:
        model, opt_state, start, best, key = loaded

    # No `start >= target -> mark_done` short-circuit here. A run preempted after its last
    # training step but before the final eval writes metrics.json resumes with start == target;
    # short-circuiting then skipped the eval, so metrics.json was never written and the run was
    # mislabeled. Falling through with n_steps = 0 still runs the final eval.

    # target = (tc.max_iters // tc.eval_interval) * tc.eval_interval
    target = tc.max_iters
    n_steps = max(0, target - start)

    mb = micro_batch_cfg or choose_micro_batch(mc, tc.batch_size, model.count_params(), device, n_dev)
    budget = max_seconds if max_seconds > 0 else None
    buffer = buffer_override if buffer_override is not None else choose_loader_buffer(data_dir)
    # Divergence threshold: divergence_loss_factor * untrained entropy (ln vocab), or None
    # for NaN/inf only.
    max_loss = (tc.divergence_loss_factor * math.log(mc.vocab_size)
                if tc.divergence_loss_factor is not None else None)
    # Resolved from the CONFIG alone, deliberately not from `mb`: tying the eval sample to the
    # training micro-batch shrank it exactly where the model is biggest, since mb is pinned to 1
    # on the top rungs.
    eval_seqs = resolve_eval_seqs(tc)
    print(f"worker: seed={seed} start={start} target={target} micro_batch={mb} n_dev={n_dev} "
          f"budget={budget} buffer={buffer} max_loss={max_loss} eval_seqs={eval_seqs} "
          f"save_best={save_best} data={data_dir}")

    *_, end = run_training(model, opt_state, optim, spec, data_dir, key, seed, start,
                           n_steps, best, tc.batch_size, mb, tc.eval_interval, eval_seqs,
                           results_dir, ckpt_dir, mesh, budget, buffer=buffer, max_loss=max_loss,
                           save_best=save_best, tc=tc)   # tc -> the realized HP table for run_log

    if end >= target:
        worker_api.mark_done(results_dir)
        print(f"worker: DONE at step {end}")
    elif is_diverged(load_metrics(results_dir)):
        # The run went NaN/inf and stopped early; resuming would only reproduce the divergence,
        # and the non-finite final metric already signals it downstream.
        worker_api.mark_done(results_dir)
        print(f"worker: DIVERGED at step {end} -- marking done (no resume)")
    else:
        print(f"worker: stopped at step {end} (budget) -- will resume next allocation")


if __name__ == "__main__":
    main()
