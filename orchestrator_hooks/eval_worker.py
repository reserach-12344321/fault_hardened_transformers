"""Cluster-side fault-eval worker, on the same contract as the training worker:

    python eval_worker.py <inputs_dir> <results_dir> <static_data_json> <max_seconds>

Instead of training, it re-evaluates one already-trained model under new fault conditions:
loads the run's FINAL checkpoint and, for the (k, p) pairs in eval_config.json, estimates
the faulted loss at each, writing the whole curve to one lightweight JSON.

Each point is sampled to a target PRECISION rather than a fixed budget, so every number on
the curve carries the same absolute uncertainty.
"""
import os
import sys
import json
import time
import contextlib
import io

import jax

from nano_llama.llama import Llama
from nano_llama.fault import FaultConfig
from nano_llama.train_core import TrainConfig, choose_loader_buffer_for_file
from nano_llama.token_data import SlidingLoader
from nano_llama.fault_eval import (EvalConfig, make_per_seq_eval_block, choose_eval_micro_batch,
                                   estimate_point, k_major_order)
from cluster_orchestrator import worker_api

DATA_KEY = "data"
RESULTS_NAME = "eval_results.json"


def _write_results(path: str, header: dict, points: list) -> None:
    """Atomically (re)write the results JSON: provenance header + the points scored so far."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({**header, "points": points}, f, indent=2)
    os.replace(tmp, path)


def _load_done(path: str) -> dict:
    """Already-scored points from a previous allocation: {(k, p): point_dict}."""
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return {(int(pt["k"]), float(pt["p"])): pt for pt in json.load(f).get("points", [])}


def main() -> None:
    inputs_dir, results_dir, static_data_json, max_seconds_arg = sys.argv[1:5]
    max_seconds = float(max_seconds_arg)

    # ---- eval spec + the run's own configs (read-only inputs) ----
    ec = EvalConfig.load(os.path.join(inputs_dir, "eval_config.json"))
    kp_pairs = list(ec.kp_pairs)                           # [(k, p), ...]; k STATIC -> int, p traced
    target_se = float(ec.target_se)                        # stop a point at se <= this (nats)
    min_evals, max_evals = int(ec.min_evals), int(ec.max_evals)
    batch_size_cfg = ec.batch_size                         # None -> choose adaptively on-node
    seed = int(ec.seed)

    tc = TrainConfig.load(os.path.join(inputs_dir, "train_config.json"))
    fc = FaultConfig.load(os.path.join(inputs_dir, "fault_config.json"))   # TRAIN fault (provenance)

    with open(static_data_json) as f:
        static = json.load(f)
    eval_file = static[DATA_KEY]                # a .bin FILE (e.g. the held-out test split) to score
    # Optional loader override in the static-data map (same contract as the training worker):
    # buffer_gb pins the SlidingLoader buffer (GB); absent -> auto-select by file size.
    buffer_override = static.get("buffer_gb")
    device = jax.local_devices()[0]

    # ---- load the trained model (config + weights) from final_model.{json,eqx} ----
    with contextlib.redirect_stdout(io.StringIO()):        # Llama.deserialize is chatty; hush it
        model = Llama.deserialize(os.path.join(inputs_dir, "final_model"))
    mc = model.config
    n_params = int(model.count_params())
    # D = tokens actually seen. The checkpoint meta records the FINAL step reached, which is
    # the truth even when the run stopped short of max_iters; fall back to nominal only if the
    # meta is missing.
    meta_path = os.path.join(inputs_dir, "checkpoint_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            final_step = int(json.load(f)["step"])
    else:
        final_step = int(tc.max_iters)
    n_train_tokens = int(final_step * tc.batch_size * mc.block_size)

    # The forward-only eval batch for this GPU: both the memory unit and the stopping
    # granularity, since the SE test runs on batch boundaries.
    batch_size = (int(batch_size_cfg) if batch_size_cfg else
                  choose_eval_micro_batch(mc, n_params, device, max_evals=max_evals))

    os.makedirs(results_dir, exist_ok=True)
    # One SlidingLoader over the eval file, shared by every point, which draw on their own
    # streams. Which split gets scored is an orchestration choice; the held-out test split is
    # the target for the final curves.
    buffer = buffer_override if buffer_override is not None else choose_loader_buffer_for_file(eval_file)
    # prefetch=False: an eval never slides. Every point restarts its step counter at 0 and runs
    # far fewer steps than refresh_every_steps, so one megablock serves the whole job and a
    # look-ahead would fetch a block no point ever samples.
    loader = SlidingLoader(eval_file, mc.block_size, seed=seed, batch_size=batch_size,
                           buffer_gb=buffer, prefetch=False)
    # Background-load the first megablock so the first point doesn't stall on a cold read,
    # and so a many-process array de-synchronizes its reads.
    loader.warm(0)
    block = make_per_seq_eval_block(device)
    base_key = jax.random.PRNGKey(seed)

    n_k = len({k for k, _ in kp_pairs})
    print(f"eval_worker: N={n_params/1e6:.2f}M D={n_train_tokens} | {len(kp_pairs)} (k,p) pairs "
          f"over {n_k} distinct k | target_se={target_se:g} nats "
          f"min_evals={min_evals} max_evals={max_evals} | "
          f"batch {batch_size}{'(auto)' if not batch_size_cfg else '(pinned)'} | "
          f"loader buffer={buffer if buffer is not None else 'whole-in-RAM'} | eval_file={eval_file}")

    # One per-job output: a provenance header + the points, rewritten after each one is scored.
    out_path = os.path.join(results_dir, RESULTS_NAME)
    header = {"n_params": n_params, "n_train_tokens": n_train_tokens, "final_step": final_step,
              "k_train": fc.k, "p_train": fc.p, "target_se": target_se,
              "min_evals": min_evals, "max_evals": max_evals, "batch_size": batch_size,
              "loader": "SlidingLoader", "buffer_gb": buffer}
    done = _load_done(out_path)                   # (k, p) already scored in a prior allocation
    # One slot per configured pair, filled in k-major order but written in kp_pairs order, so
    # the on-disk curve reads in the caller's order however the job was scheduled.
    scored: list = [done.get((int(k), float(p))) for k, p in kp_pairs]
    for i, pt in enumerate(scored):
        if pt is not None:
            print(f"  k={kp_pairs[i][0]} p={kp_pairs[i][1]:.4g}: already done -> skip")

    t0 = time.time()
    try:
        for i in k_major_order(kp_pairs):
            if scored[i] is not None:
                continue
            k, p = kp_pairs[i]
            # Both the data windows and the fault draw key off this point's INDEX, not the
            # running order, so the estimate is identical however the sweep was interrupted.
            point = estimate_point(block, model, loader, k, p, stream=i, batch_size=batch_size,
                                   key=jax.random.fold_in(base_key, i), target_se=target_se,
                                   min_evals=min_evals, max_evals=max_evals)
            scored[i] = {"k": int(point.k), "p": float(point.p), "loss": point.mean, "se": point.se,
                         "n": point.n_seq, "reached_target": bool(point.reached_target)}
            _write_results(out_path, header, [s for s in scored if s is not None])   # persist (atomic)
            flag = "" if point.reached_target else "  [!] hit max_evals, UNDER-CONVERGED"
            print(f"  k={k} p={p:.4g}: eval_loss {point.mean:.4f} ± {point.se:.4f} "
                  f"(n={point.n_seq}, rel_se={point.rel_se:.2%}){flag}")

            # Self-exit before the wall-time budget: scored points are on disk, so a resume skips them.
            if max_seconds > 0 and time.time() - t0 > max_seconds:
                n_left = sum(1 for s in scored if s is None)
                print(f"  max_seconds reached -> stopping ({n_left} pair(s) left for the next allocation)")
                return
    finally:
        loader.close()          # shut the prefetch thread down on every exit path (budget stop included)

    n_short = sum(1 for s in scored if not s["reached_target"])
    worker_api.mark_done(results_dir)
    print(f"eval_worker: DONE ({len(kp_pairs)} pairs -> {out_path})"
          + (f"  [!] {n_short} did not reach the SE target" if n_short else ""))


if __name__ == "__main__":
    main()
