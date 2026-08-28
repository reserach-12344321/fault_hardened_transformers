"""Sample a model's logits, on the same 4-arg worker contract as the other workers.

Holds a FIXED context set and scores it under many chips, keeping the full next-token
probability vector, to build the fault-marginalised predictive

    p_bar(. | context) = E_chip[ softmax(z(context, chip)) ]     (softmax THEN average)

for each eval fault (k, p). The marginal, not any single trajectory, is where the fault's
effect lives: a constant logit offset cancels in the softmax, so only the probability
average integrates the chip out correctly.
"""
import os
import sys
import json
import time
import contextlib
import io

import numpy as np
import jax
import jax.numpy as jnp

from nano_llama.llama import Llama
from nano_llama.fault import FaultConfig
from nano_llama.train_core import TrainConfig, choose_loader_buffer_for_file
from nano_llama.token_data import SlidingLoader
from nano_llama.fault_eval import (LogitSampleConfig, make_final_prob_block, make_clean_logit_block,
                                   choose_marginal_micro_batch, collect_marginal_predictive,
                                   context_mesh, context_sharding, k_major_order)
from cluster_orchestrator import worker_api

DATA_KEY = "data"
BASE_NAME = "base.npz"
JSON_NAME = "logit_marginals.json"


def point_name(i: int) -> str:
    """The npz holding point i's arrays. Zero-padded so the files sort in kp_pairs order."""
    return f"point_{int(i):03d}.npz"


def _kl(p, q, eps=1e-12):
    """Mean over contexts of KL(p || q) in nats; p, q are (C, V) probability rows."""
    p = np.clip(p, eps, 1.0); q = np.clip(q, eps, 1.0)
    return float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=1)))


def _entropy(p, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    return float(np.mean(-np.sum(p * np.log(p), axis=1)))


def _write_npz(path, **arrays):
    """Write one npz atomically (tmp + os.replace), so a death mid-write cannot leave a torn file."""
    np.savez_compressed(path + ".tmp.npz", **arrays)
    os.replace(path + ".tmp.npz", path)


def _write_digests(json_path, header, digests):
    """(Re)write the small digest JSON -- kilobytes, so rewriting it per point costs nothing."""
    tmp = json_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({**header, "points": digests}, f, indent=2)
    os.replace(tmp, json_path)


def _load_done(results_dir, json_path):
    """Points finished by a prior allocation: {(k, p): digest}."""
    if not os.path.isfile(json_path):
        return {}
    with open(json_path) as f:
        digests = json.load(f).get("points", [])
    return {(int(d["k"]), float(d["p"])): d for d in digests
            if os.path.isfile(os.path.join(results_dir, point_name(int(d["index"]))))}


def main() -> None:
    inputs_dir, results_dir, static_data_json, max_seconds_arg = sys.argv[1:5]
    max_seconds = float(max_seconds_arg)

    sc = LogitSampleConfig.load(os.path.join(inputs_dir, "logit_sample_config.json"))
    kp_pairs = list(sc.kp_pairs)
    tc = TrainConfig.load(os.path.join(inputs_dir, "train_config.json"))
    fc = FaultConfig.load(os.path.join(inputs_dir, "fault_config.json"))    # TRAIN fault = the arm

    with open(static_data_json) as f:
        static = json.load(f)
    eval_file = static[DATA_KEY]
    buffer_override = static.get("buffer_gb")
    # Context parallelism over every GPU: one replica per device, each holding its own slice
    # of the fixed context set. Trimmed to a divisor of n_contexts so micro-batches split.
    devices = jax.local_devices()
    mesh = context_mesh(int(sc.n_contexts), devices)
    n_dev = int(mesh.devices.size)
    device = devices[0]                      # representative device for the memory heuristics

    with contextlib.redirect_stdout(io.StringIO()):
        model = Llama.deserialize(os.path.join(inputs_dir, "final_model"))
    mc = model.config
    n_params = int(model.count_params())
    meta_path = os.path.join(inputs_dir, "checkpoint_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            final_step = int(json.load(f)["step"])
    else:
        final_step = int(tc.max_iters)
    n_train_tokens = int(final_step * tc.batch_size * mc.block_size)

    # The forward materialises (micro_batch, T, V) at the readout, so this caps how many
    # contexts go through at once. A memory knob only: which chips a context sees is keyed on
    # its global index, not its batch position.
    if sc.micro_batch:
        # A pinned micro-batch still has to tile the context set and split across the mesh.
        # Snap it down loudly rather than failing the job over a knob that cannot change any
        # number it reports.
        micro_batch = min(int(sc.micro_batch), int(sc.n_contexts))
        if micro_batch % n_dev or int(sc.n_contexts) % micro_batch:
            snapped = choose_marginal_micro_batch(mc, int(sc.n_contexts), n_params, device, n_dev)
            print(f"  [warn] pinned micro_batch={micro_batch} does not both divide n_contexts="
                  f"{sc.n_contexts} and split over {n_dev} device(s) -> using {snapped}")
            micro_batch = snapped
    else:
        micro_batch = choose_marginal_micro_batch(mc, int(sc.n_contexts), n_params, device, n_dev)

    # ---- the SHARED, FIXED context set: the same sequences for every job in the array ----
    buffer = buffer_override if buffer_override is not None else choose_loader_buffer_for_file(eval_file)
    ctx_loader = SlidingLoader(eval_file, mc.block_size, seed=int(sc.context_seed),
                               batch_size=int(sc.n_contexts), buffer_gb=buffer, prefetch=False)
    ctx_loader.warm(0)
    x_ctx, y_ctx = ctx_loader.batch(int(sc.n_contexts), step=0, stream=int(sc.context_stream))
    ctx_loader.close()
    # Placed once, sharded on the context axis: the set is reused by every chip of every
    # point, so the layout is paid for at startup rather than per forward.
    x_ctx = jax.device_put(jnp.asarray(x_ctx), context_sharding(mesh))
    targets = np.asarray(y_ctx[:, -1])                       # true next token at each context

    prob_block = make_final_prob_block(mesh)
    clean_block = make_clean_logit_block(mesh)
    base_key = jax.random.PRNGKey(int(sc.seed))

    # clean reference z0 (fault off), micro-batched to bound the (mb, T, V) readout
    with contextlib.redirect_stdout(io.StringIO()):
        z0 = np.concatenate([np.asarray(clean_block(model, x_ctx[a:a + micro_batch]))
                             for a in range(0, int(sc.n_contexts), micro_batch)], axis=0)   # (C, V)

    n_k = len({k for k, _ in kp_pairs})
    print(f"logit_sample_worker: N={n_params/1e6:.2f}M D={n_train_tokens} (k_train={fc.k}, "
          f"p_train={fc.p:g}) | {len(kp_pairs)} (k,p) over {n_k} k | C={sc.n_contexts} contexts, "
          f"M={sc.n_chips} chips, K={sc.n_groups} groups, n_raw={sc.n_raw}@ctx{sc.raw_context} | "
          f"micro_batch {micro_batch} over {n_dev} device(s)"
          + (f" [{len(devices) - n_dev} IDLE: {sc.n_contexts} contexts do not divide by "
             f"{len(devices)}]" if n_dev < len(devices) else "")
          + f" | eval_file={eval_file}")

    json_path = os.path.join(results_dir, JSON_NAME)
    os.makedirs(results_dir, exist_ok=True)
    header = {"n_params": n_params, "n_train_tokens": n_train_tokens, "final_step": final_step,
              "k_train": fc.k, "p_train": fc.p, "n_contexts": int(sc.n_contexts),
              "n_chips": int(sc.n_chips), "n_groups": int(sc.n_groups), "n_raw": int(sc.n_raw),
              "raw_context": int(sc.raw_context), "context_seed": int(sc.context_seed),
              "context_stream": int(sc.context_stream), "block_size": mc.block_size,
              "vocab_size": mc.vocab_size, "eval_file": eval_file}

    done = _load_done(results_dir, json_path)
    # The (k, p)-independent arrays: written once, on the first allocation that gets this far.
    base_path = os.path.join(results_dir, BASE_NAME)
    if not os.path.isfile(base_path):
        _write_npz(base_path, clean_logits=z0.astype(np.float32),
                   targets=targets.astype(np.int32),
                   context_tokens=np.asarray(x_ctx, dtype=np.int32))
    scored = [done.get((int(k), float(p))) for k, p in kp_pairs]
    for i, d in enumerate(scored):
        if d is not None:
            print(f"  k={kp_pairs[i][0]} p={kp_pairs[i][1]:.4g}: already done -> skip")

    t0 = time.time()
    try:
        for i in k_major_order(kp_pairs):
            if scored[i] is not None:
                continue
            k, p = kp_pairs[i]
            if p == 0.0:
                # The p=0 marginal is deterministic and equals the clean pass, which
                # clean_logits already carries. prepare_logit_sample_array drops p=0 from
                # kp_pairs, so this is belt and braces.
                scored[i] = {"index": i, "k": int(k), "p": 0.0, "n_chips": 0,
                             "pbar_entropy": _entropy(_softmax_rows(z0)),
                             "clean_entropy": _entropy(_softmax_rows(z0)),
                             "kl_pbar_from_clean": 0.0, "noise_floor_kl": 0.0, "is_clean": True}
                # A clean stub carries no arrays, but still needs a file so resume sees it as done.
                _write_npz(os.path.join(results_dir, point_name(i)))
                _write_digests(json_path, header, [d for d in scored if d is not None])
                continue
            res = collect_marginal_predictive(prob_block, model, x_ctx, k, p,
                                              n_chips=int(sc.n_chips), n_groups=int(sc.n_groups),
                                              n_raw=int(sc.n_raw), raw_context=int(sc.raw_context),
                                              micro_batch=micro_batch,
                                              base_key=jax.random.fold_in(base_key, i))
            pbar = res.p_bar()
            # split-group KL noise floor: KL between the mean of the first and second halves of groups
            half = int(sc.n_groups) // 2
            floor = (_kl(res.group_means[:half].mean(0), res.group_means[half:].mean(0))
                     if half >= 1 else float("nan"))
            scored[i] = {"index": i, "k": int(k), "p": float(p), "n_chips": res.n_chips,
                         "pbar_entropy": _entropy(pbar), "clean_entropy": _entropy(_softmax_rows(z0)),
                         "kl_pbar_from_clean": _kl(pbar, _softmax_rows(z0)),
                         "noise_floor_kl": floor}
            # Point file first, then the digest: the digest is the authority, so a death
            # between the two leaves an orphan npz the next allocation overwrites, never a
            # digest claiming a point whose arrays are missing.
            _write_npz(os.path.join(results_dir, point_name(i)),
                       groups=res.group_means, raw=res.raw)
            _write_digests(json_path, header, [d for d in scored if d is not None])
            s = scored[i]
            print(f"  k={k} p={p:.4g}: H(p_bar)={s['pbar_entropy']:.3f} "
                  f"(clean {s['clean_entropy']:.3f})  KL(p_bar||clean)={s['kl_pbar_from_clean']:.4f}  "
                  f"floor={floor:.5f}")

            if max_seconds > 0 and time.time() - t0 > max_seconds:
                n_left = sum(1 for d in scored if d is None)
                print(f"  max_seconds reached -> stopping ({n_left} pair(s) left for next allocation)")
                return
    finally:
        pass

    worker_api.mark_done(results_dir)
    print(f"logit_sample_worker: DONE ({len(kp_pairs)} pairs -> {results_dir}/point_*.npz)")


def _softmax_rows(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    main()
