"""The training loop and everything it needs: TrainConfig, the optimizer and LR schedule,
the micro-batch heuristic, the data-parallel jitted train/eval blocks, and checkpoint I/O.

orchestrator_hooks/worker.py is the outer loop: build model/optimizer/spec from the configs,
restore from checkpoint, call run_training for the remaining steps.
"""
import os
import math
import json
import time
import shutil
import socket
import datetime
import platform
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from nano_llama.config_base import ConfigMixin
from nano_llama.llama import Llama, LlamaConfig, FaultSpec
from nano_llama.train import step, per_sequence_loss, build_standard_optimizer
from nano_llama.token_data import SlidingLoader
from nano_llama.fault_eval import choose_eval_micro_batch, mean_se


@dataclass(frozen=True)
class TrainConfig(ConfigMixin):
    """Base training config."""
    batch_size: int = 128
    learning_rate: float = 1e-3
    max_iters: int = 10000
    warmup_iters: int = 100
    lr_decay_iters: int = 10000
    min_lr: float = 0.0
    eval_interval: int = 1000
    # Sequences scored by one in-training monitoring eval. A config quantity, so it
    # is machine-invariant; how they are chunked through the GPU is choose_eval_chunk's
    # separate runtime choice. Kept small: eval_interval_for's compute floor scales
    # with it, so a large value trades away the eval COUNT.
    eval_seqs: int = 1024
    weight_decay: float = 1e-1         # decoupled AdamW WD, applied to 2D matmul weights only
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8
    init_std: float = 0.02             # used only at init, hence here and not in LlamaConfig
    grad_clip: float = 1.0
    # Divergence early-stop, as a multiple of the untrained entropy (ln vocab): a block
    # whose mean loss climbs past it is diverging. None -> stop on NaN/inf only.
    divergence_loss_factor: Optional[float] = None


def build_lr_schedule(tc: TrainConfig) -> optax.Schedule:
    """Linear warmup to tc.learning_rate, then cosine to tc.min_lr over lr_decay_iters."""
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=tc.learning_rate, warmup_steps=tc.warmup_iters,
        decay_steps=tc.lr_decay_iters, end_value=tc.min_lr,
    )


def realized_hparams(tc: TrainConfig, model: Llama) -> dict:
    """The hyperparameters the optimizer actually receives for this (config, model)."""
    sched = build_lr_schedule(tc)
    cfg = model.config
    peak_step = max(0, int(tc.warmup_iters))
    peak_lr = float(sched(peak_step))
    # The two trainable groups differ only in weight decay.
    groups = {"decay": dict(peak_lr=peak_lr, eps=float(tc.adam_eps), weight_decay=float(tc.weight_decay)),
              "no_decay": dict(peak_lr=peak_lr, eps=float(tc.adam_eps), weight_decay=0.0)}
    return {
        "parametrization": "standard (llama2.c) -- no width/depth/duration scaling",
        "hparams": {"learning_rate": float(tc.learning_rate), "min_learning_rate": float(tc.min_lr),
                    "weight_decay": float(tc.weight_decay), "adam_eps": float(tc.adam_eps),
                    "beta1": float(tc.beta1), "beta2": float(tc.beta2),
                    "init_std": float(tc.init_std)},
        "schedule": {"warmup_iters": int(tc.warmup_iters), "lr_decay_iters": int(tc.lr_decay_iters),
                     "lr_at_0": float(sched(0)), "lr_at_peak": peak_lr,
                     "lr_at_half": float(sched(int(tc.max_iters // 2))),
                     "lr_at_end": float(sched(int(tc.max_iters)))},
        "groups": groups,
        "grad_clip": float(tc.grad_clip),
        "shape": {"n_embd": int(cfg.n_embd), "n_layer": int(cfg.n_layer)},
        "horizon": {"max_iters": int(tc.max_iters), "batch_size": int(tc.batch_size),
                    "block_size": int(cfg.block_size),
                    "tokens": int(tc.max_iters) * int(tc.batch_size) * int(cfg.block_size)},
    }


def print_realized_hparams(rh: dict) -> None:
    """Render realized_hparams as the banner's hyperparameter block."""
    h, s, hz = rh["hparams"], rh["schedule"], rh["horizon"]
    sh = rh["shape"]
    print(f"  param'n : {rh['parametrization']}")
    print(f"  shape   : d{sh['n_embd']} L{sh['n_layer']} | horizon {hz['tokens']:,} tok = "
          f"{hz['max_iters']:,} it x {hz['batch_size']} x {hz['block_size']}")
    print(f"  hparams : eta={h['learning_rate']:.4g} min={h['min_learning_rate']:.4g} | "
          f"wd={h['weight_decay']:.4g} | eps={h['adam_eps']:.3g} | "
          f"b1={h['beta1']:.6g} b2={h['beta2']:.6g} | clip={rh['grad_clip']:g} | "
          f"init_std={h['init_std']:.4g}")
    print(f"  lr sched: warmup {s['warmup_iters']:,} -> decay {s['lr_decay_iters']:,} | "
          f"peak {s['lr_at_peak']:.4g} | half {s['lr_at_half']:.4g} | end {s['lr_at_end']:.4g}")
    print(f"  {'group':<12}{'peak_lr':>12}{'eps':>12}{'weight_decay':>14}")
    for name, v in rh["groups"].items():
        print(f"  {name:<12}{v['peak_lr']:>12.4g}{v['eps']:>12.3g}{v['weight_decay']:>14.4g}")


def build_optimizer(tc: TrainConfig, model: Llama) -> optax.GradientTransformation:
    """Warmup-cosine AdamW: one schedule, epsilon and beta pair for every parameter."""
    return build_standard_optimizer(build_lr_schedule(tc), model, tc.beta1, tc.beta2,
                                    tc.adam_eps, tc.weight_decay, tc.grad_clip)


def choose_micro_batch(mc: LlamaConfig, batch_size: int, n_params: int, device: jax.Device,
                       n_dev: int = 1, mem_fraction: float = 0.8, reserve_gb: float = 3.0,
                       act_safety: float = 1.6, max_micro_batch: int = 32) -> int:
    """Largest micro-batch estimated to fit across `n_dev` devices.

    Budgets XLA's preallocated pool minus persistent state for one micro-batch's fwd+bwd
    activations -- 16*T*V logits, 24*L*T*d attention/residual, 6*L*T*ffn_hidden SwiGLU --
    then scales by `act_safety` for the temps and fragmentation the linear model misses.
    Returns the GLOBAL per-step batch, constrained to divide batch_size and to be a
    multiple of n_dev.
    """
    try:
        total = int(device.memory_stats()["bytes_limit"])
    except Exception:
        total = 16 * 1024**3   # fallback if the backend doesn't report memory
    T, V, d, L, m = mc.block_size, mc.vocab_size, mc.n_embd, mc.n_layer, mc.multiple_of
    ffn_hidden = m * ((int(2 * (4 * d) / 3) + m - 1) // m)   # SwiGLU hidden dim (matches the model)
    persistent = 16 * n_params                              # params(4) + AdamW mu,nu(8) + grads(4)
    budget = total * mem_fraction - persistent - reserve_gb * 1024**3
    per_seq = act_safety * (16 * T * V + 24 * L * T * d + 6 * L * T * ffn_hidden)  # bytes/seq, one device
    mb_local = max(1, int(budget // per_seq))
    mb = min(mb_local * n_dev, batch_size, max_micro_batch * n_dev)
    # Must divide batch_size (clean gradient accumulation) and be a multiple of n_dev (even shard).
    while mb > n_dev and (batch_size % mb != 0 or mb % n_dev != 0):
        mb -= 1
    if mb % n_dev != 0 or batch_size % mb != 0:
        mb = n_dev                                          # batch_size % n_dev == 0 is asserted upstream
    return max(1, mb)


def resolve_eval_seqs(tc: TrainConfig) -> int:
    """tc.eval_seqs snapped up to a whole multiple of batch_size."""
    return int(math.ceil(max(1, int(tc.eval_seqs)) / tc.batch_size) * tc.batch_size)


def choose_eval_chunk(mc: LlamaConfig, n_eval_seq: int, n_params: int, device: jax.Device,
                      n_dev: int = 1) -> int:
    """How many sequences to push through the device at once during an eval."""
    cap = choose_eval_micro_batch(mc, n_params, device, max_evals=n_eval_seq)
    n_dev = max(1, n_dev)
    if n_eval_seq % n_dev != 0:
        return n_dev
    # d = n_dev * j fits and divides n_eval_seq  <=>  j divides (n_eval_seq // n_dev) and j <= cap//n_dev
    j = _largest_divisor_leq(n_eval_seq // n_dev, max(1, cap // n_dev))
    return max(n_dev, n_dev * j)


# Amortized jitted blocks: one dispatch per eval interval. State is replicated over the mesh
# and batches sharded on their batch axis, via shard_map rather than the jit
# auto-partitioner -- the embedding lookup is a gather with a sharded index the
# auto-partitioner cannot place, so it errors or all-gathers the batch and kills the
# parallelism. step() then all-reduces the gradients explicitly via pmean.
_REPL = P()                          # replicated across the "data" axis
_SHARD = P(None, "data", None)       # staged (n_steps, batch, block_size): shard the batch axis


def _make_train_block(optim: optax.GradientTransformation, accum_steps: int,
                      mesh: jax.sharding.Mesh, donate: str = "all-except-first") -> Callable:
    """A jitted+shard_mapped fn running len(xs) optimizer steps in one dispatch."""
    cache = {}

    def _block(model: Llama, opt_state: optax.OptState, xs: jax.Array, ys: jax.Array,
               base_key: jax.Array, start_step: int, spec: FaultSpec):
        # Split the model into array leaves (which shard_map maps over) and static leaves (Python ints
        # like n_head, the config -- must stay Python values, so we close over them rather than pass
        # them through shard_map). `static` is constant across a run, so build the mapped fn once.
        arrays, static = eqx.partition(model, eqx.is_array)
        idxs = start_step + jnp.arange(xs.shape[0], dtype=jnp.int32)   # global step index per scanned step

        if "fn" not in cache:
            def _sharded(arrays, opt_state, xs, ys, seq_keys, spec):
                def one_step(carry, batch):
                    arrays, opt_state = carry
                    x, y, ks = batch                                  # ks: this step's per-sequence keys
                    m = eqx.combine(arrays, static)
                    m, opt_state, loss = step(m, opt_state, x, y, ks, optim, accum_steps,
                                              spec, axis_name="data")
                    new_arrays, _ = eqx.partition(m, eqx.is_array)
                    return (new_arrays, opt_state), loss

                (arrays, opt_state), losses = jax.lax.scan(
                    one_step, (arrays, opt_state), (xs, ys, seq_keys))
                return arrays, opt_state, losses

            smapped = jax.shard_map(
                _sharded, mesh=mesh,
                # seq_keys is (n_steps, batch, 2) -> _SHARD slices the batch axis, so each device gets
                # exactly its sequences' keys (assigned by GLOBAL index -> device-count invariant).
                in_specs=(_REPL, _REPL, _SHARD, _SHARD, _SHARD, _REPL),
                out_specs=(_REPL, _REPL, _REPL), check_vma=False)

            # `keep` bundles the DO-NOT-DONATE args (base_key reused across blocks; idxs/spec cheap)
            # into the first positional arg so donate="all-except-first" spares them while still
            # donating arrays/opt_state/xs/ys.
            def _wrapped(keep, arrays, opt_state, xs, ys):
                base_key, idxs, spec = keep
                # Per-step key = fold_in(base_key, global step); per-sequence keys = split over the
                # GLOBAL batch. Built here (replicated, before shard_map) purely from the global index,
                # so nothing depends on the device count; shard_map then routes each key to its sequence.
                step_keys = jax.vmap(lambda i: jax.random.fold_in(base_key, i))(idxs)   # (n_steps, 2)
                seq_keys = jax.vmap(lambda sk: jax.random.split(sk, xs.shape[1]))(step_keys)  # (n_steps, batch, 2)
                return smapped(arrays, opt_state, xs, ys, seq_keys, spec)

            cache["fn"] = eqx.filter_jit(_wrapped, donate=donate)

        arrays, opt_state, losses = cache["fn"]((base_key, idxs, spec), arrays, opt_state, xs, ys)
        return eqx.combine(arrays, static), opt_state, losses

    return _block


def _make_eval_block(mesh: jax.sharding.Mesh) -> Callable:
    """A jitted+shard_mapped fn computing forward-only per-sequence mean CE under faults."""

    cache = {}

    def _eval(model: Llama, xs: jax.Array, ys: jax.Array, spec: FaultSpec, seq_keys: jax.Array):
        arrays, static = eqx.partition(model, eqx.is_array)

        if "fn" not in cache:
            def _sharded(arrays, xs, ys, spec, seq_keys):
                model = eqx.combine(arrays, static)

                def body(_, batch):
                    x, y, ks = batch                                  # ks: this chunk's per-seq keys
                    return None, per_sequence_loss(model, x, y, spec=spec, keys=ks)

                _, losses = jax.lax.scan(body, None, (xs, ys, seq_keys))
                return losses                                         # (n_chunk, eval_mb/n_dev)

            cache["fn"] = eqx.filter_jit(jax.shard_map(
                _sharded, mesh=mesh,
                # seq_keys (n_chunk, eval_mb, 2) sharded on the sequence axis, like the batches.
                in_specs=(_REPL, _SHARD, _SHARD, _REPL, _SHARD),
                out_specs=P(None, "data"), check_vma=False))

        return cache["fn"](arrays, xs, ys, spec, seq_keys)

    return _eval


# =============================================================================
# Checkpointing (model + optimizer + step + best + fault key), atomic-ish swap
# =============================================================================
def save_checkpoint(checkpoint_dir: str, model: Llama, opt_state: optax.OptState,
                    step_num: int, best: float, key: jax.Array) -> None:
    """Write the resumable state into checkpoint_dir."""
    tmp, bak = checkpoint_dir + ".tmp", checkpoint_dir + ".bak"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    eqx.tree_serialise_leaves(os.path.join(tmp, "model.eqx"), model)
    eqx.tree_serialise_leaves(os.path.join(tmp, "opt_state.eqx"), opt_state)
    with open(os.path.join(tmp, "meta.json"), "w") as f:
        json.dump(dict(step=int(step_num), best_val_loss=float(best),
                       key=[int(x) for x in np.asarray(key)]), f)
    if os.path.exists(checkpoint_dir):
        os.rename(checkpoint_dir, bak)
    os.rename(tmp, checkpoint_dir)
    if os.path.exists(bak):
        shutil.rmtree(bak)


def load_checkpoint(checkpoint_dir: str, model: Llama, opt_state: optax.OptState
                    ) -> Optional[Tuple[Llama, optax.OptState, int, float, jax.Array]]:
    """(model, opt_state, step, best, key) if checkpoint_dir holds a checkpoint, else None."""
    meta_path = os.path.join(checkpoint_dir, "meta.json")
    if not os.path.isfile(meta_path):
        return None
    model = eqx.tree_deserialise_leaves(os.path.join(checkpoint_dir, "model.eqx"), model)
    opt_state = eqx.tree_deserialise_leaves(os.path.join(checkpoint_dir, "opt_state.eqx"), opt_state)
    with open(meta_path) as f:
        m = json.load(f)
    return (model, opt_state, int(m["step"]), float(m["best_val_loss"]),
            jnp.asarray(m["key"], dtype=jnp.uint32))


# =============================================================================
# Shared inner training loop (data-source agnostic)
# =============================================================================
CHECKPOINT_SECONDS = float(os.environ.get("NANO_LLAMA_CHECKPOINT_SECONDS", 600.0))
# Cap the staged training block so device_put'ing (chunk, batch, block) int32 for both x
# and y can't OOM the GPU. chunk grows with eval_interval, and choose_micro_batch does not
# count the staged block, so probe_checkpoint_iter bounds chunk against this budget.
STAGE_BUDGET_GB = float(os.environ.get("NANO_LLAMA_STAGE_BUDGET_GB", 1.0))
# The loader step the in-training eval draws its probe at -- a constant, not the current
# step, so every eval scores the same sequences under the same chips and the metrics curve
# carries no resampling noise. The price is a small fixed offset in the absolute level;
# unbiased numbers come from the offline fault-eval worker on the test split.
EVAL_ANCHOR_STEP = 0


def _largest_divisor_leq(n: int, cap: int) -> int:
    """Largest divisor of n that is <= cap (>= 1)."""
    for d in range(min(cap, n), 0, -1):
        if n % d == 0:
            return d
    return 1


class _AsyncCheckpointer:
    """Writes checkpoints on a background thread, one at a time."""

    def __init__(self, checkpoint_dir: str):
        self.dir = checkpoint_dir
        self._exec = ThreadPoolExecutor(max_workers=1)
        self._pending = None      # Future of the in-flight write
        self._last_step = None    # step of the last SUBMITTED (not skipped) write

    def maybe_write(self, model, opt_state, key, step, best) -> None:
        if self._pending is not None and not self._pending.done():
            print(f"  [ckpt] prior write still in flight -- skipping checkpoint at step {step}")
            return
        if self._pending is not None:
            self._pending.result()                    # completed: surface any write error
        snap = jax.device_get((model, opt_state, key))
        self._pending = self._exec.submit(save_checkpoint, self.dir, snap[0], snap[1], step, best, snap[2])
        self._last_step = step

    def finish(self, model, opt_state, key, step, best) -> None:
        if self._pending is not None:
            self._pending.result()
        if self._last_step != step:                   # last submitted write didn't cover `step`
            snap = jax.device_get((model, opt_state, key))
            save_checkpoint(self.dir, snap[0], snap[1], step, best, snap[2])
        self._exec.shutdown(wait=True)


def probe_checkpoint_iter(model: Llama, opt_state: optax.OptState, optim: optax.GradientTransformation,
                          spec: FaultSpec, key: jax.Array, batch_size: int, micro_batch: int,
                          eval_interval: int, mesh: jax.sharding.Mesh,
                          checkpoint_seconds: float = CHECKPOINT_SECONDS,
                          stage_budget_gb: float = STAGE_BUDGET_GB) -> Tuple[int, Optional[float]]:
    """Size the checkpoint block (in optimizer steps) to ~checkpoint_seconds of wall-clock."""
    if eval_interval < 2:
        return max(1, eval_interval), None
    accum_steps = batch_size // micro_batch
    replicated = NamedSharding(mesh, P())
    shard_batch = NamedSharding(mesh, P(None, "data", None))
    m = eqx.filter_shard(model, replicated)
    o = eqx.filter_shard(opt_state, replicated)
    base_train, _ = jax.random.split(eqx.filter_shard(key, replicated))
    # donate="none": the probe calls tb twice on the SAME (m, o, zx) to separate compile
    # from timing, and m/o may alias the real model/opt_state -- donating them would
    # delete a buffer the second call still needs. The loop's own block does donate.
    tb = _make_train_block(optim, accum_steps, mesh, donate="none")
    zx = jax.device_put(jnp.zeros((1, batch_size, model.config.block_size), dtype=jnp.int32), shard_batch)
    tb(m, o, zx, zx, base_train, 0, spec)[2].block_until_ready()               # compile (discarded)
    t0 = time.time()
    tb(m, o, zx, zx, base_train, 0, spec)[2].block_until_ready()               # time (discarded)
    step_time = max(1e-6, time.time() - t0)
    ckpt_interval = max(1, round(checkpoint_seconds / step_time))
    # Memory cap: the loop stages (chunk, batch, block) int32 for x and y (8 bytes/token),
    # sharded over the mesh, so per-device it is chunk * (batch/n_dev) * block * 8.
    n_dev = max(1, mesh.devices.size)
    per_dev_batch = max(1, batch_size // n_dev)
    stage_cap = max(1, int(stage_budget_gb * 1024**3) // (per_dev_batch * model.config.block_size * 8))
    # Deliberately NOT snapped to a divisor of eval_interval: the loop clamps every block
    # to the distance to the next eval boundary, so evals stay on the grid for any block
    # size. eval_interval is often prime, and snapping collapsed the block to 1-2 steps.
    chunk = max(1, min(ckpt_interval, stage_cap, eval_interval))
    return chunk, step_time


RUN_LOG = "run_log.jsonl"
# Training-loss series, one appended line per training block: {step, train_loss, n_steps}.
# Its own file rather than metrics.json, whose consumers read the run's final loss as the
# max-step record's VAL loss -- mixing the two record kinds would make a diverged run read
# as healthy. Appended, so it accumulates across resumes like run_log.
TRAIN_LOG = "train_loss.jsonl"
_SLURM_KEYS = ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_PARTITION", "SLURM_CLUSTER_NAME",
               "SLURM_JOB_NODELIST", "SLURMD_NODENAME", "SLURM_NODEID", "SLURM_PROCID",
               "SLURM_LOCALID", "SLURM_JOB_GPUS", "SLURM_GPUS", "CUDA_VISIBLE_DEVICES")


def log_run_provenance(model: Llama, mesh: jax.sharding.Mesh, spec: FaultSpec, start_step: int,
                       end_step: int, n_steps: int, batch_size: int, micro_batch: int, accum_steps: int,
                       eval_interval: int, eval_seqs: int, checkpoint_iter: int, results_dir: str,
                       checkpoint_dir: str, extra: Optional[dict] = None) -> dict:
    """Print the start-of-segment banner and append a provenance record to run_log.jsonl."""
    devs = list(mesh.devices.flat)
    cfg = model.config
    slurm = {k: os.environ[k] for k in _SLURM_KEYS if k in os.environ}
    rec = {
        "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": "resume" if start_step > 0 else "start",
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "python": platform.python_version(),
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "n_devices": len(devs),
        "device_kind": devs[0].device_kind if devs else None,
        "slurm": slurm,
        "steps": {"start": int(start_step), "end": int(end_step), "n": int(n_steps)},
        "batch_size": int(batch_size), "micro_batch": int(micro_batch), "accum_steps": int(accum_steps),
        "eval_interval": int(eval_interval), "eval_seqs": int(eval_seqs),
        "checkpoint_iter": int(checkpoint_iter),
        "fault": {"p": float(spec.p), "k": int(spec.k)},
        "model": {"n_embd": cfg.n_embd, "n_layer": cfg.n_layer, "n_head": cfg.n_head,
                  "n_kv_head": cfg.n_kv_head, "vocab_size": cfg.vocab_size, "block_size": cfg.block_size,
                  "dtype": str(cfg.dtype), "n_params": int(model.count_params())},
        "results_dir": results_dir, "checkpoint_dir": checkpoint_dir,
    }
    if extra:
        rec.update(extra)

    # ---- verbose banner ----
    n_dev = max(1, rec["n_devices"])
    sd = rec.get("step_seconds")
    probe_s = f" | probe {sd * 1e3:.1f} ms/step" if sd else ""
    buf = rec.get("buffer_gb")
    loader_s = f" | loader {rec['loader']}" + (f" ({buf}GB buffer)" if buf else "") if "loader" in rec else ""
    bar = "=" * 78
    print(f"\n{bar}\n  TRAIN SEGMENT [{rec['event']}]  {rec['time']}  pid {rec['pid']}\n{bar}")
    print(f"  host    : {rec['host']}  |  {rec['n_devices']}x {rec['device_kind']}  "
          f"(backend {rec['backend']}, jax {rec['jax_version']}, py {rec['python']})")
    if slurm:
        print(f"  slurm   : job {slurm.get('SLURM_JOB_ID', '?')} | part {slurm.get('SLURM_JOB_PARTITION', '?')} "
              f"| node {slurm.get('SLURMD_NODENAME', '?')} | gpus "
              f"{slurm.get('SLURM_JOB_GPUS', slurm.get('CUDA_VISIBLE_DEVICES', '?'))}")
    print(f"  model   : d={cfg.n_embd} L={cfg.n_layer} h={cfg.n_head}/{cfg.n_kv_head} V={cfg.vocab_size} "
          f"T={cfg.block_size} {cfg.dtype} | {rec['model']['n_params']:,} params")
    print(f"  steps   : {start_step}..{end_step}  ({n_steps} this segment)")
    print(f"  batch   : global {batch_size} = {micro_batch} micro x {accum_steps} accum  "
          f"(per-device {micro_batch // n_dev} seq/step over {rec['n_devices']} dev)")
    print(f"  cadence : eval every {eval_interval} ({eval_seqs} seq) | checkpoint every "
          f"{checkpoint_iter} steps{probe_s}")
    print(f"  data    : seed {rec.get('seed', '?')} | {rec.get('data_dir', '?')}{loader_s} "
          f"| fault p={float(spec.p):.3g} k={spec.k}")
    if rec.get("hparams"):
        print_realized_hparams(rec["hparams"])
    print(f"  results : {results_dir}\n{bar}\n")

    # ---- append provenance record (one line per node the job ran on) ----
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, RUN_LOG), "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def run_training_loop(model: Llama, opt_state: optax.OptState, optim: optax.GradientTransformation,
                      spec: FaultSpec, key: jax.Array, start_step: int, n_steps: int, best_val: float,
                      batch_size: int, micro_batch: int, eval_interval: int, eval_seqs: int,
                      results_dir: str, checkpoint_dir: str, mesh: jax.sharding.Mesh,
                      train_loader: SlidingLoader, val_loader: SlidingLoader, checkpoint_iter: int,
                      max_seconds: float = None, run_meta: Optional[dict] = None,
                      max_loss: Optional[float] = None, eval_chunk: Optional[int] = None,
                      save_best: bool = True
                      ) -> Tuple[Llama, optax.OptState, jax.Array, float, int]:
    """The data-parallel block loop, driven by two loaders the caller owns and closes.

    Two decoupled cadences: an EVAL every eval_interval steps (faulted loss on a fixed val
    probe, plus best_model when `save_best`) and a CHECKPOINT every `checkpoint_iter`,
    written asynchronously so a slow shared FS never blocks the loop. `best_val` is tracked
    and returned whether or not it is written. `eval_chunk` is a memory choice that cannot
    move the numbers, since windows and fault keys are addressed by global sequence index.

    Returns (model, opt_state, key, best_val, final_step).
    """

    n_dev = mesh.devices.size
    assert batch_size % n_dev == 0, f"batch_size {batch_size} must be divisible by n_dev {n_dev}"
    assert micro_batch % n_dev == 0, f"micro_batch {micro_batch} must be divisible by n_dev {n_dev}"
    accum_steps = batch_size // micro_batch
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Replicated state, batches sharded on their batch axis. filter_shard rather than
    # jax.device_put so only ARRAY leaves are placed -- the model's static Python ints
    # must stay Python ints under jit.
    replicated = NamedSharding(mesh, P())
    shard_batch = NamedSharding(mesh, P(None, "data", None))
    model = eqx.filter_shard(model, replicated)
    opt_state = eqx.filter_shard(opt_state, replicated)
    key = eqx.filter_shard(key, replicated)
    # `key` is the run's constant base key, never threaded or advanced; per-step fault keys
    # are fold_in(base, global_step). Split once into train/eval namespaces so a train and
    # an eval key at the same step differ.
    base_train, base_eval = jax.random.split(key)

    # ---- fixed monitoring probe: size (config) and chunking (memory) resolved independently ----
    n_eval_seq = int(math.ceil(max(1, eval_seqs) / batch_size) * batch_size)   # whole nominal draws
    eval_mb = eval_chunk if eval_chunk is not None else choose_eval_chunk(
        model.config, n_eval_seq, model.count_params(), mesh.devices.flat[0], n_dev)
    assert n_eval_seq % eval_mb == 0 and eval_mb % n_dev == 0, (
        f"eval_chunk {eval_mb} must divide eval_seqs {n_eval_seq} and be a multiple of n_dev {n_dev}")
    n_eval_chunks = n_eval_seq // eval_mb
    # Chunks staged per dispatch, under the same per-device budget the train block respects.
    per_chunk_bytes = max(1, (eval_mb // n_dev) * model.config.block_size * 8)   # x AND y, int32
    eval_group = max(1, min(n_eval_chunks, int(STAGE_BUDGET_GB * 1024**3) // per_chunk_bytes))

    train_block = _make_train_block(optim, accum_steps, mesh)
    eval_block = _make_eval_block(mesh)
    best_model_path = os.path.join(results_dir, "best_model")
    metrics_path = os.path.join(results_dir, "metrics.json")
    train_log_path = os.path.join(results_dir, TRAIN_LOG)
    metrics = []
    if os.path.isfile(metrics_path):
        with open(metrics_path) as f:
            metrics = [m for m in json.load(f) if m["step"] < start_step]

    end_step = start_step + n_steps
    chunk = max(1, min(checkpoint_iter, eval_interval))     # block size (raw); evals stay on the grid

    log_run_provenance(model, mesh, spec, start_step, end_step, n_steps, batch_size, micro_batch,
                       accum_steps, eval_interval, n_eval_seq, chunk, results_dir, checkpoint_dir,
                       extra=dict(run_meta or {}, eval_chunk=int(eval_mb)))

    ckpt = _AsyncCheckpointer(checkpoint_dir)

    # Host-side I/O accounting, logged to run_log at the end. The first train-block read is
    # the cold megablock load; val_read is the one-off read of the probe below, since evals
    # themselves do no I/O.
    io = dict(first_block_read_s=None, train_stage_s=0.0, val_read_s=0.0, n_evals=0)

    # The probe: one fixed set of sequences with one fixed set of chips, drawn here and
    # reused by every eval in this segment, so an eval is just a forward sweep. Keys are
    # split over the whole set, so sequence i always meets chip i however the set is
    # chunked or spread over devices.
    t_io = time.time()
    px, py = val_loader.block(n_eval_seq // batch_size, batch_size, EVAL_ANCHOR_STEP, stream=1)
    probe_x = np.asarray(px).reshape(n_eval_chunks, eval_mb, -1)
    probe_y = np.asarray(py).reshape(n_eval_chunks, eval_mb, -1)
    probe_keys = jax.random.split(base_eval, n_eval_seq).reshape(n_eval_chunks, eval_mb, -1)
    io["val_read_s"] = time.time() - t_io

    def do_eval(model, step_now, best):
        """Score the fixed val probe under the run's fault spec."""
        t_ev = time.time()
        s = np.zeros(2)          # sums of L, L^2  (float64 host accumulators)
        for g0 in range(0, n_eval_chunks, eval_group):
            g1 = min(n_eval_chunks, g0 + eval_group)
            xs = jax.device_put(jnp.asarray(probe_x[g0:g1]), shard_batch)
            ys = jax.device_put(jnp.asarray(probe_y[g0:g1]), shard_batch)
            ks = jax.device_put(probe_keys[g0:g1], shard_batch)
            losses = np.asarray(eval_block(model, xs, ys, spec, ks), dtype=np.float64).ravel()
            s += (losses.sum(), (losses * losses).sum())
        va, va_se = mean_se(s[0], s[1], n_eval_seq)
        io["n_evals"] += 1
        print(f"step {step_now}: val {va:.4f}+-{va_se:.4f} (p={float(spec.p):g}) | "
              f"{n_eval_seq} seq in {time.time() - t_ev:.1f}s")
        metrics.append(dict(step=int(step_now), val_loss_fault=va, val_loss_fault_se=va_se,
                            fault_p=float(spec.p), fault_k=int(spec.k),
                            n_eval_seq=int(n_eval_seq), eval_chunk=int(eval_mb)))
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        if va < best:
            best = va
            if save_best:                    # the WRITE is optional; `best` itself always advances
                model.serialize(best_model_path)
        return best

    t_run = time.time()
    step_num = start_step
    try:
        while step_num < end_step:
            if max_seconds is not None and time.time() - t_run > max_seconds:
                print(f"  max_seconds reached -> stopping at step {step_num} for resume")
                break
            if (step_num % eval_interval == 0) and (step_num > 0):                    # exact eval boundary
                best_val = do_eval(model, step_num, best_val)
            # Train one block, clamped so it never crosses an eval boundary or the end. This
            # clamp is what keeps evals on the grid without `chunk` having to divide
            # eval_interval; each distinct `n` is its own jit specialization, so a span uses
            # two (chunk and the constant remainder), both compiled once.
            n = min(chunk, eval_interval - (step_num % eval_interval), end_step - step_num)
            t_io = time.time()
            host_batch = train_loader.block(n, batch_size, step_num, stream=0)   # host read (cold megablock on the 1st)
            dt = time.time() - t_io
            io["train_stage_s"] += dt
            if io["first_block_read_s"] is None:
                io["first_block_read_s"] = dt
            xs, ys = jax.device_put(host_batch, shard_batch)
            model, opt_state, losses = train_block(model, opt_state, xs, ys, base_train, step_num, spec)
            losses.block_until_ready()
            step_num += n
            # The block's mean training loss, over n * batch_size sequences under the run's
            # own fault draw. Logged at the block (= checkpoint) cadence, and reused by the
            # divergence check below.
            block_loss = float(jnp.mean(losses))
            with open(train_log_path, "a") as f:
                f.write(json.dumps(dict(step=int(step_num), train_loss=block_loss,
                                        n_steps=int(n))) + "\n")
            if (not math.isfinite(block_loss)) or (max_loss is not None and block_loss > max_loss):
                why = ("non-finite loss" if not math.isfinite(block_loss)
                       else f"loss {block_loss:.2f} > max_loss {max_loss:.2f}")
                print(f"  DIVERGED: {why} at step {step_num} -> stopping run")
                metrics.append(dict(step=int(step_num), val_loss_fault=float("nan"),
                                    fault_p=float(spec.p), fault_k=int(spec.k), diverged=True))
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=2)
                break
            ckpt.maybe_write(model, opt_state, key, step_num, best_val)
        else:
            best_val = do_eval(model, step_num, best_val)        # final eval at end_step
    finally:
        ckpt.finish(model, opt_state, key, step_num, best_val)        # flush + guarantee final state durable

    # I/O summary -> run_log, one line per segment. A large first_block_read_s or
    # train_stage_frac means workers are FS-bound.
    wall = max(1e-9, time.time() - t_run)
    fb = io["first_block_read_s"]
    summary = dict(event="io_summary", segment_wall_s=round(wall, 2),
                   steps_trained=int(step_num - start_step),
                   first_block_read_s=(round(fb, 3) if fb is not None else None),
                   train_stage_s=round(io["train_stage_s"], 3),
                   train_stage_frac=round(io["train_stage_s"] / wall, 4),
                   val_read_s=round(io["val_read_s"], 3), n_evals=io["n_evals"])
    # Per loader, not summed: train slides across megablocks while val reads one block at
    # the anchor step and never moves, so a sum would hide both.
    for tag, ldr in (("train", train_loader), ("val", val_loader)):
        st = getattr(ldr, "stats", None)
        if st:
            summary[f"{tag}_loader"] = {k: (round(v, 3) if isinstance(v, float) else v)
                                        for k, v in st.items()}
    print(f"  I/O    : first-block read {'n/a' if fb is None else f'{fb:.1f}s'} | train staging "
          f"{io['train_stage_s']:.1f}s ({100 * io['train_stage_s'] / wall:.1f}% of {wall:.0f}s wall) | "
          f"probe reads {io['val_read_s']:.1f}s (once) / {io['n_evals']} eval(s)")
    ts = getattr(train_loader, "stats", {})
    if ts:
        # A prefetch miss is the training loop sitting idle on a megablock read.
        print(f"  loader : {ts['n_refreshes']} refresh(es) "
              f"({ts['prefetch_hits']} prefetched / {ts['prefetch_misses']} cold) | "
              f"blocked {ts['refresh_wait_s']:.1f}s | slowest block read "
              f"{ts['max_block_load_s']:.1f}s | {ts['n_perm_builds']} slot perm(s) in "
              f"{ts['perm_build_s']:.2f}s | {ts['n_windows']:,} windows | epoch {ts['max_epoch']}")
    with open(os.path.join(results_dir, RUN_LOG), "a") as f:
        f.write(json.dumps(summary) + "\n")

    return model, opt_state, key, best_val, step_num


# train.bin at/under SIMPLE_MAX_GB is held whole in host RAM; above it we slide a BUFFER_GB
# buffer. BUFFER_GB is a global constant rather than per-node-RAM auto because the loader's
# schedule is a function of it, so every node a run touches must agree or a resume samples
# different blocks. Overriding it via env changes the data stream.
BUFFER_GB = float(os.environ.get("NANO_LLAMA_BUFFER_GB", 2.5))
VAL_BUFFER_GB = float(os.environ.get("NANO_LLAMA_VAL_BUFFER_GB", 0.25))
SIMPLE_MAX_GB = 64.0


def choose_loader_buffer_for_file(path: str) -> Optional[float]:
    """The SlidingLoader buffer for one token file: None (whole file resident) if it fits."""
    gb = os.path.getsize(path) / 1e9
    return None if gb <= SIMPLE_MAX_GB else BUFFER_GB


def choose_loader_buffer(data_dir: str) -> Optional[float]:
    """choose_loader_buffer_for_file on data_dir/train.bin."""
    return choose_loader_buffer_for_file(os.path.join(data_dir, "train.bin"))


# =============================================================================
# Public inner training loop (selects the data loader by `buffer`)
# =============================================================================
def run_slides(start_step: int, n_steps: int, refresh_steps: int) -> bool:
    """True iff [start_step, start_step+n_steps) crosses a megablock refresh boundary."""
    if n_steps <= 0:
        return False
    end_step = start_step + n_steps
    return (end_step - 1) // refresh_steps > start_step // refresh_steps


def run_training(model: Llama, opt_state: optax.OptState, optim: optax.GradientTransformation,
                 spec: FaultSpec, data_dir: str, key: jax.Array,
                 seed: int, start_step: int, n_steps: int, best_val: float,
                 batch_size: int, micro_batch: int, eval_interval: int, eval_seqs: int,
                 results_dir: str, checkpoint_dir: str, mesh: jax.sharding.Mesh,
                 max_seconds: float = None, buffer: float = None, prefetch: bool = True,
                 checkpoint_seconds: float = CHECKPOINT_SECONDS, max_loss: Optional[float] = None,
                 save_best: bool = True, tc: Optional[TrainConfig] = None
                 ) -> Tuple[Llama, optax.OptState, jax.Array, float, int]:
    """Run `n_steps` of training over data_dir and return the advanced state."""
    bs = model.config.block_size
    # The eager next-megablock prefetch only pays if the run actually slides to that block;
    # on a short run one megablock covers the whole span and it is a wasted buffer-sized
    # read. refresh_steps is a function of the buffer alone, so this is decidable up front,
    # and prefetch never affects which data is sampled.
    train_prefetch = (prefetch and buffer is not None
                      and run_slides(start_step, n_steps, SlidingLoader.refresh_steps(buffer, batch_size, bs)))
    train_loader = SlidingLoader(os.path.join(data_dir, "train.bin"), bs, seed=seed,
                                 batch_size=batch_size, buffer_gb=buffer, prefetch=train_prefetch)
    val_loader = SlidingLoader(os.path.join(data_dir, "val.bin"), bs, seed=seed,
                               batch_size=batch_size, buffer_gb=VAL_BUFFER_GB, prefetch=False)
    # Background-load the val probe now so the first eval doesn't stall on NFS. The probe
    # is fixed, so the block to warm is the anchor's, read once for the whole segment.
    val_loader.warm(EVAL_ANCHOR_STEP)

    # Size the checkpoint block to wall-clock here, then hand the loop the raw step count.
    checkpoint_iter, step_time = probe_checkpoint_iter(model, opt_state, optim, spec, key, batch_size,
                                                       micro_batch, eval_interval, mesh, checkpoint_seconds)
    run_meta = {"seed": int(seed), "data_dir": data_dir, "step_seconds": step_time,
                "loader": "SlidingLoader", "buffer_gb": buffer,      # buffer_gb None -> whole-in-RAM
                "val_buffer_gb": VAL_BUFFER_GB, "train_prefetch": train_prefetch}   # for I/O-cost analysis
    # Optional only so tests that drive run_training directly keep working; the worker
    # always passes it, so every real run records what its optimizer was built with.
    if tc is not None:
        run_meta["hparams"] = realized_hparams(tc, model)
    try:
        return run_training_loop(
            model, opt_state, optim, spec, key, start_step, n_steps, best_val,
            batch_size, micro_batch, eval_interval, eval_seqs, results_dir, checkpoint_dir, mesh,
            train_loader, val_loader, checkpoint_iter, max_seconds=max_seconds, run_meta=run_meta,
            max_loss=max_loss, save_best=save_best)
    finally:
        train_loader.close(); val_loader.close()
