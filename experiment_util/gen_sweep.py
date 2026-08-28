"""Build a training job array from (model, train, fault) config triples.

A job array is a directory of self-contained job subdirs, each holding the three configs
that specify one run. It carries no seeds and no resources -- those are stamped on
separately, so the same array can be re-seeded or re-resourced without regenerating it.

    specs = sweep_specs(MODELS, [20, 80, 320, 1280],
                        [FaultConfig(p=0.0, k=4), FaultConfig(p=0.01, k=4)])
    write_job_array(specs, "/home/trevor/data/job_arrays/fineweb_std")

Run under scienv: n_params instantiates the model to count parameters.
"""
from __future__ import annotations

import os
import io
import math
import contextlib
from collections import Counter
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import jax

from nano_llama.llama import Llama, LlamaConfig
from nano_llama.fault import FaultConfig
from nano_llama.train_core import TrainConfig

Spec = Tuple[LlamaConfig, TrainConfig, FaultConfig]

# Locked training protocol. Every HP except the peak LR is held fixed across the whole
# ladder; the peak LR is the single shape-dependent one, and it is baked into each run's
# TrainConfig here rather than rescaled at train time -- nothing downstream knows about width.
GLOBAL_BATCH = 128            # effective batch, in sequences
BASE_LR      = 1e-3           # peak LR at BASE_WIDTH; sqrt-width scaled from there
BASE_WIDTH   = 64
MIN_LR_FRAC  = 0.1            # cosine decays to peak/10
# Warmup as a FRACTION of the horizon, so the schedule shape is identical at every duration
# and matches the array this sweep is compared against.
WARMUP_FRAC  = 0.1
WEIGHT_DECAY = 0.1
BETA1, BETA2 = 0.9, 0.95
ADAM_EPS     = 1e-8
GRAD_CLIP    = 1.0
INIT_STD     = 0.02           # init std; wo/w3 additionally carry 1/sqrt(2*n_layer) (initializations.py)

# Periodic monitoring evals, bounded both by a target count and by a share of training
# compute -- an unbounded count makes the eval bill dominant on the long runs.
TARGET_EVALS  = 50            # at most this many evenly-spaced evals per run (None -> end-only)
MAX_EVAL_FRAC = 0.02          # and never more than this fraction of training compute
# Sequences per eval, uniform across the ladder on purpose: eval_seqs sets the eval's
# PRECISION, and points carrying different error bars put noise where the design did not ask
# for it. Cost is not held uniform -- eval_interval absorbs it per run via MAX_EVAL_FRAC.
# Keep it small: eval_interval_for's compute floor scales with it, so a large value silently
# trades away the TARGET_EVALS you asked for.
EVAL_SEQS     = 1024


def _peak_lr(n_embd: int) -> float:
    """The ladder's only shape-dependent HP: eta(d) = BASE_LR * sqrt(BASE_WIDTH/d)."""
    return BASE_LR * (BASE_WIDTH / n_embd) ** 0.5


def eval_overhead_frac(eval_seqs: int, batch_size: int, eval_interval: int) -> float:
    """Eval compute as a fraction of training compute."""
    return float(eval_seqs) / (3.0 * max(1, batch_size) * max(1, eval_interval))


def eval_interval_for(max_iters: int, eval_seqs: int, batch_size: int, target_evals: Optional[int],
                      max_eval_frac: float = 0.02) -> int:
    """Periodic eval_interval honouring two caps at once:"""
    if not target_evals or target_evals < 1:
        return max_iters
    target_interval = max(1, int(round(max_iters / target_evals)))
    floor = (math.ceil(float(eval_seqs) / (3.0 * max(1, batch_size) * max_eval_frac))
             if max_eval_frac > 0 else 1)
    return min(max_iters, max(target_interval, floor))


def n_params(model: LlamaConfig) -> int:
    """Total parameter count, by instantiating on a throwaway key -- counts are seed-independent."""
    with contextlib.redirect_stdout(io.StringIO()):
        return int(Llama(model, jax.random.PRNGKey(0)).count_params())


def train_config_for(model: LlamaConfig, tok_per_param: float, *,
                     batch_size: int = GLOBAL_BATCH, n_total: int = None,
                     target_evals: Optional[int] = TARGET_EVALS,
                     max_eval_frac: float = MAX_EVAL_FRAC,
                     eval_seqs: int = EVAL_SEQS) -> TrainConfig:
    """The matched-cosine TrainConfig for training `model` at `tok_per_param` tokens/param."""
    N = n_total if n_total is not None else n_params(model)
    tokens_per_iter = batch_size * model.block_size
    max_iters = max(1, round(tok_per_param * N / tokens_per_iter))

    lr = _peak_lr(model.n_embd)
    eval_interval = eval_interval_for(max_iters, eval_seqs, batch_size, target_evals, max_eval_frac)
    warmup = max(1, round(WARMUP_FRAC * max_iters))
    return TrainConfig(
        batch_size=batch_size, learning_rate=lr, max_iters=max_iters,
        warmup_iters=warmup, lr_decay_iters=max_iters, min_lr=MIN_LR_FRAC * lr,
        eval_interval=eval_interval, eval_seqs=eval_seqs,
        weight_decay=WEIGHT_DECAY, beta1=BETA1, beta2=BETA2, adam_eps=ADAM_EPS,
        init_std=INIT_STD, grad_clip=GRAD_CLIP, divergence_loss_factor=1.5,
    )


def spec_dirname(mc: LlamaConfig, tc: TrainConfig, fc: FaultConfig) -> str:
    """Content-derived subdir name for one spec, e.g. 'd320_L10_it13000_k4_p0.01'."""
    return f"d{mc.n_embd}_L{mc.n_layer}_it{tc.max_iters}_k{fc.k}_p{fc.p}"


def write_job_array(specs: Iterable[Spec], data_dir: str, *, verbose: bool = True,
                    name_fn: Callable[[LlamaConfig, TrainConfig, FaultConfig], str] = spec_dirname
                    ) -> List[str]:
    """Serialize each triple to a fresh subdir of data_dir holding the three configs."""
    specs = list(specs)
    base_names = [name_fn(mc, tc, fc) for mc, tc, fc in specs]
    counts = Counter(base_names)
    widths = {name: len(str(n - 1)) for name, n in counts.items()}   # pad so _rK sorts by K
    seen: dict = {}

    os.makedirs(data_dir, exist_ok=True)
    written: List[str] = []
    for (mc, tc, fc), base in zip(specs, base_names):
        if counts[base] > 1:
            k = seen.get(base, 0)
            seen[base] = k + 1
            name = f"{base}_r{k:0{widths[base]}d}"
        else:
            name = base
        job_dir = os.path.join(data_dir, name)
        os.makedirs(job_dir, exist_ok=False)
        mc.save(os.path.join(job_dir, "model_config.json"))
        tc.save(os.path.join(job_dir, "train_config.json"))
        fc.save(os.path.join(job_dir, "fault_config.json"))
        written.append(job_dir)
    if verbose:
        n_rep = sum(n for n in counts.values() if n > 1)
        extra = f" ({len(counts)} unique specs; {n_rep} replicated)" if n_rep else ""
        print(f"wrote {len(written)} job dirs to {data_dir}{extra}")
    return written
