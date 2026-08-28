#!/usr/bin/env python
#!/usr/bin/env python
#!/usr/bin/env python
r"""Materialize training sweeps into EvalResult JSONs, in process_eval_arrays' layout.

Both emit one flat dir that load_eval_results reads identically, so a partially complete
training array can be fitted without waiting for an eval array. The collision check, flat
write and skip accounting are IMPORTED from process_eval_arrays, not copied.

Records the FINAL eval rather than best-val (a minimum over ~50 noisy evals is biased low),
D from the real final step, and n_params from run_log.jsonl so no checkpoint is opened.
Diverged and short runs are excluded -- the worker marks a diverged run DONE.

FIT THESE ON THEIR OWN: the loss is the in-loop monitoring probe on val, not a converged
estimate on a held-out split. Nor pool across the 2026-08-03 corpus chunk-shuffle.
"""
import datetime
import glob
import json
import os
import sys

# Nothing here builds a model, but EvalResult -> LlamaConfig -> jax. Pin CPU so this never
# contends with live training on the shared GPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Importable whether invoked from the repo root or from inside this package.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    # repo root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cluster_orchestrator import worker_api

from nano_llama.eval_result import EvalResult
from nano_llama.llama import LlamaConfig
from nano_llama.train_core import TrainConfig

# Shared with the eval-array path on purpose: `materialize` carries the collision check and
# flat write, the rest are the record-shaping rules that keep the two outputs the same shape.
from experiment_util.process_eval_arrays import (
    _Point, diverged_reason, materialize, n_non_embedding)


def _final_step(job_dir: str, results_dir: str) -> int | None:
    """The run's actual last optimizer step: checkpoint meta first, metrics as a fallback."""
    ck = os.path.join(results_dir, "checkpoint", "meta.json")
    if os.path.isfile(ck):
        try:
            return int(json.load(open(ck))["step"])
        except (OSError, ValueError, KeyError):
            pass
    try:
        rows = json.load(open(os.path.join(results_dir, "metrics.json")))
        return max(int(r["step"]) for r in rows) if rows else None
    except (OSError, ValueError, KeyError):
        return None


def _n_params(results_dir: str) -> int | None:
    """Total parameter count from run_log.jsonl, so no checkpoint is deserialised."""
    path = os.path.join(results_dir, "run_log.jsonl")
    if not os.path.isfile(path):
        return None
    n = None
    try:
        with open(path) as fh:
            for line in fh:
                rec = json.loads(line)
                if isinstance(rec.get("model"), dict) and "n_params" in rec["model"]:
                    n = int(rec["model"]["n_params"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return n


def eval_result_from_run(job_dir: str) -> tuple:
    """One EvalResult from a finished training run, or (None, reason)."""
    results_dir = os.path.join(job_dir, "results")
    if not worker_api.is_done(results_dir):
        return None, "not finished (no results/DONE)"

    try:
        model_config = LlamaConfig.load(os.path.join(job_dir, "model_config.json"))
        train_config = TrainConfig.load(os.path.join(job_dir, "train_config.json"))
        with open(os.path.join(job_dir, "fault_config.json")) as fh:
            fault = json.load(fh)
        with open(os.path.join(results_dir, "metrics.json")) as fh:
            metrics = json.load(fh)
    except (OSError, ValueError, KeyError) as e:
        return None, f"unreadable job inputs ({type(e).__name__}: {e})"
    if not metrics:
        return None, "done but metrics.json has no eval records"

    total_n_params = _n_params(results_dir)
    if total_n_params is None:
        return None, "no n_params in results/run_log.jsonl (pre-provenance run?)"
    step = _final_step(job_dir, results_dir)
    if step is None:
        return None, "cannot determine final step (no checkpoint meta or metrics)"

    # worker.py writes DONE on divergence too, so DONE alone does not mean "trained to
    # completion": a DONE run short of its horizon is a divergence or an abort.
    target = int(train_config.max_iters)
    if step < target:
        return None, (f"done but SHORT: final step {step:,} < max_iters {target:,} "
                      f"({step / target:.1%} of the horizon)")

    # The FINAL eval, not the best. Selected by max step rather than by position, since a
    # resume can append a record for a step already present.
    final = max(metrics, key=lambda r: int(r.get("step", -1)))
    loss = final.get("val_loss_fault", final.get("val_loss_clean", final.get("val_loss")))
    if loss is None:
        return None, f"final metric record has no val loss (keys: {sorted(final)})"

    # Length-1 curve: a training run is scored at exactly one condition, its own train fault.
    # The probe's SE is recorded where present, rather than EvalResult's NaN placeholder.
    points = [_Point(k=fault["k"], p=fault["p"], mean=float(loss),
                     se=float(final.get("val_loss_fault_se", float("nan"))),
                     n_seq=int(final.get("n_eval_seq", 0)), reached_target=True)]

    bad = diverged_reason(points, model_config.vocab_size)
    if bad:
        return None, bad

    return EvalResult.from_sweep_points(
        points,
        model_config=model_config, train_config=train_config,
        k_train=fault["k"], p_train=fault["p"],
        total_n_params=total_n_params,
        # D = tokens ACTUALLY seen, from the real final step (see _final_step).
        n_train_tokens=int(step * train_config.batch_size * model_config.block_size),
        n_non_embedding_params=n_non_embedding(model_config, total_n_params),
    ), None


def load_array(array_dir: str) -> tuple:
    """Every finished run in one training sweep dir -> EvalResult.

    Same signature as process_eval_arrays.load_array, so it can go straight to materialize().
    """
    results, skipped = [], []
    for cfg_path in sorted(glob.glob(os.path.join(array_dir, "*", "train_config.json"))):
        job_dir = os.path.dirname(cfg_path)
        result, reason = eval_result_from_run(job_dir)
        (skipped if result is None else results).append(
            (os.path.basename(job_dir), reason if result is None else result))
    return results, skipped


if __name__ == "__main__":
    # One or more training array dirs, pooled into one flat out_dir exactly as eval waves are.
    # materialize() aborts on a name collision rather than silently overwriting.
    sweep_dirs = ["/mnt/storage/job_arrays/full_sweep_2026-08-04-14-47-05",
                  "/mnt/storage/job_arrays/full_sweep_2026-08-04-19-44-30",
                  "/media/trevor/data_flash/job_arrays/full_sweep_2026-08-06-11-01-34"]
    out_dir = "/mnt/storage/eval_summaries/training_{}".format(
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

    written, skipped = materialize(sweep_dirs, out_dir, loader=load_array)
    print(f"\ntotal: wrote {written} EvalResult(s), skipped {skipped} run(s) into {out_dir}")
    print(f"load with: data_loading.load_eval_results('{out_dir}')")
