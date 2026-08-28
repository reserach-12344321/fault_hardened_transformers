#!/usr/bin/env python
#!/usr/bin/env python
#!/usr/bin/env python
r"""Materialize fault-eval job arrays into EvalResult JSONs, one per completed job.

Waves of one campaign are pooled into ONE FLAT out_dir, whose file count is then the model
count. Job names are unique across waves by construction, and materialize() verifies that
before writing, since flat output turns a repeated name into silent overwriting.

Only finished jobs are emitted, so every record carries the complete (k, p) grid it was
configured with. KEEP THIS OUTPUT SEPARATE from process_training_runs.py's -- the two define
"the model" and "the loss" differently.
"""
import datetime
import glob
import json
import math
import os
import sys
from collections import namedtuple

# Nothing here builds a model, but EvalResult -> LlamaConfig -> jax. Pin CPU so this never
# waits on the GPU the eval array itself is using.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Importable whether invoked from the repo root or from inside this package.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    # repo root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cluster_orchestrator import worker_api

from nano_llama.eval_result import EvalResult
from nano_llama.llama import LlamaConfig
from nano_llama.train_core import TrainConfig

# EvalResult.from_sweep_points duck-types its input on these names. The worker serialises the
# same numbers under shorter on-disk keys, so adapt rather than reimplement the mapping.
_Point = namedtuple("_Point", "k p mean se n_seq reached_target")


def _adapt_points(record: dict) -> list:
    """The results file's points, shaped for EvalResult.from_sweep_points, in order."""
    return [_Point(k=pt["k"], p=pt["p"], mean=pt["loss"], se=pt["se"], n_seq=pt["n"],
                   reached_target=pt.get("reached_target", True))
            for pt in record["points"]]


def n_non_embedding(config: LlamaConfig, total_n_params: int) -> int:
    """Non-embedding parameter count, without opening the checkpoint."""
    embedding = config.vocab_size * config.n_embd
    return int(total_n_params - embedding * (1 if config.tie_embeddings else 2))


def diverged_reason(points, vocab_size: int):
    """Why this curve is unusable, or None if it looks like a real model.

    Two symptoms, both meaning the underlying TRAINING run blew up rather than the eval
    misbehaving: a non-finite loss, or every loss at or above ln(vocab_size), i.e. worse than
    predicting uniformly. The second uses the MINIMUM over the curve -- fault only degrades a
    model, so if the cleanest condition cannot beat uniform, nothing on the curve can.

    Diverged runs reach here because the training worker marks them DONE, so the DONE marker
    cannot exclude them, and one NaN is enough to turn a whole cohort's fit into NaN.
    """
    losses = [pt.mean for pt in points]
    n_bad = sum(1 for x in losses if not math.isfinite(x))
    if n_bad:
        return f"DIVERGED: non-finite loss at {n_bad} of {len(losses)} points"
    uniform = math.log(vocab_size)
    if min(losses) >= uniform:
        return (f"DIVERGED: best loss {min(losses):.3f} >= ln(vocab={vocab_size})={uniform:.3f} "
                f"-- worse than predicting uniformly")
    return None


def eval_result_from_job(job_dir: str) -> tuple:
    """Build one EvalResult from a finished eval job, or ``(None, reason)`` if it isn't usable."""
    results_dir = os.path.join(job_dir, "results")
    if not worker_api.is_done(results_dir):
        return None, "not finished (no results/DONE)"

    results_path = os.path.join(results_dir, "eval_results.json")
    try:
        with open(results_path) as fh:
            record = json.load(fh)
        model_config = LlamaConfig.load(os.path.join(job_dir, "final_model.json"))
        train_config = TrainConfig.load(os.path.join(job_dir, "train_config.json"))
        with open(os.path.join(job_dir, "eval_config.json")) as fh:
            n_configured = len(json.load(fh)["kp_pairs"])
    except (OSError, ValueError, KeyError) as e:
        return None, f"unreadable job inputs ({type(e).__name__}: {e})"

    points = _adapt_points(record)
    if len(points) != n_configured:
        return None, f"done but curve is short: {len(points)} of {n_configured} (k, p) points"
    bad = diverged_reason(points, model_config.vocab_size)
    if bad:
        return None, bad

    total_n_params = int(record["n_params"])
    return EvalResult.from_sweep_points(
        points,
        model_config=model_config,
        train_config=train_config,
        # The header is what the worker actually ran with; fault_config.json is the same numbers.
        k_train=record["k_train"], p_train=record["p_train"],
        total_n_params=total_n_params,
        # D = tokens actually seen, from the checkpoint's real final step (NOT the nominal max_iters
        # -- a run stopped early by its wall-clock budget or a divergence never reached that).
        n_train_tokens=int(record["n_train_tokens"]),
        n_non_embedding_params=n_non_embedding(model_config, total_n_params),
    ), None


def eval_array_dirs(paths) -> list:
    """Expand experiment root(s) into their waves; pass plain array dirs through unchanged."""
    from experiment_util.prepare_eval_array import wave_dirs      # local: keeps this module's import light
    out = []
    for path in ([paths] if isinstance(paths, str) else list(paths)):
        out.extend(wave_dirs(path) or [path])
    return out


def wave_spec(array_dir: str):
    """The eval spec one wave was staged under, or None for anything that is not a wave."""
    try:
        with open(os.path.join(array_dir, "wave_spec.json")) as fh:
            return json.load(fh)["spec"]
    except (OSError, ValueError, KeyError):
        return None


def _report_spec_drift(array_dirs) -> None:
    """Warn, loudly, when the arrays being pooled were staged under different eval specs."""
    specs = [(d, wave_spec(d)) for d in array_dirs]
    seen = {}
    for d, spec in specs:
        if spec is not None:
            seen.setdefault(json.dumps(spec, sort_keys=True), []).append(os.path.basename(d.rstrip("/")))
    if len(seen) < 2:
        return
    groups = [(json.loads(key), names) for key, names in seen.items()]
    keys = sorted({k for spec, _ in groups for k in spec})
    differing = [k for k in keys if len({json.dumps(spec.get(k), sort_keys=True)
                                         for spec, _ in groups}) > 1]
    print(f"  !! SPEC DRIFT: the {len(array_dirs)} array(s) being pooled were staged under "
          f"{len(groups)} DIFFERENT eval specs. Their models are written into ONE dir and load as one\n"
          f"     population, so cohorts will mix measurement conditions. Differing key(s): "
          f"{', '.join(differing) or '(none -- ordering only)'}")
    for spec, names in groups:
        detail = ", ".join(f"{k}={spec.get(k)!r}" if k != "extra_kp_pairs"
                           else f"extra_kp_pairs=<{len(spec.get(k) or [])} pairs>" for k in differing)
        print(f"       {len(names)} wave(s) [{', '.join(names)}]: {detail}")


def _some(items, cap: int = 10) -> str:
    """``items`` as an indented name list."""
    lines = "".join(f"\n       {name}" for name in items[:cap])
    return lines + (f"\n       ... and {len(items) - cap} more" if len(items) > cap else "")


def load_array(array_dir: str) -> tuple:
    """Every FINISHED job in one eval array -> EvalResult. Returns (results, skipped) as
    ([(job_name, EvalResult), ...], [(job_name, reason), ...]). Jobs are found by their
    eval_config.json, which prepare_eval_array writes into every one."""
    job_cfgs = sorted(glob.glob(os.path.join(array_dir, "*", "eval_config.json")))
    if not job_cfgs:
        # An experiment ROOT holds waves, not jobs, so it globs to nothing here. Left alone that reads
        # as "the array has no finished jobs yet" -- an empty output dir and a clean exit -- which is
        # the one failure mode of this script that produces no error at all. Name it.
        from experiment_util.prepare_eval_array import wave_dirs
        waves = wave_dirs(array_dir)
        if waves:
            raise SystemExit(
                f"ABORT: {array_dir} is an experiment ROOT ({len(waves)} wave(s)), not a job array -- "
                f"its jobs live one level deeper, so materializing it would write NOTHING and report "
                f"success.\n  Pass eval_array_dirs(root) (or wave_dirs(root)) instead.")
    results, skipped = [], []
    for cfg_path in job_cfgs:
        job_dir = os.path.dirname(cfg_path)
        result, reason = eval_result_from_job(job_dir)
        (skipped if result is None else results).append(
            (os.path.basename(job_dir), reason if result is None else result))
    return results, skipped


def materialize(array_dirs, out_dir: str, loader=load_array) -> tuple:
    """Write an EvalResult JSON for every finished job across `array_dirs` into one flat
    `out_dir`. Returns (n_written, n_skipped) and prints a per-array summary.

    `loader` is the per-array reader, dir -> (results, skipped). It defaults to load_array;
    process_training_runs passes its own so training sweeps land in the identical layout.
    Parameterised rather than copied, because everything below the read is what makes the two
    outputs interchangeable to load_eval_results, and a second copy would drift.

    FLAT, not one subdir per array: a campaign is normally several waves of one experiment
    describing one pooled population, and a flat dir makes it a single loadable unit whose
    file count IS the model count.

    Which makes job-name collisions dangerous rather than untidy: flat, a repeated name is one
    file silently overwriting another, i.e. a model disappearing from the fit with no error.
    Waves are disjoint by construction, so a collision means that invariant was broken. It is
    checked across all arrays BEFORE anything is written, so a bad call leaves no partial output.

    The skip reasons are printed too: a silent count would hide a systematic problem behind
    what looks like ordinary in-flight progress.
    """
    array_dirs = list(array_dirs)          # consumed twice (per-array read, then the spec check)
    per_array = [(os.path.basename(d.rstrip("/")), *loader(d)) for d in array_dirs]

    owner, dupes = {}, []
    for array_name, results, _ in per_array:
        for name, _ in results:
            if name in owner:
                dupes.append((name, owner[name], array_name))
            owner[name] = array_name
    if dupes:
        raise SystemExit(
            f"ABORT: {len(dupes)} job name(s) appear in more than one array. Written flat they would "
            f"overwrite each other, dropping a model from the cohorts with no error:\n"
            + "\n".join(f"    {n}   ({a} and {b})" for n, a, b in dupes)
            + "\n  Waves are disjoint by construction, so this means a job was staged twice -- delete "
              "the stale copy, or materialize the arrays separately.")

    _report_spec_drift(array_dirs)

    # Only create the output dir once we know something goes in it: this is normally run against a
    # draining array, and a run that finds nothing ready should not leave an empty timestamped dir
    # behind to be mistaken later for a materialization that produced no models.
    n_total = sum(len(results) for _, results, _ in per_array)
    if n_total:
        os.makedirs(out_dir, exist_ok=True)
    written = skipped_total = under_converged = 0
    for array_name, results, skipped in per_array:
        for name, result in results:
            result.save(os.path.join(out_dir, name + ".json"))
            under_converged += sum(1 for ok in result.eval_reached_se_target if not ok)
        written += len(results)
        skipped_total += len(skipped)

        by_reason = {}
        for _, reason in skipped:
            key = reason.split(":")[0]
            by_reason[key] = by_reason.get(key, 0) + 1
        detail = ("  (skipped: " + ", ".join(f"{n} {r}" for r, n in sorted(by_reason.items())) + ")"
                  if by_reason else "")
        print(f"{array_name}: wrote {len(results)} EvalResult(s){detail}")
        # Anything other than a plain not-yet-finished job is worth naming -- especially a DIVERGED
        # one, which is a training-recipe failure surfacing here rather than an eval problem. Names are
        # CAPPED (_some): a legacy campaign is every job in the array, and 2,800 identical lines hide
        # the one-line reason underneath them.
        div = [n for n, r in skipped if r.startswith("DIVERGED")]
        other = [(n, r) for n, r in skipped
                 if not r.startswith(("DIVERGED", "not finished"))]
        if div:
            print(f"  !! {len(div)} DIVERGED run(s) EXCLUDED -- their training blew up; one NaN loss "
                  f"would make a whole cohort's fit NaN:{_some(div)}")
        for name, reason in other[:10]:
            print(f"    !! {name}: {reason}")
        if len(other) > 10:
            print(f"    ... and {len(other) - 10} more")

    if under_converged:
        # Kept, not dropped -- the record carries eval_reached_se_target=False per point, and
        # fit_matched_scaling_law counts them -- but silence here would let a whole under-converged
        # array look like a clean one.
        print(f"\n  !! {under_converged} eval point(s) in the written records hit max_evals WITHOUT "
              f"reaching target_se. They are kept (flagged eval_reached_se_target=False); a fit that "
              f"weights by eval_se is trusting an SE that never converged.")
    if not n_total:
        print(f"\nnothing to write -- no finished job produced a usable record, so {out_dir} was not "
              f"created.")
    return written, skipped_total


if __name__ == "__main__":
    # Name the experiment ROOT: eval_array_dirs expands it into its waves and materialize()
    # pools them into ONE flat out_dir. A single wave or plain array dir works too. Re-run as
    # later waves drain -- it is idempotent, and a fresh timestamped out_dir keeps each
    # materialization a self-contained snapshot of the campaign so far.
    root = "/media/trevor/data_flash/job_arrays/eval_pgrid_512"                       # <-- SET
    out_dir = "/mnt/storage/eval_summaries/eval_pgrid_512_{}".format(     # <-- SET
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

    written, skipped = materialize(eval_array_dirs(root), out_dir)
    print(f"\ntotal: wrote {written} EvalResult(s), skipped {skipped} job(s) into {out_dir}")
    print(f"load with: data_loading.load_eval_results('{out_dir}')")
