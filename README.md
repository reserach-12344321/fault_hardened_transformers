# fault_hardened_transformers

Train small Llama models with hardware faults injected into the matmuls, score them under
faults afterwards, and fit scaling laws to what comes out. 

This code was designed to train models on supercomputers: all compute jobs were driven through the  `cluster_orchestrator` package. 


## analysis/

[fit_matched_scaling_law.py](analysis/fit_matched_scaling_law.py) is the script that produces
the fits. It loads the processed JSONs, screens them, builds each cohort's $(N, D, L)$ arrays,
fits, bootstraps and dumps the result. The notebooks read what it wrote.

* [scaling_law_plots.ipynb](analysis/scaling_law_plots.ipynb) reads a saved fit and makes the
  main figures: the residuals, the pairwise bootstrap correlations between fit parameters, the
  capacity retained (main text Fig. 1) and the exponents with error bars (main text Fig. 3).
* [scaling_law_effective_capacity.ipynb](analysis/scaling_law_effective_capacity.ipynb)
  subtracts the fitted $D$ dependence off the raw losses and projects what is left back onto
  the $p = 0$ law, which recovers the same capacity retained without committing to a
  functional form for the $N$ dependence.
* [scaling_law_full_fit_residuals.ipynb](analysis/scaling_law_full_fit_residuals.ipynb) is
  where you can play with point fits and their residuals and see for yourself that $\alpha_2$
  is needed to represent the structure in our data.
* [scaling_law_fixed_E_profiles.ipynb](analysis/scaling_law_fixed_E_profiles.ipynb) refits
  each cohort with $E$ pinned across a range of values, plotted as how much worse each cohort
  does than its own free-$E$ best.
* [reversal_d512.ipynb](analysis/reversal_d512.ipynb) shows that models trained without faults
  can get worse as a function of $D$ when you evaluate them with faults (main text Fig. 4a).
* [logit_temperature_d512.ipynb](analysis/logit_temperature_d512.ipynb) fits the single global
  temperature that best explains the fault-marginalised predictive in forward KL, and splits
  the total distortion into the part a temperature accounts for and the part it does not.

## nano_llama/

The model and the training loop. Nothing in here knows about clusters or sweeps.

* `llama.py` builds the model, and holds `FaultSpec` and the faulted matmul itself.
* `fault.py` is the fault config, `initializations.py` the llama2.c/GPT-2 init.
* `train_core.py` is the training loop and `TrainConfig`: LR schedule, micro-batch heuristic,
  the data-parallel train and eval blocks, checkpoint I/O. `train.py` holds the optimizer
  groups and the loss.
* `token_data.py` is the data loader. Windows are drawn without replacement and the sampling
  is step-keyed, so a restart or a change of chunk size sees the same tokens.
* `fault_eval.py` scores an existing checkpoint at a list of $(k, p)$, streaming batches until
  the standard error hits a target absolute precision in nats. It also holds the logit-sampling
  side, which keeps the full next-token probability vector instead of reducing to a loss.
* `eval_result.py` is the on-disk `(trained model, eval condition) -> loss` record everything
  downstream reads. `metrics.py` reads a run's `metrics.json` and is stdlib only, so the
  monitors can import it without paying for jax.

## experiment_util/

Building job arrays and reducing them back down. A job array is a directory of self-contained
job subdirs, each holding the three configs for one run.

* `standard_models.py` is the ladder, rungs 1-24, locked architecture with only `n_embd` and
  `n_layer` varying.
* `gen_sweep.py` turns (model, train, fault) triples into an array, `gen_full_sweep.py`
  generates the full sweep from families, and `gen_d512_tpp_range_sweep.py` is the complement
  that holds the model at one rung and sweeps the horizon over two decades of TPP at many
  seeds.
* `inject_resources.py` stamps the cpu/mem/gres request on, `assign_gpus_by_share.py` decides
  how many GPUs each job gets so nothing straggles at the tail.
* `prepare_eval_array.py` and `prepare_logit_sample_array.py` stage the follow-on arrays in
  waves as training runs finish.
* `process_training_runs.py`, `process_eval_arrays.py` and `process_logit_marginals.py`
  materialize finished arrays into the flat directories of JSON the analysis reads.
  `logit_marginals.py` loads the marginal predictives and fits temperatures to them.

## orchestrator_hooks/

The cluster side. `worker.py`, `eval_worker.py` and `logit_sample_worker.py` are the three
workers, all on the same four-argument contract the orchestrator sbatches. `cannon_config.py`,
`engaging_config.py` and `local_config.py` describe the clusters, and `scripts/` holds the
end-to-end launchers plus the job files.

## scaling_law/

Fitting the surface. `surface.py` is the model and the Huber objective, `solver.py` the
multi-start bounded L-BFGS-B, `starts.py` the centroid-anchored initialisations, and
`resample.py` the pairs bootstrap. `data_loading.py` reads the processed EvalResult JSONs and
screens them, and `fit_store.py` writes fits to disk as an npz plus a JSON manifest, so the
plots do not depend on this repo's class layout to read a fit back.


## tests/

`python -m unittest` over `tests/`. They cover the faulted forward pass and its RNG, the
training core, data parallelism, the loaders, serialisation, and the array staging logic.
