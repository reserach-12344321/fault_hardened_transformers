"""nano_llama hooks for running fault-sweep training through cluster_orchestrator.

A job's inputs are the run spec (model_config / train_config / fault_config) plus meta.json
(the RNG seed) and resources.json (the cpu/mem/gres request), both injected by scripts/.
worker.py is the cluster-side training worker the orchestrator sbatches.
"""

# The prefixes within results/ holding worker.py's RESUME STATE, declared as
# ClusterRun(resume_paths=RESUME_PATHS): which changes make this job's staging on OTHER
# clusters stale. Declaring it matters because of the SMALL files -- the worker appends to
# four logs every segment, so the generic "any content change" default is true on essentially
# every pull and reaps copies whose checkpoint is still byte-identical.
RESUME_PATHS = ("checkpoint/",)
