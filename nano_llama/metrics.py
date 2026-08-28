"""Reading a run's results/metrics.json.

Stdlib only: the cluster monitors import this on a laptop and must not pay a jax
import to print a table.
"""
import json
import math
import os
from typing import List, Optional

VAL_LOSS_KEY = "val_loss_fault"


def val_loss_of(record: dict) -> Optional[float]:
    """The val loss carried by one metrics record, or None."""
    v = record.get(VAL_LOSS_KEY)
    return float(v) if v is not None else None


def load_metrics(results_dir: str) -> List[dict]:
    """results/metrics.json as a list, or [] if missing/unreadable/empty."""
    try:
        with open(os.path.join(results_dir, "metrics.json")) as f:
            m = json.load(f)
        return m if isinstance(m, list) else []
    except (OSError, ValueError):
        return []


def final_record(metrics: List[dict]) -> Optional[dict]:
    """The record at the largest step, or None."""
    recs = [r for r in metrics if "step" in r]
    return max(recs, key=lambda r: r["step"]) if recs else None


def final_val_loss(metrics: List[dict]) -> Optional[float]:
    """The final eval's val loss, or None if there is no eval / no loss field."""
    rec = final_record(metrics)
    return val_loss_of(rec) if rec is not None else None


def is_diverged(metrics: List[dict]) -> bool:
    """True iff the final record's val loss is present and non-finite."""
    v = final_val_loss(metrics)
    return v is not None and not math.isfinite(v)
