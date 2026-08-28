"""EvalResult: the on-disk record of one (trained model, eval condition) -> loss.

Fault coordinates are split into train-time and eval-time: a run trains under one
(k_train, p_train) but is scored at many eval conditions in a single pass, so
k_eval / p_eval / eval_loss and the Monte-Carlo provenance are PARALLEL tuples of
equal length -- index i of each describes one measurement.

Older JSONs wrote these as scalars; __post_init__ coerces those to 1-element
tuples, so pre-existing processed dirs still load.
"""
from dataclasses import dataclass, fields

from nano_llama.config_base import ConfigMixin
from nano_llama.llama import LlamaConfig
from nano_llama.train_core import TrainConfig


def _as_tuple(v, cast) -> tuple:
    """A parallel field as a tuple of `cast` -- JSON hands these back as lists."""
    return tuple(cast(x) for x in v)


@dataclass(frozen=True)
class EvalPoint:
    """One (k_eval, p_eval) measurement flattened together with its model's identity."""
    k_eval: float
    p_eval: float
    eval_loss: float
    eval_se: float
    n_eval_seq: int
    reached_se_target: bool
    result: "EvalResult"

    @property
    def model_config(self) -> LlamaConfig:
        return self.result.model_config

    @property
    def train_config(self) -> TrainConfig:
        return self.result.train_config

    @property
    def k_train(self) -> float:
        return self.result.k_train

    @property
    def p_train(self) -> float:
        return self.result.p_train

    @property
    def total_n_params(self) -> int:
        return self.result.total_n_params

    @property
    def n_non_embedding_params(self) -> int:
        return self.result.n_non_embedding_params

    @property
    def n_train_tokens(self) -> int:
        return self.result.n_train_tokens

    @property
    def size_key(self) -> str:
        return self.result.size_key


@dataclass(frozen=True)
class EvalResult(ConfigMixin):
    model_config: LlamaConfig     # architecture of the trained model
    train_config: TrainConfig     # training recipe (lr, batch, max_iters, ...)
    k_train: float                # fault block size k at TRAIN time (one per run -> scalar)
    p_train: float                # fault probability p at TRAIN time (one per run -> scalar)
    # ---- PARALLEL eval-condition tuples: index i of each describes one measurement ----------------
    k_eval: tuple                 # fault block size k at EVAL time, per condition
    p_eval: tuple                 # fault probability p at EVAL time, per condition
    eval_loss: tuple              # FAULTED val loss (nats) measured at each (k_eval, p_eval)
    total_n_params: int           # total trainable params -- the scaling-law N
    n_non_embedding_params: int   # total minus the token embedding (and the untied readout)
    n_train_tokens: int           # tokens seen during training -- the scaling-law D
    # Monte-Carlo provenance, also parallel: eval_se is the standard error of the mean CE
    # over n_eval_seq faulted sequences, and eval_reached_se_target is False where the
    # sweep hit max_evals first.
    eval_se: tuple
    n_eval_seq: tuple
    eval_reached_se_target: tuple

    def __post_init__(self):
        """Normalise every parallel field to a tuple, then assert they really are parallel."""
        set_ = object.__setattr__
        set_(self, "k_eval", _as_tuple(self.k_eval, float))
        set_(self, "p_eval", _as_tuple(self.p_eval, float))
        set_(self, "eval_loss", _as_tuple(self.eval_loss, float))
        set_(self, "eval_se", _as_tuple(self.eval_se, float))
        set_(self, "n_eval_seq", _as_tuple(self.n_eval_seq, int))
        set_(self, "eval_reached_se_target", _as_tuple(self.eval_reached_se_target, bool))
        lengths = {len(self.k_eval), len(self.p_eval), len(self.eval_loss), len(self.eval_se),
                   len(self.n_eval_seq), len(self.eval_reached_se_target)}
        if len(lengths) != 1:
            raise ValueError(
                "EvalResult eval-condition fields must be parallel, got lengths "
                f"k_eval={len(self.k_eval)} p_eval={len(self.p_eval)} "
                f"eval_loss={len(self.eval_loss)} eval_se={len(self.eval_se)} "
                f"n_eval_seq={len(self.n_eval_seq)} "
                f"eval_reached_se_target={len(self.eval_reached_se_target)}")

    @property
    def n_eval_points(self) -> int:
        return len(self.eval_loss)

    def points(self) -> list:
        """Flatten into one EvalPoint per eval condition, in stored order."""
        return [EvalPoint(k_eval=self.k_eval[i], p_eval=self.p_eval[i], eval_loss=self.eval_loss[i],
                          eval_se=self.eval_se[i], n_eval_seq=self.n_eval_seq[i],
                          reached_se_target=self.eval_reached_se_target[i], result=self)
                for i in range(self.n_eval_points)]

    @property
    def size_key(self) -> str:
        """Model-size label; runs with the same (n_embd, n_layer) share it."""
        return f"d{self.model_config.n_embd:04d}_L{self.model_config.n_layer:02d}"

    @classmethod
    def from_dict(cls, d: dict) -> "EvalResult":
        """Rebuild from JSON, reconstructing the two nested configs through their own."""
        d = dict(d)
        d["model_config"] = LlamaConfig.from_dict(d["model_config"])
        d["train_config"] = TrainConfig.from_dict(d["train_config"])
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    @classmethod
    def from_sweep_points(cls, points, *, model_config: LlamaConfig, train_config: TrainConfig,
                          k_train: float, p_train: float,
                          total_n_params: int, n_train_tokens: int,
                          n_non_embedding_params: int) -> "EvalResult":
        """Build one EvalResult from a whole fault-eval sweep, preserving point order."""
        points = list(points)
        return cls(
            model_config=model_config, train_config=train_config,
            k_train=k_train, p_train=p_train,
            k_eval=tuple(pt.k for pt in points),
            p_eval=tuple(pt.p for pt in points),
            eval_loss=tuple(pt.mean for pt in points),
            total_n_params=int(total_n_params),
            n_train_tokens=int(n_train_tokens),
            n_non_embedding_params=int(n_non_embedding_params),
            eval_se=tuple(pt.se for pt in points),
            n_eval_seq=tuple(pt.n_seq for pt in points),
            eval_reached_se_target=tuple(bool(pt.reached_target) for pt in points),
        )
