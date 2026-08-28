from dataclasses import dataclass

from nano_llama.config_base import ConfigMixin


@dataclass(frozen=True)
class FaultConfig(ConfigMixin):
    """Per-block fault probability p and block size k (FMAs between error checks)."""
    p: float = 0.0
    k: int = 4

    def to_spec(self):
        from nano_llama.llama import FaultSpec      # lazy: keeps this module jax-free
        return FaultSpec(p=self.p, k=self.k)
