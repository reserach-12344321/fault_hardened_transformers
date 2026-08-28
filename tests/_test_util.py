"""Shared helpers for the test suite. Not a test module -- the leading underscore keeps unittest
discovery off it.

Nothing here is production code. Anything that lives in this file is a convenience the TESTS want and
that no worker, prep script or analysis path uses; keeping it out of ``nano_llama`` is what stops a
test fixture from looking like part of the library.
"""
from __future__ import annotations

from nano_llama.llama import LlamaConfig


def build_model_config(n_embd: int, n_layer: int, *, head_dim: int = 32, vocab_size: int = 8192,
                       block_size: int = 1024, multiple_of: int = 32, dtype: str = "bfloat16",
                       attn_impl: str = "flash", tie_embeddings: bool = False) -> LlamaConfig:
    """One ladder-shaped LlamaConfig, built from (n_embd, n_layer) alone."""
    assert n_embd % head_dim == 0, f"n_embd {n_embd} not divisible by head_dim {head_dim}"
    n_head = n_embd // head_dim
    return LlamaConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=n_head,        # no GQA
        n_embd=n_embd,
        multiple_of=multiple_of,
        dtype=dtype,
        attn_impl=attn_impl,
        tie_embeddings=tie_embeddings,
    )
