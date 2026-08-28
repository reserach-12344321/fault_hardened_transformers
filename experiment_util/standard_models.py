"""The scaling-ladder model configs, as one list of LlamaConfig (rungs 1-24 by position).

Locked architecture, identical for every rung: head_dim 32 (so n_head = n_embd // 32 and
n_kv_head = n_head, i.e. no GQA), vocab_size 8192, block_size 1024, bf16 compute, flash
attention, untied embeddings. Only (n_embd, n_layer) vary.

The configs carry no parametrization multipliers -- every rung is a plain shape, and the only
thing that varies with it across a sweep is the sqrt-width peak LR that gen_sweep bakes into
each run's TrainConfig.
"""
from typing import List

from nano_llama.llama import LlamaConfig

__all__ = ["MODELS"]

_LOCKED = dict(vocab_size=8192, block_size=1024, dtype="bfloat16", attn_impl="flash",
               tie_embeddings=False)

# Aspect ratio d_model / n_layer ~= 32; params ~0.1M -> ~0.93B, log-spaced (~1.14x/rung).
# The former rung 25 (d1536/L48, ~1.38B) is dropped: it is where the replicated fp32
# optimizer state stops fitting a single 40 GB GPU.
MODELS: List[LlamaConfig] = [
    LlamaConfig(n_embd=64,   n_layer=2,  n_head=2,  n_kv_head=2,  **_LOCKED),
    LlamaConfig(n_embd=64,   n_layer=3,  n_head=2,  n_kv_head=2,  **_LOCKED),
    LlamaConfig(n_embd=96,   n_layer=3,  n_head=3,  n_kv_head=3,  **_LOCKED),
    LlamaConfig(n_embd=96,   n_layer=4,  n_head=3,  n_kv_head=3,  **_LOCKED),
    LlamaConfig(n_embd=128,  n_layer=4,  n_head=4,  n_kv_head=4,  **_LOCKED),
    LlamaConfig(n_embd=128,  n_layer=6,  n_head=4,  n_kv_head=4,  **_LOCKED),
    LlamaConfig(n_embd=160,  n_layer=6,  n_head=5,  n_kv_head=5,  **_LOCKED),
    LlamaConfig(n_embd=192,  n_layer=6,  n_head=6,  n_kv_head=6,  **_LOCKED),
    LlamaConfig(n_embd=224,  n_layer=7,  n_head=7,  n_kv_head=7,  **_LOCKED),
    LlamaConfig(n_embd=256,  n_layer=8,  n_head=8,  n_kv_head=8,  **_LOCKED),
    LlamaConfig(n_embd=288,  n_layer=9,  n_head=9,  n_kv_head=9,  **_LOCKED),
    LlamaConfig(n_embd=320,  n_layer=10, n_head=10, n_kv_head=10, **_LOCKED),
    LlamaConfig(n_embd=384,  n_layer=11, n_head=12, n_kv_head=12, **_LOCKED),
    LlamaConfig(n_embd=448,  n_layer=12, n_head=14, n_kv_head=14, **_LOCKED),
    LlamaConfig(n_embd=512,  n_layer=13, n_head=16, n_kv_head=16, **_LOCKED),
    LlamaConfig(n_embd=512,  n_layer=16, n_head=16, n_kv_head=16, **_LOCKED),
    LlamaConfig(n_embd=576,  n_layer=18, n_head=18, n_kv_head=18, **_LOCKED),
    LlamaConfig(n_embd=640,  n_layer=20, n_head=20, n_kv_head=20, **_LOCKED),
    LlamaConfig(n_embd=704,  n_layer=22, n_head=22, n_kv_head=22, **_LOCKED),
    LlamaConfig(n_embd=768,  n_layer=24, n_head=24, n_kv_head=24, **_LOCKED),
    LlamaConfig(n_embd=896,  n_layer=28, n_head=28, n_kv_head=28, **_LOCKED),
    LlamaConfig(n_embd=1024, n_layer=32, n_head=32, n_kv_head=32, **_LOCKED),
    LlamaConfig(n_embd=1184, n_layer=37, n_head=37, n_kv_head=37, **_LOCKED),
    LlamaConfig(n_embd=1344, n_layer=42, n_head=42, n_kv_head=42, **_LOCKED),   # rung 24 (~0.93B)
]
