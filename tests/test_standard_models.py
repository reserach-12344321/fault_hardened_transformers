"""Invariant checks for the hand-written model ladder in experiment_util.standard_models.MODELS.

Since the LlamaConfigs are written out explicitly (not built by a helper), these pin the family
invariants so a future edit can't silently drift a rung off the locked architecture. Together they
fully characterize each rung's shape.

Run on CPU:
    JAX_PLATFORMS=cpu PYTHONPATH=<repo> /home/trevor/scienv/bin/python -m unittest tests.test_standard_models -v
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest

from experiment_util.standard_models import MODELS

HEAD_DIM = 32


class StandardModelsTest(unittest.TestCase):
    def test_count(self):
        self.assertEqual(len(MODELS), 24)

    def test_head_dim_is_32(self):
        for c in MODELS:
            with self.subTest(d=c.n_embd, L=c.n_layer):
                self.assertEqual(c.n_embd % c.n_head, 0, "n_embd must be divisible by n_head")
                self.assertEqual(c.n_embd // c.n_head, HEAD_DIM, "head_dim must be 32")

    def test_no_gqa_n_kv_head_equals_n_head(self):
        # n_kv_head is set explicitly to n_head (an int, not None): no GQA anywhere on the ladder.
        for c in MODELS:
            with self.subTest(d=c.n_embd, L=c.n_layer):
                self.assertEqual(c.n_kv_head, c.n_head)

    def test_locked_architecture(self):
        for c in MODELS:
            with self.subTest(d=c.n_embd, L=c.n_layer):
                self.assertEqual(c.vocab_size, 8192)
                self.assertEqual(c.block_size, 1024)
                self.assertEqual(c.multiple_of, 32)
                self.assertEqual(c.dtype, "bfloat16")
                self.assertEqual(c.attn_impl, "flash")
                self.assertFalse(c.tie_embeddings)

    def test_rung_dims_unique(self):
        pairs = [(c.n_embd, c.n_layer) for c in MODELS]
        self.assertEqual(len(pairs), len(set(pairs)), "duplicate (n_embd, n_layer) rung")


if __name__ == "__main__":
    unittest.main()
