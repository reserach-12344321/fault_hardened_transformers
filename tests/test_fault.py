"""Unit tests for the hardware-fault mask `_fault` -- the primitive the whole fault model rests on.

Covers the two properties every call site depends on: the fraction of zeroed entries matches the fault
rate p, and a block that does not divide the axis is sliced back (not padded) with its short final
block still all-live-or-all-dead.

Runnable under pytest or directly (``python tests/test_fault.py``).
"""
import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")   # never fight live training on the shared GPU

import jax
import jax.numpy as jnp

from nano_llama.llama import _fault


class FaultMaskTest(unittest.TestCase):
    def test_zero_fraction_matches_p(self):
        p, block = 0.3, 4
        w = jnp.ones((512, 512))  # block divides 512 -> no ragged tail
        out = _fault(jax.random.PRNGKey(0), w, axis=1, block=block, p=p)

        zero_frac = float((out == 0.0).mean())

        # ~65k independent block draws -> std of the estimate ~0.0018; 0.01 is comfortably >4 sigma.
        assert abs(zero_frac - p) < 0.01


    def test_ragged_tail_when_block_does_not_divide(self):
        """When block does not divide the axis."""
        p, block, rows, n = 0.5, 8, 256, 50  # 50 = 6*8 + 2  -> 7 blocks, last of size 2
        w = jnp.ones((rows, n))
        out = _fault(jax.random.PRNGKey(1), w, axis=1, block=block, p=p)


        assert out.shape == (rows, n)  # not padded up to n_blocks*block = 56

        n_blocks = (n + block - 1) // block  # = 7
        for bi in range(n_blocks):
            lo, hi = bi * block, min((bi + 1) * block, n)
            col = out[:, lo:hi]  # final block is [48:50], width 2
            assert bool(jnp.all(col == col[:, :1])), f"block {bi} ([{lo}:{hi}]) not constant"


if __name__ == "__main__":
    unittest.main()
