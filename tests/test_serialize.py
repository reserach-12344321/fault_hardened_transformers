"""Round-trip tests for Llama.serialize / Llama.deserialize.

Run on CPU:
    JAX_PLATFORMS=cpu PYTHONPATH=<repo> /home/trevor/scienv/bin/python -m unittest tests.test_serialize -v
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")     # don't grab the shared GPU for a tiny serialize test

import tempfile
import unittest

import jax
import jax.numpy as jnp
import equinox as eqx

from nano_llama.llama import LlamaConfig, Llama


class LlamaSerializationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base_path = os.path.join(self.tmp.name, "test_model")
        # Tiny config (valid LlamaConfig fields only) to keep it fast.
        self.config = LlamaConfig(block_size=32, vocab_size=100, n_layer=2, n_head=2, n_embd=32)
        self.model = Llama(self.config, jax.random.PRNGKey(42))

    def tearDown(self):
        self.tmp.cleanup()

    def test_serialize_creates_both_files(self):
        """serialize(base) writes base.eqx (weights) + base.json (config)."""
        self.model.serialize(self.base_path)
        self.assertTrue(os.path.isfile(f"{self.base_path}.eqx"), "weights file not created")
        self.assertTrue(os.path.isfile(f"{self.base_path}.json"), "config file not created")

    def test_deserialize_config_matches(self):
        """The deserialized model carries an identical config."""
        self.model.serialize(self.base_path)
        loaded = Llama.deserialize(self.base_path)
        self.assertEqual(self.model.config, loaded.config)

    def test_deserialize_weights_match(self):
        """Every array leaf round-trips bit-for-bit, with identical pytree structure."""
        self.model.serialize(self.base_path)
        loaded = Llama.deserialize(self.base_path)
        orig_leaves, orig_def = jax.tree_util.tree_flatten(self.model)
        load_leaves, load_def = jax.tree_util.tree_flatten(loaded)
        self.assertEqual(orig_def, load_def, "pytree structure differs after round-trip")
        for a, b in zip(orig_leaves, load_leaves):
            if eqx.is_array(a):
                self.assertTrue(jnp.array_equal(a, b), "array weights differ after round-trip")
            else:
                self.assertEqual(a, b)

    def test_forward_pass_matches(self):
        """Original and deserialized models produce identical logits for the same input."""
        self.model.serialize(self.base_path)
        loaded = Llama.deserialize(self.base_path)
        idx = jnp.array([1, 5, 10, 50, 99])
        logits_orig = self.model(idx)              # spec=None -> deterministic (no fault)
        logits_loaded = loaded(idx)
        self.assertEqual(logits_orig.shape, (idx.shape[0], self.config.vocab_size))
        self.assertTrue(jnp.allclose(logits_orig, logits_loaded, atol=1e-6),
                        "logits differ between original and deserialized model")


if __name__ == "__main__":
    unittest.main()
