"""Verification tests for the STANDARD (llama2.c / GPT-2) parametrization.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import io
import math
import contextlib
import unittest
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from nano_llama.llama import Llama, LlamaConfig, INIT_STD, apply_rotary_emb
from nano_llama.initializations import build_model, standard_init, standard_init_stds
from nano_llama.train import get_param_labels, build_standard_optimizer
from nano_llama.train_core import TrainConfig, build_lr_schedule, build_optimizer, realized_hparams, \
    print_realized_hparams


def cfg(n_embd=64, n_layer=4, **kw):
    """A small manual-attention config (head_dim 32, matching the ladder)."""
    base = dict(n_embd=n_embd, n_layer=n_layer, n_head=max(1, n_embd // 32),
                n_kv_head=max(1, n_embd // 32), vocab_size=256, block_size=32,
                dtype="float32", attn_impl="manual", tie_embeddings=False)
    base.update(kw)
    return LlamaConfig(**base)


def quiet_model(c, seed=0, init_std=INIT_STD):
    with contextlib.redirect_stdout(io.StringIO()):          # Llama.__init__ prints the attn banner
        return build_model(c, jax.random.PRNGKey(seed), init_std=init_std)


# ============================================================================================
# 1. Initialization
# ============================================================================================
class InitTest(unittest.TestCase):
    def test_stds_are_plain_and_depth_scaled(self):
        for L in (2, 4, 16, 42):
            s = standard_init_stds(cfg(n_layer=L))
            self.assertEqual(s["plain"], 0.02)
            self.assertAlmostEqual(s["residual"], 0.02 / math.sqrt(2 * L), places=15)
            self.assertEqual(s["norm"], 1.0)

    def test_stds_are_width_invariant(self):
        """Init std is a constant, not a function of width: it must not move with n_embd."""
        for d in (64, 384, 1344):
            self.assertEqual(standard_init_stds(cfg(n_embd=d, n_layer=8))["plain"], 0.02)
            self.assertAlmostEqual(standard_init_stds(cfg(n_embd=d, n_layer=8))["residual"],
                                   0.02 / math.sqrt(16), places=15)

    def test_empirical_stds_match(self):
        L = 8
        m = quiet_model(cfg(n_embd=128, n_layer=L))
        exp = standard_init_stds(m.config)
        b = m.layers[0]
        for name, w in [("wq", b.attention.wq.weight), ("wk", b.attention.wk.weight),
                        ("wv", b.attention.wv.weight), ("w1", b.feed_forward.w1.weight),
                        ("w2", b.feed_forward.w2.weight), ("emb", m.tok_embeddings.weight),
                        ("lm_head", m.lm_head.weight)]:
            self.assertAlmostEqual(float(jnp.std(w)), exp["plain"], delta=0.0025, msg=name)
        for name, w in [("wo", b.attention.wo.weight), ("w3", b.feed_forward.w3.weight)]:
            self.assertAlmostEqual(float(jnp.std(w)), exp["residual"], delta=0.0015, msg=name)

    def test_residual_weights_shrink_with_depth(self):
        """wo/w3 must get SMALLER as the model gets deeper -- the whole point of the depth init."""
        shallow = quiet_model(cfg(n_layer=2))
        deep = quiet_model(cfg(n_layer=32))
        r = float(jnp.std(deep.layers[0].attention.wo.weight)) / \
            float(jnp.std(shallow.layers[0].attention.wo.weight))
        self.assertAlmostEqual(r, math.sqrt(2 / 32), delta=0.05)

    def test_norm_gains_left_at_ones(self):
        m = quiet_model(cfg())
        for blk in m.layers:
            np.testing.assert_allclose(np.asarray(blk.attention_norm.weight), 1.0)
            np.testing.assert_allclose(np.asarray(blk.ffn_norm.weight), 1.0)
        np.testing.assert_allclose(np.asarray(m.norm.weight), 1.0)

    def test_init_is_pure_in_key(self):
        c = cfg()
        with contextlib.redirect_stdout(io.StringIO()):
            skel = Llama(c, jax.random.PRNGKey(3))
        a = standard_init(skel, jax.random.PRNGKey(11))
        b = standard_init(skel, jax.random.PRNGKey(11))
        d = standard_init(skel, jax.random.PRNGKey(12))
        np.testing.assert_array_equal(np.asarray(a.layers[0].attention.wq.weight),
                                      np.asarray(b.layers[0].attention.wq.weight))
        self.assertFalse(np.array_equal(np.asarray(a.layers[0].attention.wq.weight),
                                        np.asarray(d.layers[0].attention.wq.weight)))

    def test_init_std_is_threaded_through(self):
        m = quiet_model(cfg(n_layer=2), init_std=0.05)
        self.assertAlmostEqual(float(jnp.std(m.tok_embeddings.weight)), 0.05, delta=0.006)


# ============================================================================================
# 2. Optimizer -- one global AdamW, nothing shape-dependent
# ============================================================================================
class OptimizerTest(unittest.TestCase):
    def tc(self, **kw):
        base = dict(learning_rate=1e-3, min_lr=1e-4, max_iters=100, warmup_iters=10,
                    lr_decay_iters=100, weight_decay=0.1, adam_eps=1e-8, beta1=0.9, beta2=0.95,
                    grad_clip=1.0)
        base.update(kw)
        return TrainConfig(**base)

    def test_labels_cover_params_and_freeze_rope(self):
        m = quiet_model(cfg(n_layer=3))
        labels = get_param_labels(m)
        flat = [l for l in jax.tree_util.tree_leaves(labels, is_leaf=lambda x: isinstance(x, str))
                if isinstance(l, str)]
        counts = Counter(flat)
        # 3 blocks x 7 matmuls + embedding + lm_head = 23 decayed
        self.assertEqual(counts["decay"], 3 * 7 + 2)
        # 3 blocks x 2 pre-LN gains + final norm = 7 undecayed
        self.assertEqual(counts["no_decay"], 3 * 2 + 1)
        self.assertEqual(counts["frozen"], 2)                 # freqs_cos, freqs_sin

    def test_hparams_do_not_depend_on_shape(self):
        """The core regression this revert exists to prevent: no HP may move with width or depth."""
        tc = self.tc()
        seen = set()
        for d, L in [(64, 2), (384, 11), (1344, 42)]:
            rh = realized_hparams(tc, quiet_model(cfg(n_embd=d, n_layer=L)))
            g = rh["groups"]
            seen.add((g["decay"]["peak_lr"], g["decay"]["eps"], g["decay"]["weight_decay"],
                      g["no_decay"]["peak_lr"], g["no_decay"]["eps"], g["no_decay"]["weight_decay"]))
        self.assertEqual(len(seen), 1, f"optimizer hparams varied with model shape: {seen}")

    def test_both_groups_share_one_lr_and_eps(self):
        tc = self.tc()
        g = realized_hparams(tc, quiet_model(cfg()))["groups"]
        self.assertEqual(g["decay"]["peak_lr"], g["no_decay"]["peak_lr"])
        self.assertEqual(g["decay"]["eps"], g["no_decay"]["eps"])
        # peak_lr comes back through the optax schedule, which evaluates in float32 -> compare at
        # float32 precision rather than exactly.
        self.assertAlmostEqual(g["decay"]["peak_lr"], tc.learning_rate, places=9)
        self.assertEqual(g["decay"]["eps"], tc.adam_eps)

    def test_weight_decay_on_matmuls_only(self):
        tc = self.tc()
        g = realized_hparams(tc, quiet_model(cfg()))["groups"]
        self.assertEqual(g["decay"]["weight_decay"], tc.weight_decay)
        self.assertEqual(g["no_decay"]["weight_decay"], 0.0)

    def test_schedule_uses_config_lr_verbatim(self):
        tc = self.tc(learning_rate=7e-4, min_lr=7e-5)
        s = build_lr_schedule(tc)                                       # optax evaluates in float32
        self.assertAlmostEqual(float(s(0)), 0.0, places=12)             # warmup starts at 0
        self.assertAlmostEqual(float(s(tc.warmup_iters)), 7e-4, places=9)
        self.assertAlmostEqual(float(s(tc.lr_decay_iters)), 7e-5, places=9)

    def test_rope_is_frozen_and_weights_train(self):
        m = quiet_model(cfg(n_layer=2))
        tc = self.tc()
        opt = build_optimizer(tc, m)
        params, _ = eqx.partition(m, eqx.is_array)
        st, cur = opt.init(params), params
        for _ in range(12):                                   # past the 10-step warmup
            g = jax.tree_util.tree_map(jnp.ones_like, cur)
            upd, st = opt.update(g, st, cur)
            cur = eqx.apply_updates(cur, upd)
        np.testing.assert_array_equal(np.asarray(cur.freqs_cos), np.asarray(params.freqs_cos))
        np.testing.assert_array_equal(np.asarray(cur.freqs_sin), np.asarray(params.freqs_sin))
        self.assertGreater(float(jnp.max(jnp.abs(
            cur.layers[0].attention.wq.weight - params.layers[0].attention.wq.weight))), 0)
        self.assertGreater(float(jnp.max(jnp.abs(
            cur.layers[0].attention_norm.weight - params.layers[0].attention_norm.weight))), 0)

    def test_identical_updates_across_width_for_matched_grads(self):
        """Same gradient -> same AdamW update regardless of n_embd."""
        tc = self.tc(grad_clip=1e9, weight_decay=0.0)
        outs = []
        for d in (64, 128):
            m = quiet_model(cfg(n_embd=d, n_layer=2))
            opt = build_optimizer(tc, m)
            params, _ = eqx.partition(m, eqx.is_array)
            st, cur = opt.init(params), params
            for _ in range(12):
                g = jax.tree_util.tree_map(jnp.ones_like, cur)
                upd, st = opt.update(g, st, cur)
                cur = eqx.apply_updates(cur, upd)
            outs.append(float(jnp.max(jnp.abs(
                cur.layers[0].attention.wq.weight - params.layers[0].attention.wq.weight))))
        self.assertAlmostEqual(outs[0], outs[1], places=9)

    def test_printer_runs(self):
        rh = realized_hparams(self.tc(), quiet_model(cfg()))
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            print_realized_hparams(rh)
        out = buf.getvalue()
        self.assertIn("standard", out)
        for g in ("decay", "no_decay"):
            self.assertIn(g, out)


# ============================================================================================
# 3. Forward pass -- no parametrization multipliers
# ============================================================================================
class ForwardTest(unittest.TestCase):
    def test_residual_stream_is_a_plain_sum(self):
        """x_out == x_in + attn(norm(x)) + ffn(norm(x)) with NO branch multiplier."""
        m = quiet_model(cfg(n_layer=1))
        blk = m.layers[0]
        T = 4
        x = jax.random.normal(jax.random.PRNGKey(1), (T, m.config.n_embd))
        fc, fs = m.freqs_cos[:T], m.freqs_sin[:T]
        got = blk(x, fc, fs)
        h = x + blk.attention(jax.vmap(blk.attention_norm)(x), fc, fs)
        expect = h + blk.feed_forward(jax.vmap(blk.ffn_norm)(h))
        np.testing.assert_allclose(np.asarray(got), np.asarray(expect), rtol=1e-6, atol=1e-6)

    def test_readout_has_no_output_multiplier(self):
        """logits == lm_head @ norm(x), with no multiplier between the final norm and the readout."""
        m = quiet_model(cfg(n_layer=1))
        idx = jnp.arange(4) % m.config.vocab_size
        logits = m(idx)
        x = jax.vmap(m.tok_embeddings)(idx)
        fc, fs = m.freqs_cos[:4], m.freqs_sin[:4]
        x = m.layers[0](x, fc, fs)
        x = jax.vmap(m.norm)(x)
        expect = jax.vmap(lambda r: m.lm_head.weight @ r)(x)
        np.testing.assert_allclose(np.asarray(logits), np.asarray(expect), rtol=1e-5, atol=1e-5)

    def test_attention_scale_is_inv_sqrt_head_dim(self):
        """Reference-implement the manual attention path and confirm the logit scale is 1/sqrt(d_head)."""
        m = quiet_model(cfg(n_embd=64, n_layer=1), init_std=1.0)
        att = m.layers[0].attention
        T, H, hd = 5, att.n_head, att.head_dim
        x = jax.random.normal(jax.random.PRNGKey(7), (T, m.config.n_embd))
        got = att(x, m.freqs_cos[:T], m.freqs_sin[:T])

        def reference(scale):
            q = jax.vmap(lambda r: att.wq.weight @ r)(x).reshape(T, H, hd)
            k = jax.vmap(lambda r: att.wk.weight @ r)(x).reshape(T, H, hd)
            v = jax.vmap(lambda r: att.wv.weight @ r)(x).reshape(T, H, hd)
            q, k = apply_rotary_emb(q, k, m.freqs_cos[:T], m.freqs_sin[:T])
            q, k, v = (z.transpose(1, 0, 2) for z in (q, k, v))
            a = (q @ k.transpose(0, 2, 1)) * scale
            a = jnp.where(jnp.tril(jnp.ones((T, T), dtype=bool)), a, -jnp.inf)
            a = jax.nn.softmax(a.astype(jnp.float32), axis=-1)
            y = (a @ v).transpose(1, 0, 2).reshape(T, H * hd)
            return jax.vmap(lambda r: att.wo.weight @ r)(y)

        # rtol is loose because init_std=1.0 puts the activations near |90|, where float32 matmul
        # accumulation order alone moves the last digits; the two candidate scales differ by ~5.7x,
        # far above this.
        np.testing.assert_allclose(np.asarray(got), np.asarray(reference(1.0 / math.sqrt(hd))),
                                   rtol=1e-3, atol=1e-3)
        self.assertFalse(np.allclose(np.asarray(got), np.asarray(reference(1.0 / hd)),
                                     rtol=1e-2, atol=1e-2),
                         "attention is using the 1/head_dim scale, not 1/sqrt(head_dim)")


# ============================================================================================
# 4. Sweep recipe -- the sqrt-width LR rule, matching the July OWT runs
# ============================================================================================
class SweepRecipeTest(unittest.TestCase):
    # Peak LRs read out of the July OWT EvalResult JSONs
    # (/home/trevor/data/llama_training_summaries/2026-07-09-12-52-28), keyed by n_embd.
    JULY_LR = {64: 1e-3, 96: 0.000816496580927726, 128: 0.0007071067811865475,
               160: 0.0006324555320336759, 192: 0.0005773502691896258, 224: 0.0005345224838248488,
               256: 0.0005, 288: 0.0004714045207910317, 320: 0.0004472135954999579,
               384: 0.0004082482904638631, 448: 0.0003779644730092272, 512: 0.00035355339059327376,
               576: 0.0003333333333333333, 640: 0.00031622776601683794, 704: 0.00030151134457776363,
               768: 0.0002886751345948129, 896: 0.00026726124191242437, 1024: 0.00025,
               1184: 0.00023249527748763847}

    def test_peak_lr_matches_july_owt_runs(self):
        from experiment_util.gen_sweep import _peak_lr
        for d, expect in self.JULY_LR.items():
            self.assertAlmostEqual(_peak_lr(d), expect, places=15, msg=f"n_embd={d}")

    def test_peak_lr_is_sqrt_width(self):
        from experiment_util.gen_sweep import _peak_lr
        self.assertAlmostEqual(_peak_lr(256) / _peak_lr(64), 0.5, places=12)     # 4x width -> 1/2 LR

    def test_ladder_is_rungs_1_to_24(self):
        from experiment_util.standard_models import MODELS
        self.assertEqual(len(MODELS), 24)
        self.assertEqual((MODELS[0].n_embd, MODELS[0].n_layer), (64, 2))
        self.assertEqual((MODELS[22].n_embd, MODELS[22].n_layer), (1184, 37))
        self.assertEqual((MODELS[-1].n_embd, MODELS[-1].n_layer), (1344, 42))
        self.assertNotIn((1536, 48), [(m.n_embd, m.n_layer) for m in MODELS])

    def test_only_the_lr_varies_across_the_ladder(self):
        from experiment_util.standard_models import MODELS
        from experiment_util.gen_sweep import train_config_for
        fixed = set()
        for mc in MODELS[:8]:
            tc = train_config_for(mc, 20, n_total=10_000_000)     # fixed N -> identical horizon
            fixed.add((tc.weight_decay, tc.adam_eps, tc.beta1, tc.beta2, tc.grad_clip,
                       tc.init_std, tc.batch_size))
        self.assertEqual(len(fixed), 1, f"a non-LR hyperparameter varied across the ladder: {fixed}")

    def test_min_lr_is_one_tenth_of_peak(self):
        from experiment_util.standard_models import MODELS
        from experiment_util.gen_sweep import train_config_for
        for mc in (MODELS[0], MODELS[12], MODELS[-1]):
            tc = train_config_for(mc, 20, n_total=10_000_000)
            self.assertAlmostEqual(tc.min_lr, 0.1 * tc.learning_rate, places=15)

    def test_generated_config_is_loadable(self):
        """A freshly generated TrainConfig must round-trip through to_dict/from_dict unchanged."""
        from experiment_util.standard_models import MODELS
        from experiment_util.gen_sweep import train_config_for
        tc = train_config_for(MODELS[0], 20, n_total=10_000_000)
        self.assertEqual(TrainConfig.from_dict(tc.to_dict()), tc)


if __name__ == "__main__":
    unittest.main()
