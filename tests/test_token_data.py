"""Unit tests for nano_llama.token_data: the SlidingLoader (sole loader) + shared helpers.

Covers BOTH modes of the one loader:
  * whole-in-RAM (buffer_gb=None or >= file)  -- one resident block, no sliding (== the old SimpleLoader)
  * sliding                                   -- multiple megablocks, refresh + prefetch

The fixture writes token[i] == i (< 65536 so uint16 doesn't wrap), so a token's VALUE encodes its file
offset -- letting a test assert exactly which megablock a sampled window came from.

Run on CPU:
    JAX_PLATFORMS=cpu PYTHONPATH=<repo> /home/trevor/scienv/bin/python -m unittest tests.test_token_data -v
"""
import os
import shutil
import tempfile
import unittest

import numpy as np

from nano_llama.token_data import SlidingLoader, slots_in


class TokenDataTest(unittest.TestCase):
    # 60k train tokens / 6k per buffer = 10 megablocks; block 8, batch 4.
    N_TRAIN, BLOCK, BS, BUF = 60_000, 8, 4, 6_000

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="tokendata_")
        self.train = os.path.join(self.dir, "train.bin")
        self.val = os.path.join(self.dir, "val.bin")
        np.arange(self.N_TRAIN, dtype=np.uint16).tofile(self.train)
        (np.arange(2000, dtype=np.uint16) + 20_000).tofile(self.val)     # val tokens 20000..21999
        self.buffer_gb = self.BUF * 2 / 1e9                              # sliding buffer -> 10 blocks
        self.n_blocks = self.N_TRAIN // self.BUF                         # 10
        self.R = self.BUF // (self.BS * self.BLOCK)                      # refresh cadence (steps)
        self._loaders = []

    def tearDown(self):
        for ld in self._loaders:
            ld.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def whole(self, path=None, seed=7):
        """A whole-in-RAM loader (buffer_gb=None) -- the plain non-sliding case."""
        ld = SlidingLoader(path or self.train, self.BLOCK, seed=seed, batch_size=self.BS)
        self._loaders.append(ld)
        return ld

    def sliding(self, seed=7, buffer_gb=None, **kw):
        ld = SlidingLoader(self.train, self.BLOCK, seed=seed, batch_size=self.BS,
                           buffer_gb=self.buffer_gb if buffer_gb is None else buffer_gb, **kw)
        self._loaders.append(ld)
        return ld

    # ==================================================================== interface
    def test_window_is_next_token_shift(self):
        x, y = self.whole().batch(self.BS)
        xn, yn = np.asarray(x), np.asarray(y)
        self.assertEqual(xn.shape, (self.BS, self.BLOCK))
        self.assertTrue((xn[:, 1:] == yn[:, :-1]).all())

    def test_block_shape(self):
        xs, ys = self.whole().block(3, self.BS)
        self.assertEqual(xs.shape, (3, self.BS, self.BLOCK))
        self.assertEqual(ys.shape, (3, self.BS, self.BLOCK))

    def test_batch_is_first_row_of_block(self):
        d = self.whole()
        bx, by = d.batch(self.BS, step=3, stream=1)
        kx, ky = d.block(1, self.BS, step=3, stream=1)
        self.assertTrue(np.array_equal(np.asarray(bx), np.asarray(kx)[0]))
        self.assertTrue(np.array_equal(np.asarray(by), np.asarray(ky)[0]))

    # ==================================================================== whole-in-RAM mode
    def test_whole_ram_is_a_single_block(self):
        d = self.whole()
        d.batch(self.BS, step=0)
        self.assertEqual(d.n_blocks, 1)
        self.assertEqual(len(d._resident), self.N_TRAIN)         # the whole file is resident
        self.assertIsNone(d._exec)                               # no prefetch thread when not sliding

    def test_none_buffer_equals_oversized_buffer(self):
        # buffer_gb=None must draw identically to an explicit buffer >= the file (both whole-in-RAM)
        a = np.asarray(self.whole().block(4, self.BS, step=0)[0])
        big = self.sliding(buffer_gb=4 * self.N_TRAIN * 2 / 1e9)
        b = np.asarray(big.block(4, self.BS, step=0)[0])
        self.assertEqual(big.n_blocks, 1)
        self.assertTrue(np.array_equal(a, b))

    def test_chunk_invariant(self):
        d = self.whole()
        whole = np.asarray(d.block(6, self.BS, step=0)[0])
        parts = np.concatenate([np.asarray(d.block(3, self.BS, step=s)[0]) for s in (0, 3)])
        self.assertTrue(np.array_equal(whole, parts))

    def test_reload_invariant(self):
        whole = np.asarray(self.whole().block(6, self.BS, step=0)[0])
        resumed = np.asarray(self.whole().block(4, self.BS, step=2)[0])
        self.assertTrue(np.array_equal(whole[2:], resumed))

    def test_step_keys_the_draw(self):
        d = self.whole()
        self.assertTrue(np.array_equal(np.asarray(self.whole().batch(self.BS, step=0)[0]),
                                       np.asarray(d.batch(self.BS, step=0)[0])))
        self.assertFalse(np.array_equal(np.asarray(d.batch(self.BS, step=0)[0]),
                                        np.asarray(d.batch(self.BS, step=7)[0])))

    def test_stream_separates_train_and_eval(self):
        d = self.whole()
        tr = np.asarray(d.batch(self.BS, step=3, stream=0)[0])
        ev = np.asarray(d.batch(self.BS, step=3, stream=1)[0])
        self.assertFalse(np.array_equal(tr, ev))

    def test_samples_stay_in_file(self):
        # train and val loaders share (seed, stream, step) but point at different files -> each draws
        # only from its own file (here: distinct token-value ranges), so their streams are independent.
        xt = np.asarray(self.whole(path=self.train).block(4, self.BS, stream=1)[0])
        xv = np.asarray(self.whole(path=self.val).block(4, self.BS, stream=1)[0])
        self.assertTrue(((xt >= 0) & (xt < self.N_TRAIN)).all(), "train loader must draw only from train.bin")
        self.assertTrue(((xv >= 20_000) & (xv < 22_000)).all(), "val loader must draw only from val.bin")

    # ==================================================================== sliding mode
    def test_block_geometry_and_refresh_cadence(self):
        d = self.sliding(); d._init_geometry()
        self.assertEqual(d.n_tokens, self.N_TRAIN)
        self.assertEqual(d.n_blocks, self.n_blocks)
        self.assertEqual(d.refresh_every_steps, self.R)
        self.assertEqual(sorted(d.perm.tolist()), list(range(self.n_blocks)))

    def test_samples_come_from_resident_megablock(self):
        d = self.sliding()
        xn = np.asarray(d.batch(self.BS, step=0)[0])
        start0 = int(d.perm[0]) * self.BUF
        self.assertTrue(((xn >= start0) & (xn < start0 + self.BUF)).all())

    def test_resident_is_pure_function_of_step(self):
        d = self.sliding()
        for s in [0, self.R - 1, self.R, 3 * self.R, 9 * self.R, 10 * self.R, 25 * self.R + 5]:
            with self.subTest(step=s):
                d.batch(self.BS, step=s)
                self.assertEqual(d._resident_pos, (s // self.R) % self.n_blocks)

    def test_prefetch_matches_synchronous_load(self):
        d = self.sliding()
        d.batch(self.BS, step=0)
        self.assertEqual(d._prefetch_pos, 1)
        prefetched = d._prefetch_future.result()
        self.assertTrue(np.array_equal(prefetched, d._load_block(1)))
        st, en = d._block_bounds(1)
        self.assertTrue(np.array_equal(prefetched, np.asarray(d._mm[st:en])))

    def test_warm_matches_cold_and_populates_resident(self):
        # a whole-in-RAM (val-style) loader has no executor until warm() lazily makes one; the warmed
        # draw must be byte-identical to a cold draw (warm only moves the read off the critical path).
        cold = self.whole(path=self.val)
        xc, yc = cold.block(4, self.BS, step=1000, stream=1)
        warm = SlidingLoader(self.val, self.BLOCK, seed=7, batch_size=self.BS)
        self._loaders.append(warm)
        self.assertIsNone(warm._exec)
        warm.warm(0)
        self.assertIsNotNone(warm._exec)
        warm._prefetch_future.result()                          # join the background load
        xw, yw = warm.block(4, self.BS, step=1000, stream=1)
        self.assertEqual(warm._resident_pos, 0)
        self.assertIsNotNone(warm._resident)
        self.assertTrue(np.array_equal(np.asarray(xc), np.asarray(xw)))
        self.assertTrue(np.array_equal(np.asarray(yc), np.asarray(yw)))

    def test_refresh_steps_pure_matches_loader(self):
        # The pre-construction cadence must equal what _init_geometry computes, so a caller
        # can decide "will this run slide?" without building the loader. Only for a buffer
        # that FITS the file: a bigger one degenerates to whole-in-RAM, and the divergence is
        # harmless there since it only gates prefetch, suppressed at n_blocks == 1 anyway.
        for bgb in (self.BUF * 2 / 1e9, self.N_TRAIN * 2 / 1e9):          # both <= the file
            with self.subTest(buffer_gb=bgb):
                ld = SlidingLoader(self.train, self.BLOCK, seed=7, batch_size=self.BS, buffer_gb=bgb)
                ld._init_geometry()
                self.assertEqual(SlidingLoader.refresh_steps(bgb, self.BS, self.BLOCK),
                                 ld.refresh_every_steps)

    def test_prefetch_does_not_change_draws(self):
        # THE invariant the run_training prefetch-gate relies on: prefetch is a pure performance
        # optimization, so a full sliding sweep across several megablock boundaries must draw
        # byte-identically whether prefetch is on or off. If this ever failed, gating prefetch by run
        # length would silently change the data stream.
        steps = 3 * self.R + 5                              # spans blocks 0->3
        on = self.sliding(prefetch=True)
        off = self.sliding(prefetch=False)
        xon, yon = on.block(steps, self.BS, step=0)
        xoff, yoff = off.block(steps, self.BS, step=0)
        self.assertTrue(np.array_equal(np.asarray(xon), np.asarray(xoff)))
        self.assertTrue(np.array_equal(np.asarray(yon), np.asarray(yoff)))

    def test_warm_on_sliding_multiblock_matches_no_warm(self):
        # the new small-buffer val is a SLIDING loader (n_blocks>1), not whole-in-RAM. warm() must load
        # only the block the target step reads and yield draws identical to the same loader without warm.
        step = 5 * self.R + 3
        ref = self.sliding()
        xr, yr = ref.block(4, self.BS, step=step, stream=1)
        w = self.sliding()
        w.warm(step)
        if w._prefetch_future is not None:
            w._prefetch_future.result()                    # join the background load
        xw, yw = w.block(4, self.BS, step=step, stream=1)
        self.assertGreater(w.n_blocks, 1)                  # genuinely sliding, not one resident block
        self.assertEqual(w._resident_pos, w._desired_pos(step))
        self.assertTrue(np.array_equal(np.asarray(xr), np.asarray(xw)))
        self.assertTrue(np.array_equal(np.asarray(yr), np.asarray(yw)))

    def test_prefetch_disabled_path(self):
        d = self.sliding(prefetch=False)
        d.batch(self.BS, step=3 * self.R)
        self.assertIsNone(d._exec)
        self.assertEqual(d._resident_pos, (3 * self.R // self.R) % self.n_blocks)

    def test_resume_continues_the_sweep(self):
        step = 25 * self.R + 5
        a = self.sliding(); a.batch(self.BS, step=step)
        b = self.sliding(); b.batch(self.BS, step=step)
        self.assertEqual(b._resident_pos, a._resident_pos)

    def test_same_seed_identical_perm_diff_seed_reorders(self):
        a, b, c = self.sliding(seed=7), self.sliding(seed=7), self.sliding(seed=99)
        a._init_geometry(); b._init_geometry(); c._init_geometry()
        self.assertTrue(np.array_equal(a.perm, b.perm))
        self.assertFalse(np.array_equal(a.perm, c.perm))

    def test_window_draws_reload_invariant_across_refresh(self):
        d = self.sliding()
        whole = np.asarray(d.block(6, self.BS, step=self.R - 3)[0])         # steps R-3..R+2 span 2 blocks
        resumed = np.asarray(self.sliding().block(3, self.BS, step=self.R)[0])
        self.assertTrue(np.array_equal(whole[3:], resumed))
        lo0, lo1 = int(d.perm[0]) * self.BUF, int(d.perm[1]) * self.BUF
        self.assertTrue((whole[0] >= lo0).all() and (whole[0] < lo0 + self.BUF).all())
        self.assertTrue((whole[3] >= lo1).all() and (whole[3] < lo1 + self.BUF).all())

    def test_chunk_invariant_across_refresh(self):
        d = self.sliding()
        whole = np.asarray(d.block(6, self.BS, step=self.R - 3)[0])
        parts = np.concatenate([np.asarray(self.sliding().block(3, self.BS, step=s)[0])
                                for s in (self.R - 3, self.R)])
        self.assertTrue(np.array_equal(whole, parts))

    # ============================================================ without-replacement slot tiling
    # The fixture writes token[i] == i, so x[:, 0] IS the window's start offset in the file. Every
    # test below reads start positions straight out of the returned tokens rather than reaching into
    # the loader's internals.
    def _starts(self, ld, n_steps, step=0, stream=0):
        x = np.asarray(ld.block(n_steps, self.BS, step=step, stream=stream)[0])
        return x[:, :, 0].ravel().astype(np.int64)

    def test_a_full_pass_draws_every_slot_at_most_once(self):
        """The whole point: R*BS windows over one megablock visit, no window drawn twice."""
        ld = self.sliding()
        starts = self._starts(ld, self.R)
        self.assertEqual(len(starts), self.R * self.BS)
        self.assertEqual(len(np.unique(starts)), len(starts), "a window was drawn twice")

    def test_a_full_pass_is_a_tiling_of_the_resident_block(self):
        """Starts are block_size-aligned, inside the resident megablock, and distinct."""
        ld = self.sliding()
        ld._init_geometry()
        lo = int(ld.perm[0]) * self.BUF
        starts = self._starts(ld, self.R)
        self.assertTrue(((starts - lo) % self.BLOCK == 0).all(), "windows are not block-aligned")
        self.assertTrue(((starts >= lo) & (starts < lo + self.BUF)).all(), "left the resident block")

    def test_unique_token_fraction_is_one(self):
        """The regression this change exists for."""
        ld = self.sliding()
        starts = self._starts(ld, self.R)
        drawn = self.R * self.BS * self.BLOCK
        covered = len(np.unique((starts[:, None] + np.arange(self.BLOCK)[None, :]).ravel()))
        self.assertEqual(covered, drawn, f"only {covered}/{drawn} tokens unique")

    def test_slot_permutation_invalidates_with_the_resident_block(self):
        """Serving block A's slots against block B's tokens would return windows from the wrong region
        with no error at all, so pin the cache: leave the block and come back.
        """
        ld = self.sliding()
        first = self._starts(ld, 1, step=0)
        self._starts(ld, 1, step=self.R)                      # different block -> rebuilds the cache
        again = self._starts(ld, 1, step=0)
        self.assertTrue(np.array_equal(first, again))

    def test_wrapping_a_full_file_pass_reshuffles_instead_of_replaying(self):
        """pos and j are both modular."""
        ld = self.whole()
        ld._init_geometry()
        period = ld.refresh_every_steps * ld.n_blocks
        self.assertEqual(ld.n_blocks, 1)
        self.assertFalse(np.array_equal(self._starts(ld, 1, step=0),
                                        self._starts(ld, 1, step=period)), "epoch 2 replayed epoch 1")

    def test_second_epoch_is_itself_duplicate_free(self):
        ld = self.whole()
        ld._init_geometry()
        period = ld.refresh_every_steps * ld.n_blocks
        starts = self._starts(ld, ld.refresh_every_steps, step=period)
        self.assertEqual(len(np.unique(starts)), len(starts))

    def test_streams_tile_the_same_block_differently(self):
        """Distinct streams must not score identical sequences (fault_eval gives each point."""
        from nano_llama.token_data import slots_in
        ld = self.sliding()
        a, b = self._starts(ld, self.R, stream=0), self._starts(ld, self.R, stream=1)
        self.assertFalse(np.array_equal(a, b))
        self.assertEqual(len(np.unique(b)), len(b))
        # A visit consumes R*BS of the block's slots_in() slots, so the two streams differ only by
        # which few they leave over -- NOT the same set in a different order.
        leftover = slots_in(self.BUF, self.BLOCK) - self.R * self.BS
        self.assertLessEqual(len(np.setdiff1d(a, b)), leftover)
        self.assertEqual(len(np.union1d(a, b)), len(np.unique(a)) + len(np.setdiff1d(b, a)))

    def test_batch_size_may_not_exceed_the_loader_stride(self):
        """Over-drawing would reach into the next step's slot allotment and duplicate windows."""
        ld = self.sliding()
        with self.assertRaises(ValueError):
            ld.block(1, self.BS + 1, step=0)
        ld.block(1, self.BS, step=0)                           # equal: fine
        ld.block(1, self.BS - 1, step=0)                       # fewer: fine, a subset of the allotment

    def test_under_drawing_is_still_duplicate_free_across_steps(self):
        """A smaller draw takes a SUBSET of its step's allotment."""
        ld = SlidingLoader(self.train, self.BLOCK, seed=7, batch_size=self.BS * 4,
                           buffer_gb=self.buffer_gb)
        self._loaders.append(ld)
        ld._init_geometry()
        x = np.asarray(ld.block(ld.refresh_every_steps, self.BS, step=0)[0])
        starts = x[:, :, 0].ravel()
        self.assertEqual(len(np.unique(starts)), len(starts), "under-drawing repeated a window")

    def test_oversized_buffer_is_clamped_to_the_file(self):
        """A buffer larger than the file degenerates to whole-in-RAM."""
        big = self.sliding(buffer_gb=100 * self.N_TRAIN * 2 / 1e9)
        big._init_geometry()
        self.assertEqual(big.buffer_tokens, self.N_TRAIN)
        self.assertEqual(big.n_blocks, 1)
        ref = self.whole()                                     # buffer_gb=None on the same file
        ref._init_geometry()
        self.assertEqual(big.refresh_every_steps, ref.refresh_every_steps)
        # and it must actually serve a full pass without running off the permutation
        x = np.asarray(big.block(big.refresh_every_steps, self.BS, step=0)[0])
        starts = x[:, :, 0].ravel()
        self.assertEqual(len(np.unique(starts)), len(starts))

    def test_refresh_cadence_never_outruns_the_slot_supply(self):
        """R*batch_size slots are handed out per visit and must exist in EVERY block."""
        from nano_llama.token_data import slots_in
        for buf_tokens in (6_000, 6_144, 8_192, 60_000):       # 6144 and 8192 are multiples of 8
            with self.subTest(buf_tokens=buf_tokens):
                r = SlidingLoader.refresh_steps(buf_tokens * 2 / 1e9, self.BS, self.BLOCK)
                self.assertLessEqual(r * self.BS, slots_in(buf_tokens, self.BLOCK))

    # ============================================================ geometry sweep + golden stream
    # Every defect found in this module so far was GEOMETRY-dependent (an oversized buffer, a
    # block_size that divides buffer_tokens, a batch larger than a block's slot supply) and the
    # fixture above has exactly one geometry -- 60,000/6,000 divides evenly, so even the ragged last
    # block is never exercised by it. These sweep the shapes instead of assuming one.
    GEOMETRIES = [
        # (n_tokens, block, batch, buffer_tokens or None, label)
        (60_000, 8, 4, 6_000, "even split, 10 blocks"),
        (60_000, 8, 4, None, "whole-in-RAM"),
        (6_500, 8, 4, 2_000, "RAGGED last block (3 blocks + remainder)"),
        (60_001, 8, 4, 6_000, "odd file length"),
        (60_000, 8, 4, 6_144, "block_size divides buffer_tokens"),
        (60_000, 8, 4, 8_192, "block_size divides buffer_tokens, 2^13"),
        (60_000, 16, 8, 6_000, "larger block + batch"),
        (60_000, 8, 1, 6_000, "batch of 1"),
        (60_000, 8, 4, 10 * 60_000, "buffer >> file (clamps)"),
        (2_000, 8, 4, None, "small file"),
    ]

    def _geo_loader(self, n_tokens, block, batch, buf_tokens, seed=5):
        path = os.path.join(self.dir, f"geo_{n_tokens}_{block}_{batch}_{buf_tokens}.bin")
        if not os.path.isfile(path):
            np.arange(n_tokens, dtype=np.uint16).tofile(path)
        ld = SlidingLoader(path, block, seed=seed, batch_size=batch,
                           buffer_gb=None if buf_tokens is None else buf_tokens * 2 / 1e9)
        self._loaders.append(ld)
        return ld

    def test_core_invariants_hold_across_geometries(self):
        """The four properties that must never break, over every shape the loader can take."""
        for n_tokens, block, batch, buf, label in self.GEOMETRIES:
            with self.subTest(geometry=label):
                ld = self._geo_loader(n_tokens, block, batch, buf)
                ld._init_geometry()
                R = ld.refresh_every_steps
                rd = lambda n, s: np.asarray(ld.block(n, batch, step=s)[0])[:, :, 0].ravel()

                # (a) a full block visit never repeats a window
                st = rd(R, 0)
                self.assertEqual(len(np.unique(st)), len(st), "duplicate window")
                # (b) every window is fully inside the file, target token included
                self.assertTrue((st >= 0).all() and (st + block < n_tokens).all(), "ran off the file")
                # (c) chunk-invariance, including a split that is not a clean halving
                if R >= 4:
                    self.assertTrue(np.array_equal(
                        rd(4, 0), np.concatenate([rd(1, 0), rd(2, 1), rd(1, 3)])), "not chunk-invariant")
                # (d) the cadence never outruns the slot supply of the SMALLEST block
                self.assertLessEqual(R * batch, slots_in(ld.buffer_tokens, block))

    def test_ragged_last_block_is_sampled_correctly(self):
        """The LAST block absorbs the remainder, so it is longer than the others and its slot count
        differs -- and n_slots is part of the slot-cache key.
        """
        ld = self._geo_loader(6_500, 8, 4, 2_000)
        ld._init_geometry()
        self.assertEqual(ld.n_blocks, 3)
        last = int(np.where(ld.perm == ld.n_blocks - 1)[0][0])      # when the sweep reaches it
        st = np.asarray(ld.block(ld.refresh_every_steps, 4,
                                 step=last * ld.refresh_every_steps)[0])[:, :, 0].ravel()
        lo = (ld.n_blocks - 1) * ld.buffer_tokens
        self.assertTrue(((st >= lo) & (st + 8 < 6_500)).all(), "left the ragged block")
        self.assertEqual(len(np.unique(st)), len(st))

    def test_batch_larger_than_the_block_slot_supply_is_rejected(self):
        """Without replacement, a block with fewer slots than batch_size cannot fill a step."""
        for n_tokens, block, batch, buf in [(1_000, 8, 200, 800), (20, 8, 4, None),
                                            (10_000, 8, 64, 100)]:
            with self.subTest(n_tokens=n_tokens, batch=batch):
                ld = self._geo_loader(n_tokens, block, batch, buf, seed=3)
                with self.assertRaises(ValueError):
                    ld.block(1, batch, step=0)

    def test_golden_stream_is_unchanged(self):
        """Pins the EXACT windows a known seed produces."""
        ld = self._geo_loader(60_000, 8, 4, 6_000, seed=1337)
        ld._init_geometry()
        self.assertEqual(ld.perm.tolist(), [7, 6, 3, 5, 2, 8, 4, 9, 0, 1])
        self.assertEqual(ld.refresh_every_steps, 187)
        rd = lambda n, s, st_: np.asarray(ld.block(n, 4, step=s, stream=st_)[0])[:, :, 0].ravel().tolist()
        self.assertEqual(rd(3, 0, 0), [44472, 43744, 47088, 45008, 46416, 43544,
                                       46056, 45056, 47912, 42120, 42272, 42016])
        self.assertEqual(rd(3, 0, 1), [43816, 42008, 46288, 44160, 47816, 45496,
                                       42536, 43352, 45400, 46096, 44640, 42624])
        self.assertEqual(rd(2, 187, 0), [36416, 38488, 37648, 38312, 39560, 39960, 37576, 41496])

    def test_epoch_wrap_on_a_sliding_multi_block_loader(self):
        """The epoch test above uses whole-in-RAM (n_blocks=1)."""
        ld = self.sliding()
        ld._init_geometry()
        period = ld.refresh_every_steps * ld.n_blocks
        self.assertFalse(np.array_equal(self._starts(ld, 1, step=0),
                                        self._starts(ld, 1, step=period)))
        e2 = self._starts(ld, ld.refresh_every_steps, step=period)
        self.assertEqual(len(np.unique(e2)), len(e2))

    def test_alternating_streams_do_not_corrupt_the_single_entry_cache(self):
        """The slot permutation is cached one entry deep."""
        ld = self.sliding()
        a0, b0 = self._starts(ld, 1, stream=0), self._starts(ld, 1, stream=1)
        for _ in range(3):                                  # thrash the cache
            self._starts(ld, 1, stream=1); self._starts(ld, 1, step=self.R, stream=0)
        self.assertTrue(np.array_equal(self._starts(ld, 1, stream=0), a0))
        self.assertTrue(np.array_equal(self._starts(ld, 1, stream=1), b0))

    def test_fuzz_core_invariants_over_random_geometries(self):
        """Randomised sweep of the geometry space, seeded so failures reproduce."""
        rng = np.random.default_rng(20260803)
        checked = rejected = 0
        for _ in range(250):
            block = int(rng.choice([2, 4, 8, 16, 32, 64]))
            batch = int(rng.choice([1, 2, 3, 4, 8, 16]))
            n_tokens = int(rng.integers(block * 4, 40_000))
            buf = None if rng.random() < 0.25 else int(rng.integers(block * 2, n_tokens * 2))
            seed = int(rng.integers(0, 10_000))
            path = os.path.join(self.dir, f"fz_{n_tokens}_{block}_{batch}_{buf}_{seed}.bin")
            if not os.path.isfile(path):
                np.arange(n_tokens, dtype=np.uint16).tofile(path)
            ld = SlidingLoader(path, block, seed=seed, batch_size=batch,
                               buffer_gb=None if buf is None else buf * 2 / 1e9, prefetch=False)
            self._loaders.append(ld)
            geo = dict(n_tokens=n_tokens, block=block, batch=batch, buf=buf, seed=seed)
            try:
                ld._init_geometry()
            except ValueError:
                rejected += 1
                continue                                    # documented-unsupported, not a silent bug
            with self.subTest(**geo):
                R = ld.refresh_every_steps
                rd = lambda n, s: np.asarray(ld.block(n, batch, step=s)[0])[:, :, 0].ravel()
                st = rd(min(R, 40), 0)
                self.assertEqual(len(np.unique(st)), len(st), "duplicate window")
                self.assertTrue((st >= 0).all(), "negative start")
                self.assertTrue((st + block < n_tokens).all(), "window ran off the end of the file")
                self.assertLessEqual(R * batch, slots_in(ld.buffer_tokens, block), "cadence > slots")
                if R >= 3:                                  # chunk-invariance, uneven split
                    self.assertTrue(np.array_equal(
                        rd(3, 0), np.concatenate([rd(1, 0), rd(2, 1)])), "not chunk-invariant")
                checked += 1
        self.assertGreater(checked, 100, "fuzz explored too few valid configs to be meaningful")

    def test_production_geometry(self):
        """Every other test runs at toy scale (block 8, batch 4)."""
        BLOCK, BATCH, BUF = 1024, 128, int(2.5e9) // 2
        n_slots = slots_in(BUF, BLOCK)
        R = SlidingLoader.refresh_steps(2.5, BATCH, BLOCK)
        self.assertEqual(n_slots, 1_220_703)
        self.assertEqual(R, 9_536)
        self.assertLessEqual(R * BATCH, n_slots)                      # 1,220,608 <= 1,220,703
        # the val loader at VAL_BUFFER_GB = 0.25 (P1-B's stated cadence)
        self.assertEqual(SlidingLoader.refresh_steps(0.25, BATCH, BLOCK), 953)
        # a full FineWeb pass must stay well clear of the longest run in the sweep (~241,748 steps)
        self.assertGreater(R * (350_518_081_148 // BUF), 10 * 241_748)

    def test_slot_permutation_is_built_once_per_block_not_per_step(self):
        """Guards a performance CLIFF, not a correctness bug."""
        import nano_llama.token_data as td
        ld = self.sliding()
        ld._init_geometry()
        calls, real = [], td.slot_perm

        def counting(*a, **kw):
            calls.append(a[2:4])                       # (block_index, epoch)
            return real(*a, **kw)

        td.slot_perm = counting
        try:
            # Drive it ONE STEP AT A TIME. A single block(R, ...) call is one _draw and would build
            # one permutation even with the cache disabled -- it cannot distinguish the two. The
            # repeated-batch() pattern is both the real cliff and the real access pattern:
            # fault_eval.estimate_point streams batches in a loop, one _draw per step.
            for s in range(12):
                ld.batch(self.BS, step=s)              # 12 steps, all inside block perm[0]
            self.assertEqual(len(calls), 1, f"rebuilt the permutation {len(calls)}x over 12 steps")
            for s in range(self.R, self.R + 5):        # cross into the next block
                ld.batch(self.BS, step=s)
            self.assertEqual(len(calls), 2, f"expected exactly one rebuild per block, got {len(calls)}")
        finally:
            td.slot_perm = real

    def test_last_slot_window_fits_exactly(self):
        """slots_in() is off-by-one bait: a window needs block_size+1 tokens (y is x shifted)."""
        for extra in (0, 1, 2):
            with self.subTest(extra=extra):
                n = 4 * self.BLOCK + 1 + extra
                ns = slots_in(n, self.BLOCK)
                self.assertLessEqual((ns - 1) * self.BLOCK + self.BLOCK + 1, n,
                                     "last slot's target token falls outside the file")

    # ==================================================================== observability
    def test_stats_do_not_perturb_the_draw(self):
        """The counters must be write-only from the sampling path's perspective, and silencing events
        must not move a single window either.
        """
        import nano_llama.token_data as td
        quiet = td.LOADER_QUIET
        try:
            td.LOADER_QUIET = True
            a = self._starts(self.sliding(), 3 * self.R)
            td.LOADER_QUIET = False
            b = self._starts(self.sliding(), 3 * self.R)
            self.assertTrue(np.array_equal(a, b), "logging changed the data")
        finally:
            td.LOADER_QUIET = quiet

    def test_stats_count_refreshes_and_perm_builds(self):
        ld = self.sliding(prefetch=False)
        ld._init_geometry()
        ld.block(2 * self.R, self.BS, step=0)                  # spans exactly two megablocks
        s = ld.stats
        self.assertEqual(s["n_refreshes"], 2)
        self.assertEqual(s["n_perm_builds"], 2)                # one per block, not per step
        self.assertEqual(s["prefetch_misses"], 2)              # prefetch off -> both cold
        self.assertEqual(s["prefetch_hits"], 0)
        self.assertEqual(s["n_windows"], 2 * self.R * self.BS)
        self.assertEqual(s["n_block_loads"], 2)
        self.assertIsNotNone(s["first_block_load_s"])
        self.assertEqual(s["max_epoch"], 0)

    def test_stats_record_prefetch_hits(self):
        """A hit means the training loop did NOT block on the read."""
        ld = self.sliding(prefetch=True)
        ld.block(2 * self.R, self.BS, step=0)
        self.assertGreaterEqual(ld.stats["prefetch_hits"], 1)

    def test_stats_track_epoch_wrap(self):
        ld = self.whole()
        ld._init_geometry()
        period = ld.refresh_every_steps * ld.n_blocks
        ld.batch(self.BS, step=0)
        self.assertEqual(ld.stats["max_epoch"], 0)
        ld.batch(self.BS, step=period)
        self.assertEqual(ld.stats["max_epoch"], 1)



if __name__ == "__main__":
    unittest.main()
