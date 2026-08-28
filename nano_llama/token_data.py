"""Token data loading: one loader per token file, with step-keyed sampling.

Windows are drawn WITHOUT replacement -- the resident megablock is tiled into
block_size-aligned slots consumed in a seeded permutation, so a run sees each token once
per pass. SlidingLoader keeps one megablock resident and slides every buffer's-worth of
tokens, so only sequential reads touch disk; buffer_gb=None holds the whole file instead.

Both the schedule and the draws are pure functions of the global step:

    resident = perm[(step // refresh_every_steps) % n_blocks]
    epoch    = step // (refresh_every_steps * n_blocks)
    slots    = slot_perm(seed, stream, perm[pos], epoch, n_slots)
    windows  = slots[(step % refresh_every_steps) * batch : ...] * block_size

so the data at step S is identical however many preemptions preceded it, and independent of
chunk size and device count. `stream` namespaces uses within a run (training=0, eval=1).
"""
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp

# Loader events -> stderr (not stdout, which carries the banner other tools parse), one
# prefixed line each so a slurm log can be grepped. Rare by construction: one per megablock
# refresh. NANO_LLAMA_LOADER_QUIET=1 silences them -- worth it for a many-point eval array,
# where each point builds its own permutation.
LOADER_QUIET = bool(int(os.environ.get("NANO_LLAMA_LOADER_QUIET", "0")))


def _event(msg: str) -> None:
    if not LOADER_QUIET:
        print(f"[loader] {msg}", file=sys.stderr, flush=True)

def gather_xy(data: np.ndarray, ix: np.ndarray, block_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gather windows starting at `ix` -> host int32 (x, y), y being x shifted one token."""
    win = ix[:, None] + np.arange(block_size + 1, dtype=np.int64)[None, :]   # (n, block_size+1)
    chunk = np.asarray(data[win], dtype=np.int32)
    return chunk[:, :-1], chunk[:, 1:]


def slots_in(n_tokens: int, block_size: int) -> int:
    """How many non-overlapping (block_size+1)-token windows fit in `n_tokens`."""
    return max(1, (int(n_tokens) - 1) // int(block_size))


def slot_perm(seed: int, stream: int, block_index: int, epoch: int, n_slots: int) -> np.ndarray:
    """The order one megablock's window slots are consumed in -- a seeded permutation."""
    ss = np.random.SeedSequence([int(seed), int(stream), int(block_index), int(epoch)])
    return np.random.default_rng(ss).permutation(int(n_slots))


def step_keyed_ix(slots: np.ndarray, step: int, n_steps: int, batch_size: int,
                  refresh_steps: int, block_size: int, stride: int) -> np.ndarray:
    """Window start positions for `n_steps` consecutive steps, WITHOUT replacement."""
    j = np.arange(int(step), int(step) + int(n_steps), dtype=np.int64) % int(refresh_steps)
    idx = (j[:, None] * int(stride) + np.arange(int(batch_size), dtype=np.int64)[None, :]).ravel()
    return slots[idx].astype(np.int64) * int(block_size)


class SlidingLoader:
    """One token file, served from a sliding in-RAM megablock (memmap + background prefetch)."""

    def __init__(self, path: str, block_size: int, seed: int = 1337,
                 batch_size: int = 128, buffer_gb: Optional[float] = None, prefetch: bool = True):
        self.path = path
        self.block_size = block_size
        self.seed = seed                                           # keys the schedule AND the window draws
        self.batch_size = batch_size                               # sets refresh cadence
        # None -> resolved to the whole file (one block) once we know its length in _init_geometry.
        self.buffer_tokens = None if buffer_gb is None else self._buffer_tokens_for(buffer_gb, block_size)
        self._mm = None                                            # lazy memmap
        self.n_tokens = 0
        self.n_blocks = 1
        self.perm = None
        self.refresh_every_steps = 1
        self._resident: Optional[np.ndarray] = None
        self._resident_pos: Optional[int] = None
        # A single resident block never refreshes, so prefetch is only wired up when actually sliding.
        self._exec = ThreadPoolExecutor(max_workers=1) if (prefetch and buffer_gb is not None) else None
        # Whether to eagerly read the NEXT megablock after serving one. Tracked separately
        # from `_exec` because warm() creates the executor on demand even for a
        # prefetch=False loader, so gating the look-ahead on the executor alone would
        # silently re-enable reads the caller opted out of.
        self._prefetch_next = bool(prefetch and buffer_gb is not None)
        self._prefetch_pos: Optional[int] = None
        self._prefetch_future = None
        # Single-entry cache of the resident block's slot permutation, keyed on
        # (stream, block_index, epoch). Building it is free once per refresh and ruinous
        # per step. Size 1 rather than a dict because the access pattern is always "many
        # steps on one key, then move on"; a dict would leak one large array per eval point.
        self._slots: Optional[np.ndarray] = None
        self._slots_key: Optional[tuple] = None
        # Observability only: nothing here is read back by the sampling path.
        self._stats_lock = threading.Lock()          # _load_block also runs on the prefetch thread
        self.stats = dict(
            n_block_loads=0, block_load_s=0.0, first_block_load_s=None, last_block_load_s=None,
            max_block_load_s=0.0, n_refreshes=0, prefetch_hits=0, prefetch_misses=0,
            refresh_wait_s=0.0, n_perm_builds=0, perm_build_s=0.0, max_epoch=0,
            n_draws=0, n_windows=0)

    # ------------------------------------------------------------------ cadence (pure, no file)
    @staticmethod
    def _buffer_tokens_for(buffer_gb: float, block_size: int) -> int:
        """Megablock size in tokens for a numeric buffer_gb (uint16 -> 2 bytes/token)."""
        return max(block_size + 2, int(buffer_gb * 1e9) // 2)

    @staticmethod
    def refresh_steps(buffer_gb: float, batch_size: int, block_size: int) -> int:
        """Steps a megablock serves before the loader slides."""
        buf = SlidingLoader._buffer_tokens_for(buffer_gb, block_size)
        return max(1, slots_in(buf, block_size) // batch_size)

    # ------------------------------------------------------------------ geometry (lazy, once)
    def _init_geometry(self) -> None:
        """Open the file as a memmap and set the megablock permutation + refresh cadence (once)."""
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"missing {self.path} (run: python scripts/fineweb.py pretokenize)")
        self._mm = np.memmap(self.path, dtype=np.uint16, mode="r")
        self.n_tokens = int(len(self._mm))
        if self.buffer_tokens is None:                             # whole-in-RAM: one block == the file
            self.buffer_tokens = self.n_tokens
        # Clamp so buffer_tokens and the single block agree: a buffer larger than the file
        # degenerates to whole-in-RAM, and the tiling indexes a slot permutation sized off
        # buffer_tokens, so an oversized value would derive a cadence for a block that does
        # not exist. Hit whenever val.bin is smaller than VAL_BUFFER_GB.
        self.buffer_tokens = min(self.buffer_tokens, self.n_tokens)
        # Floor division so the LAST block absorbs the remainder (length in [buffer, 2*buffer)) --
        # never a tiny unsamplable tail. One block if the buffer is >= the file.
        self.n_blocks = max(1, self.n_tokens // self.buffer_tokens)
        self.perm = np.random.default_rng(self.seed).permutation(self.n_blocks)
        # Derived from buffer_tokens (the MINIMUM block length) rather than the resident
        # block's actual length: the last block absorbs the remainder and is longer, and a
        # cadence that varied per block would not be a schedule.
        block_slots = slots_in(self.buffer_tokens, self.block_size)
        # A step is allotted batch_size DISTINCT slots, so a block with fewer than that
        # cannot satisfy the without-replacement invariant. Caught here rather than left to
        # the max(1, ...) below, which would floor the cadence at 1 and let step 0 index
        # past the permutation with a bare IndexError.
        if block_slots < self.batch_size:
            raise ValueError(
                f"batch_size {self.batch_size} exceeds the {block_slots} non-overlapping windows in a "
                f"{self.buffer_tokens:,}-token block at block_size {self.block_size}. Windows are drawn "
                f"without replacement, so a step cannot be filled. Use a larger buffer_gb (or a file "
                f"with at least {self.batch_size * self.block_size + 1:,} tokens), or a smaller batch_size.")
        self.refresh_every_steps = max(1, block_slots // self.batch_size)
        # One line per loader. perm[0] is the seed-derived region every run starts on, which
        # is what makes a run's data provenance reconstructible after the fact.
        _event(f"open {os.path.basename(self.path)} tokens={self.n_tokens:,} "
               f"blocks={self.n_blocks} buffer={self.buffer_tokens:,} slots/block={block_slots:,} "
               f"R={self.refresh_every_steps:,} batch={self.batch_size} seed={self.seed} "
               f"perm[0]={int(self.perm[0])} pass={self.refresh_every_steps * self.n_blocks:,}steps")

    def _block_bounds(self, pos: int) -> Tuple[int, int]:
        idx = int(self.perm[pos])
        start = idx * self.buffer_tokens
        end = self.n_tokens if idx == self.n_blocks - 1 else (idx + 1) * self.buffer_tokens
        return start, end

    def _load_block(self, pos: int) -> np.ndarray:
        """Read one megablock from the memmap into a fresh RAM array (a sequential read)."""
        start, end = self._block_bounds(pos)
        t0 = time.perf_counter()
        arr = np.array(self._mm[start:end], dtype=np.uint16)          # forces the read into RAM
        dt = time.perf_counter() - t0
        with self._stats_lock:
            self.stats["n_block_loads"] += 1
            self.stats["block_load_s"] += dt
            self.stats["last_block_load_s"] = dt
            if self.stats["first_block_load_s"] is None:
                self.stats["first_block_load_s"] = dt
            if dt > self.stats["max_block_load_s"]:
                self.stats["max_block_load_s"] = dt
        return arr

    def _desired_pos(self, step: int) -> int:
        return (step // self.refresh_every_steps) % self.n_blocks

    def _ensure_resident(self, pos: int) -> None:
        if self._resident_pos == pos and self._resident is not None:
            return
        arr, hit = None, False
        t0 = time.perf_counter()
        if self._exec is not None and self._prefetch_pos == pos and self._prefetch_future is not None:
            arr = self._prefetch_future.result()                     # prefetch hit: already loaded
            hit = True
        if arr is None:
            arr = self._load_block(pos)                              # miss (first block, or a jump)
        wait_s = time.perf_counter() - t0
        self._resident, self._resident_pos = arr, pos
        self.stats["n_refreshes"] += 1
        self.stats["prefetch_hits" if hit else "prefetch_misses"] += 1
        self.stats["refresh_wait_s"] += wait_s
        # Rare enough to log every one, and the most useful line for spotting an FS-bound
        # job: `wait` is time the training loop was blocked, ~0 on a prefetch hit.
        _event(f"refresh block={int(self.perm[pos])}/{self.n_blocks} pos={pos} "
               f"src={'prefetch' if hit else 'COLD READ'} wait={wait_s:.2f}s "
               f"tokens={len(arr):,} file={os.path.basename(self.path)}")
        # Kick off prefetch of the NEXT megablock in the permuted sweep while we train on this one.
        if self._exec is not None and self._prefetch_next and self.n_blocks > 1:
            nxt = (pos + 1) % self.n_blocks
            self._prefetch_pos = nxt
            self._prefetch_future = self._exec.submit(self._load_block, nxt)

    def _resident_source(self, step: int) -> np.ndarray:
        if self.perm is None:
            self._init_geometry()
        self._ensure_resident(self._desired_pos(int(step)))
        return self._resident

    # ------------------------------------------------------------------ sampling -> device
    def _slot_perm(self, stream: int, block_index: int, epoch: int, n_slots: int) -> np.ndarray:
        """The resident block's slot permutation, cached on (stream, block_index, epoch)."""
        key = (int(stream), int(block_index), int(epoch), int(n_slots))
        if self._slots_key != key:
            t0 = time.perf_counter()
            self._slots = slot_perm(self.seed, stream, block_index, epoch, n_slots)
            self.stats["perm_build_s"] += time.perf_counter() - t0
            self.stats["n_perm_builds"] += 1
            # An epoch increment means the run has consumed the whole file once and is
            # re-shuffling it, which changes what "unique tokens" means for that run.
            if epoch > self.stats["max_epoch"]:
                self.stats["max_epoch"] = epoch
                _event(f"EPOCH {epoch} -- a full pass over {os.path.basename(self.path)} complete; "
                       f"re-tiling from a fresh permutation")
            self._slots_key = key
        return self._slots

    def _draw(self, source: np.ndarray, step: int, n_steps: int, batch_size: int, stream: int
              ) -> Tuple[np.ndarray, np.ndarray]:
        step = int(step)
        n_slots = slots_in(len(source), self.block_size)              # keeps the +1 target in range
        block_index = int(self.perm[self._desired_pos(step)])
        epoch = step // (self.refresh_every_steps * self.n_blocks)
        slots = self._slot_perm(stream, block_index, epoch, n_slots)
        ix = step_keyed_ix(slots, step, n_steps, batch_size, self.refresh_every_steps,
                           self.block_size, self.batch_size)
        self.stats["n_draws"] += 1
        self.stats["n_windows"] += len(ix)
        return gather_xy(source, ix, self.block_size)

    def _block_xy(self, step: int, n_steps: int, batch_size: int, stream: int
                  ) -> Tuple[np.ndarray, np.ndarray]:
        """Draw n_steps batches from global `step`."""
        if int(batch_size) > int(self.batch_size):
            raise ValueError(
                f"batch_size {batch_size} exceeds the loader's stride {self.batch_size}: each step is "
                f"allotted {self.batch_size} slots, so a larger draw would reuse the next step's and "
                f"duplicate windows. Construct the loader with batch_size >= the draw you intend.")
        xs, ys, s, remaining = [], [], int(step), n_steps
        while remaining > 0:
            src = self._resident_source(s)                           # resident block for step s (inits geometry)
            R = self.refresh_every_steps
            here = min(remaining, R - (s % R))                       # up to the next refresh boundary
            xh, yh = self._draw(src, s, here, batch_size, stream)
            xs.append(xh); ys.append(yh)
            s += here; remaining -= here
        return (xs[0], ys[0]) if len(xs) == 1 else (np.concatenate(xs), np.concatenate(ys))

    def warm(self, step: int = 0) -> None:
        """Start a background load of the megablock resident at `step`."""
        if self.perm is None:
            self._init_geometry()
        if self._exec is None:
            self._exec = ThreadPoolExecutor(max_workers=1)
        pos = self._desired_pos(int(step))
        self._prefetch_pos = pos
        self._prefetch_future = self._exec.submit(self._load_block, pos)

    def block(self, n_steps: int, batch_size: int, step: int = 0, stream: int = 0
              ) -> Tuple[jax.Array, jax.Array]:
        x, y = self._block_xy(step, n_steps, batch_size, stream)
        x = x.reshape(n_steps, batch_size, self.block_size)
        y = y.reshape(n_steps, batch_size, self.block_size)
        return jnp.asarray(x), jnp.asarray(y)

    def batch(self, batch_size: int, step: int = 0, stream: int = 0) -> Tuple[jax.Array, jax.Array]:
        """One batch -> device int32 (x, y), each (batch_size, block_size)."""
        x, y = self.block(1, batch_size, step, stream)
        return x[0], y[0]

    def close(self) -> None:
        if self._exec is not None:
            self._exec.shutdown(wait=False, cancel_futures=True)
            self._exec = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
