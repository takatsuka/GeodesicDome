# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
Compute backends: run the heavy array work on a GPU when one is available, otherwise on every CPU core.

    from mt.geodesicdome import backend

    b = backend.get_backend()          # auto: CUDA -> Apple GPU (MPS) -> multi-core CPU
    print(b)                           # e.g. Backend(torch, cuda:0, float32)

Nothing here is required: with only numpy installed you get the multi-threaded NumPy backend.
Installing an array library switches the GPU on automatically:

    pip install torch            # NVIDIA CUDA, Apple Silicon GPU (MPS) and fast multi-threaded CPU
    pip install cupy-cuda12x     # NVIDIA CUDA through CuPy (used when torch has no CUDA)

Selection order for ``get_backend('auto')`` (the default):

    1. torch on CUDA            (NVIDIA GPU)
    2. cupy on CUDA             (NVIDIA GPU, torch not installed or built without CUDA)
    3. torch on MPS             (Apple Silicon GPU)
    4. numpy on CPU             (all cores: blocks run in a thread pool, BLAS is multi-threaded)

('torch:cpu' is available too, but the threaded NumPy backend was faster in our benchmarks.)

Choose explicitly with an argument or the environment:

    get_backend('cuda')  get_backend('cuda:1')  get_backend('mps')  get_backend('cupy')
    get_backend('torch') get_backend('cpu')     get_backend('numpy')
    MTGEODESIC_BACKEND=cpu python train.py      # force the CPU
    MTGEODESIC_NUM_THREADS=8                    # CPU threads (default: every core)
    MTGEODESIC_DTYPE=float64                    # GPU precision (default float32; CPU always float64)
    MTGEODESIC_MEMORY=512MB                     # size of one working block (default: from free memory)

A Backend offers the handful of array operations the geodesic libraries need, with the same
meaning on numpy, cupy and torch arrays, plus helpers to cut large jobs into blocks that fit
in memory (``block_rows``) and to run blocks on all CPU cores (``map_blocks``).
"""
from __future__ import annotations

import os
import re
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np

__all__ = ['Backend', 'get_backend', 'set_default_backend', 'available_backends', 'describe',
           'ENV_BACKEND', 'ENV_THREADS', 'ENV_DTYPE', 'ENV_MEMORY']

ENV_BACKEND = 'MTGEODESIC_BACKEND'
ENV_THREADS = 'MTGEODESIC_NUM_THREADS'
ENV_DTYPE = 'MTGEODESIC_DTYPE'
ENV_MEMORY = 'MTGEODESIC_MEMORY'

_MB = 1 << 20
_CPU_BLOCK = 128 * _MB                 # one working block on the CPU
_CPU_CACHE = 2048 * _MB                # largest matrix worth keeping between calls on the CPU


def _cpu_threads() -> int:
    env = os.environ.get(ENV_THREADS)
    if env:
        return max(1, int(env))
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:             # macOS / Windows
        return max(1, os.cpu_count() or 1)


def _parse_bytes(text: str) -> int:
    m = re.fullmatch(r'\s*([\d.]+)\s*([kmgt]?)i?b?\s*', text.lower())
    if not m:
        raise ValueError(f'cannot read a memory size from {text!r} (try e.g. 512MB or 2GB)')
    scale = {'': 1, 'k': 1 << 10, 'm': 1 << 20, 'g': 1 << 30, 't': 1 << 40}[m.group(2)]
    return int(float(m.group(1)) * scale)


# ======================================================================= Backend
class Backend:
    """
    One compute device plus the array library that drives it.

    Attributes
        name          'torch', 'cupy' or 'numpy'
        device        'cuda:0', 'mps', 'cpu', ...
        dtype         numpy dtype of the floating-point arrays made by `asarray`
        is_gpu        True on CUDA and MPS
        threads       CPU threads used (CPU backends)
        block_bytes   size of one working block (see block_rows)
        cache_bytes   largest matrix worth keeping on the device between calls
    """

    name = 'abstract'

    def __init__(self, device: str, dtype, is_gpu: bool, threads: int, block_bytes: int, cache_bytes: int):
        self.device = device
        self.dtype = np.dtype(dtype)
        self.is_gpu = is_gpu
        self.threads = threads
        env = os.environ.get(ENV_MEMORY)
        self.block_bytes = _parse_bytes(env) if env else int(block_bytes)
        self.cache_bytes = int(cache_bytes)

    def __repr__(self) -> str:
        extra = f', {self.threads} threads' if not self.is_gpu else ''
        return f'Backend({self.name}, {self.device}, {self.dtype.name}{extra})'

    @property
    def spec(self) -> str:
        """A string that get_backend() turns back into this backend."""
        return self.name if self.name == 'numpy' else f'{self.name}:{self.device}'

    def __reduce__(self):                          # pickles as its spec, re-resolved on load
        return get_backend, (self.spec,)

    # ----------------------------------------------------------------- blocks
    def block_rows(self, cols: int, arrays: int = 1, total: int | None = None) -> int:
        """Rows per block so that `arrays` (rows, cols) float arrays fit in one working block."""
        per_row = max(1, int(cols) * max(1, arrays) * self.dtype.itemsize)
        rows = max(1, self.block_bytes // per_row)
        return rows if total is None else max(1, min(rows, int(total)))

    def blocks(self, n: int, rows: int) -> list[slice]:
        return [slice(s, min(s + rows, n)) for s in range(0, n, rows)]

    def map_blocks(self, fn: Callable[[slice], object], n: int, rows: int) -> list:
        """fn(slice) for consecutive row blocks covering range(n); results in order."""
        return [fn(s) for s in self.blocks(n, rows)]

    def fits_cache(self, *shape: int) -> bool:
        return int(np.prod(shape)) * self.dtype.itemsize <= self.cache_bytes

    # ------------------------------------------------ operations (overridden)
    def asarray(self, x, dtype=None):  # pragma: no cover - interface
        raise NotImplementedError

    def as_index(self, x):  # pragma: no cover - interface
        raise NotImplementedError

    def to_numpy(self, a) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def synchronize(self) -> None:
        pass


# ------------------------------------------------------------ numpy-like (numpy, cupy)
class _ArrayModuleBackend(Backend):
    """Backend for libraries with the numpy interface (numpy itself and cupy)."""

    def __init__(self, xp, device, dtype, is_gpu, threads, block_bytes, cache_bytes):
        super().__init__(device, dtype, is_gpu, threads, block_bytes, cache_bytes)
        self.xp = xp

    def asarray(self, x, dtype=None):
        return self.xp.asarray(x, dtype=self.dtype if dtype is None else dtype)

    def as_index(self, x):
        return self.xp.asarray(x, dtype=self.xp.int64)

    def zeros(self, shape, dtype=None):
        return self.xp.zeros(shape, dtype=self.dtype if dtype is None else dtype)

    def exp(self, a):
        return self.xp.exp(a)

    def sqrt(self, a):
        return self.xp.sqrt(a)

    def arccos(self, a):
        return self.xp.arccos(a)

    def clip(self, a, lo, hi):
        return self.xp.clip(a, lo, hi)

    def maximum(self, a, b):
        return self.xp.maximum(a, b)

    def minimum(self, a, b):
        return self.xp.minimum(a, b)

    def abs(self, a):
        return self.xp.abs(a)

    def where(self, cond, a, b):
        return self.xp.where(cond, a, b)

    def isnan(self, a):
        return self.xp.isnan(a)

    def sum(self, a, axis=None):
        return a.sum(axis=axis)

    def min(self, a, axis=None):
        return a.min(axis=axis)

    def max(self, a, axis=None):
        return a.max(axis=axis)

    def argmin(self, a, axis=-1):
        return a.argmin(axis=axis)

    def argmax(self, a, axis=-1):
        return a.argmax(axis=axis)

    def smallest(self, a, k: int):
        """Indices of the k smallest entries of every row, in increasing order: (rows, k)."""
        xp = self.xp
        if k >= a.shape[1]:
            return xp.argsort(a, axis=1)[:, :k]
        part = xp.argpartition(a, k - 1, axis=1)[:, :k]
        order = xp.argsort(xp.take_along_axis(a, part, axis=1), axis=1)
        return xp.take_along_axis(part, order, axis=1)

    def largest(self, a, k: int):
        return self.smallest(-a, k)

    def gather(self, a, index):
        """a[i, index[i, j]] for every row i: (rows, k)."""
        return self.xp.take_along_axis(a, index, axis=1)

    def cat(self, arrays: Sequence, axis=0):
        return self.xp.concatenate(list(arrays), axis=axis)

    def index_add(self, target, index, source):
        """target[index[i]] += source[i] (repeated indices accumulate); in place, returns target."""
        np.add.at(target, index, source)
        return target

    def to_numpy(self, a) -> np.ndarray:
        return np.asarray(a)


class NumpyBackend(_ArrayModuleBackend):
    """NumPy on the CPU: BLAS uses every core, and map_blocks runs blocks on a thread pool."""

    name = 'numpy'

    def __init__(self, threads: int | None = None):
        super().__init__(np, 'cpu', np.float64, False, threads or _cpu_threads(), _CPU_BLOCK, _CPU_CACHE)
        self._pool: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()

    def _executor(self) -> ThreadPoolExecutor:
        with self._lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=self.threads, thread_name_prefix='mtgeodesic')
            return self._pool

    def block_rows(self, cols: int, arrays: int = 1, total: int | None = None) -> int:
        rows = super().block_rows(cols, arrays)
        if total is not None and self.threads > 1:      # give every core at least one block
            rows = min(rows, max(64, -(-int(total) // self.threads)))
        return rows if total is None else max(1, min(rows, int(total)))

    def map_blocks(self, fn, n, rows):
        blocks = self.blocks(n, rows)
        if self.threads == 1 or len(blocks) < 2:
            return [fn(s) for s in blocks]
        # numpy releases the GIL inside its kernels, so blocks run on all cores at once
        return list(self._executor().map(fn, blocks))

    def index_add(self, target, index, source):
        index = np.asarray(index)
        if source.ndim == 1:
            target += np.bincount(index, weights=source, minlength=len(target))
        else:                                             # much faster than np.add.at for 2-D rows
            for k in range(source.shape[1]):
                target[:, k] += np.bincount(index, weights=source[:, k], minlength=len(target))
        return target


class CupyBackend(_ArrayModuleBackend):
    """CuPy on an NVIDIA GPU."""

    name = 'cupy'

    def __init__(self, device_index: int = 0, dtype=None):
        import cupy
        import cupyx

        self._cupyx = cupyx
        self._device_index = device_index
        cupy.cuda.Device(device_index).use()
        free, _total = cupy.cuda.Device(device_index).mem_info
        dtype = dtype or os.environ.get(ENV_DTYPE, 'float32')
        super().__init__(cupy, f'cuda:{device_index}', dtype, True, 1, max(64 * _MB, free // 8),
                         max(128 * _MB, free // 4))

    @property
    def spec(self) -> str:
        return f'cupy:{self._device_index}'

    def index_add(self, target, index, source):
        self._cupyx.scatter_add(target, index, source)
        return target

    def to_numpy(self, a) -> np.ndarray:
        return self.xp.asnumpy(a)

    def synchronize(self) -> None:
        self.xp.cuda.Device(self._device_index).synchronize()


# ------------------------------------------------------------------------ torch
class TorchBackend(Backend):
    """PyTorch on CUDA, Apple MPS or the CPU (intra-op parallel on every core)."""

    name = 'torch'

    def __init__(self, device: str = 'cpu', dtype=None, threads: int | None = None):
        import torch

        self.torch = torch
        dev = torch.device(device)
        if dev.type == 'cuda' and dev.index is None:
            dev = torch.device('cuda', torch.cuda.current_device())
        self.tdevice = dev
        is_gpu = dev.type in ('cuda', 'mps')
        if dev.type == 'cpu':
            if threads or os.environ.get(ENV_THREADS):     # torch already uses every core by default
                torch.set_num_threads(threads or _cpu_threads())
            threads = torch.get_num_threads()
            dtype, block, cache = dtype or np.float64, _CPU_BLOCK, _CPU_CACHE
        else:
            threads = 1
            dtype = dtype or os.environ.get(ENV_DTYPE, 'float32')
            if dev.type == 'mps':
                dtype = 'float32'                        # the Apple GPU has no float64
            free = self._free_memory(dev)
            block, cache = max(64 * _MB, free // 8), max(128 * _MB, free // 4)
        super().__init__(str(dev), dtype, is_gpu, threads, block, cache)
        self.tdtype = {np.dtype('float32'): torch.float32, np.dtype('float64'): torch.float64,
                       np.dtype('float16'): torch.float16}[self.dtype]

    def _free_memory(self, dev) -> int:
        torch = self.torch
        try:
            if dev.type == 'cuda':
                return int(torch.cuda.mem_get_info(dev)[0])
            if dev.type == 'mps':
                return int(torch.mps.recommended_max_memory()) - int(torch.mps.driver_allocated_memory())
        except Exception:  # noqa: BLE001 - older torch builds lack these queries
            pass
        return 2048 * _MB

    @property
    def spec(self) -> str:
        return f'torch:{self.device}'

    # ------------------------------------------------------------- conversion
    def asarray(self, x, dtype=None):
        torch = self.torch
        tdtype = self.tdtype if dtype is None else self._torch_dtype(dtype)
        if isinstance(x, torch.Tensor):
            return x.to(device=self.tdevice, dtype=tdtype)
        # convert on the host first (MPS cannot take float64), then move in one copy
        a = np.ascontiguousarray(x, dtype=tdtype_to_np(tdtype))
        return torch.from_numpy(a).to(self.tdevice)

    def as_index(self, x):
        torch = self.torch
        if isinstance(x, torch.Tensor):
            return x.to(device=self.tdevice, dtype=torch.int64)
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.int64)).to(self.tdevice)

    def _torch_dtype(self, dtype):
        torch = self.torch
        if isinstance(dtype, torch.dtype):
            return dtype
        d = np.dtype(dtype)
        return {np.dtype('float32'): torch.float32, np.dtype('float64'): torch.float64,
                np.dtype('float16'): torch.float16, np.dtype('int64'): torch.int64,
                np.dtype('int32'): torch.int32, np.dtype('bool'): torch.bool}[d]

    def to_numpy(self, a) -> np.ndarray:
        if isinstance(a, self.torch.Tensor):
            return a.detach().to('cpu').numpy()
        return np.asarray(a)

    def synchronize(self) -> None:
        if self.tdevice.type == 'cuda':
            self.torch.cuda.synchronize(self.tdevice)
        elif self.tdevice.type == 'mps':
            self.torch.mps.synchronize()

    # ------------------------------------------------------------- operations
    def zeros(self, shape, dtype=None):
        tdtype = self.tdtype if dtype is None else self._torch_dtype(dtype)
        return self.torch.zeros(shape, dtype=tdtype, device=self.tdevice)

    def exp(self, a):
        return self.torch.exp(a)

    def sqrt(self, a):
        return self.torch.sqrt(a)

    def arccos(self, a):
        return self.torch.arccos(a)

    def clip(self, a, lo, hi):
        return self.torch.clamp(a, lo, hi)

    def maximum(self, a, b):
        return self.torch.clamp_min(a, b) if not isinstance(b, self.torch.Tensor) else self.torch.maximum(a, b)

    def minimum(self, a, b):
        return self.torch.clamp_max(a, b) if not isinstance(b, self.torch.Tensor) else self.torch.minimum(a, b)

    def abs(self, a):
        return self.torch.abs(a)

    def where(self, cond, a, b):
        torch = self.torch
        ref = a if isinstance(a, torch.Tensor) else b
        if not isinstance(a, torch.Tensor):
            a = torch.as_tensor(a, dtype=ref.dtype, device=ref.device)
        if not isinstance(b, torch.Tensor):
            b = torch.as_tensor(b, dtype=ref.dtype, device=ref.device)
        return torch.where(cond, a, b)

    def isnan(self, a):
        return self.torch.isnan(a)

    def sum(self, a, axis=None):
        return a.sum() if axis is None else a.sum(dim=axis)

    def min(self, a, axis=None):
        return a.min() if axis is None else a.amin(dim=axis)

    def max(self, a, axis=None):
        return a.max() if axis is None else a.amax(dim=axis)

    def argmin(self, a, axis=-1):
        return a.argmin(dim=axis)

    def argmax(self, a, axis=-1):
        return a.argmax(dim=axis)

    def smallest(self, a, k: int):
        return self.torch.topk(a, k, dim=1, largest=False, sorted=True).indices

    def largest(self, a, k: int):
        return self.torch.topk(a, k, dim=1, largest=True, sorted=True).indices

    def gather(self, a, index):
        return self.torch.gather(a, 1, index)

    def cat(self, arrays: Sequence, axis=0):
        return self.torch.cat(list(arrays), dim=axis)

    def index_add(self, target, index, source):
        return target.index_add_(0, index, source.to(target.dtype))


def tdtype_to_np(tdtype):
    import torch

    return {torch.float32: np.float32, torch.float64: np.float64, torch.float16: np.float16,
            torch.int64: np.int64, torch.int32: np.int32, torch.bool: np.bool_}[tdtype]


# ===================================================================== selection
_default: Backend | None = None
_default_lock = threading.Lock()
_cache: dict[tuple, Backend] = {}


def _torch_status():
    try:
        import torch
    except Exception:  # noqa: BLE001 - any import failure means "not available"
        return None, False, False
    cuda = bool(torch.cuda.is_available())
    mps = bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())
    return torch, cuda, mps


def _cupy_ok() -> bool:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001
        return False


def available_backends() -> list[str]:
    """Specs of every backend that can run here, best first (the order 'auto' uses)."""
    torch, cuda, mps = _torch_status()
    found = []
    if cuda:
        found += [f'torch:cuda:{i}' for i in range(torch.cuda.device_count())]
    if _cupy_ok():
        found.append('cupy:0')
    if mps:
        found.append('torch:mps')
    found.append('numpy')
    if torch is not None:
        found.append('torch:cpu')
    return found


def _make(spec: str, dtype=None) -> Backend:
    s = spec.strip().lower()
    torch, cuda, mps = (None, False, False) if s == 'numpy' else _torch_status()

    if s in ('', 'auto', 'gpu'):
        if cuda:
            return TorchBackend('cuda', dtype)
        if _cupy_ok():
            return CupyBackend(0, dtype)
        if mps:
            return TorchBackend('mps', dtype)
        if s == 'gpu':
            raise RuntimeError('no GPU found (install torch with CUDA/MPS support, or cupy)')
        return NumpyBackend()
    if s in ('numpy', 'cpu'):
        return NumpyBackend()
    if s.startswith('cupy'):
        index = int(s.split(':', 1)[1]) if ':' in s else 0
        return CupyBackend(index, dtype)
    if s.startswith('torch'):
        if torch is None:
            raise RuntimeError('torch is not installed (pip install torch)')
        device = s.split(':', 1)[1] if ':' in s else ('cuda' if cuda else 'mps' if mps else 'cpu')
        return TorchBackend(device, dtype)
    if s.startswith('cuda'):
        if cuda:
            return TorchBackend(s, dtype)
        if _cupy_ok():
            return CupyBackend(int(s.split(':', 1)[1]) if ':' in s else 0, dtype)
        raise RuntimeError('CUDA is not available (install torch with CUDA, or cupy)')
    if s == 'mps':
        if not mps:
            raise RuntimeError('the Apple GPU (MPS) is not available (needs torch on Apple Silicon)')
        return TorchBackend('mps', dtype)
    raise ValueError(f'unknown backend {spec!r}; try auto, cuda, mps, cupy, torch, cpu or numpy')


def get_backend(spec: str | Backend | None = None, dtype=None) -> Backend:
    """
    The backend for `spec` (see the module docstring).  None means the default backend:
    the one set with set_default_backend(), else $MTGEODESIC_BACKEND, else 'auto'.
    Backends are cached, so calling this repeatedly is cheap.
    """
    global _default
    if isinstance(spec, Backend):
        return spec
    if spec is None and dtype is None:
        with _default_lock:
            if _default is None:
                _default = _make(os.environ.get(ENV_BACKEND, 'auto'))
            return _default
    key = ((spec or os.environ.get(ENV_BACKEND, 'auto')).strip().lower(), None if dtype is None else str(dtype))
    with _default_lock:
        if key not in _cache:
            _cache[key] = _make(key[0], dtype)
        return _cache[key]


def set_default_backend(spec: str | Backend | None) -> Backend:
    """Sets the backend used when none is given (None re-runs the automatic choice)."""
    global _default
    chosen = get_backend(spec) if spec is not None else _make(os.environ.get(ENV_BACKEND, 'auto'))
    with _default_lock:
        _default = chosen
    return chosen


def describe() -> str:
    """One line per available backend, marking the default -- handy for a quick check."""
    default = get_backend()
    lines = []
    for spec in available_backends():
        mark = '*' if spec == default.spec or (spec == 'numpy' and default.name == 'numpy') else ' '
        lines.append(f'{mark} {spec}')
    return f'default: {default}\n' + '\n'.join(lines)
