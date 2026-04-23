'''Bundled C++ runtime headers + vendored deps.

The Python codegen emits `jit_batch_<rule>.cpp` files that `#include`
`srdatalog.h` (the bundled SRDatalog runtime). The runtime in turn
includes boost / highway / RMM / spdlog (also bundled, under
`vendor/`) plus CUDA SDK headers (NOT bundled — must come from a
system NVIDIA install; auto-detected via `find_cuda_root()`).

`runtime_include_paths()` returns every -I path needed to compile
batch files, EXCEPT the CUDA toolkit. Combine with
`cuda_include_paths()` and `cuda_compile_flags()` for a full
CompilerConfig:

    from srdatalog import CompilerConfig
    from srdatalog.runtime import (
      runtime_include_paths, cuda_include_paths,
      cuda_compile_flags, runtime_defines, runtime_undefines,
    )

    cfg = CompilerConfig(
      cxx="acpp",
      include_paths=runtime_include_paths() + cuda_include_paths(),
      defines=runtime_defines(),
      cxx_flags=cuda_compile_flags(),
    )
'''

from __future__ import annotations

import glob
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Bundled runtime + vendor paths
# ---------------------------------------------------------------------------


def runtime_include_path() -> str:
  '''Absolute path to the bundled `generalized_datalog/` headers.
  Kept for back-compat — prefer `runtime_include_paths()` (plural).'''
  return str(_HERE / "generalized_datalog")


def runtime_include_paths() -> list[str]:
  '''All bundled C++ -I paths (runtime + vendored deps). CUDA paths
  are NOT included here — see `cuda_include_paths()`.

  Order matters: more-specific subdirs first so `#include "../mir.h"`
  resolves correctly via path arithmetic (e.g. `build/../mir.h` =
  `mir.h`).
  '''
  rt = _HERE / "generalized_datalog"
  vendor = _HERE / "vendor"
  return [
    str(rt),
    str(rt / "build"),
    str(rt / "gpu" / "runtime"),
    str(rt / "gpu" / "runtime" / "instructions"),
    str(rt / "gpu" / "runtime" / "executor_impl"),
    str(vendor / "boost" / "include"),
    str(vendor / "highway" / "include"),
    str(vendor / "rmm" / "include"),
    str(vendor / "spdlog" / "include"),
  ]


def runtime_defines() -> list[str]:
  '''Preprocessor `-D` flags the runtime expects. Append to
  `CompilerConfig.defines`.

  `ENABLE_LOGGING` is intentionally NOT here — it pulls in boost::log
  and its phoenix/proto/spirit transitive deps (~50 MB of headers we
  don't ship in the wheel). Opt in by appending `"ENABLE_LOGGING"` to
  your own defines AND supplying system boost::log headers via an
  additional `-I` path, e.g.::

      cfg.defines.append("ENABLE_LOGGING")
      cfg.include_paths.append("/usr/include")  # system boost
  '''
  return [
    "LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE",
    "USE_CUDA",
    "BOOST_ATOMIC_NO_CMPXCHG16B",
    "SPDLOG_USE_STD_FORMAT",
    "NDEBUG",
    "_GLIBCXX_USE_CXX11_ABI=1",
  ]


def runtime_undefines() -> list[str]:
  '''Preprocessor `-U` undefines (paired with `defines`).'''
  return ["__GCC_HAVE_SYNC_COMPARE_AND_SWAP_16"]


def has_vendored_deps() -> bool:
  '''True if boost/highway/rmm/spdlog headers are present under
  `runtime/vendor/`. False on a fresh clone — call
  `srdatalog populate-vendor` (or use the install hook) to fetch
  them.'''
  vendor = _HERE / "vendor"
  return all((vendor / dep / "include").is_dir() for dep in ("boost", "highway", "rmm", "spdlog"))


# ---------------------------------------------------------------------------
# CUDA toolkit auto-detection
# ---------------------------------------------------------------------------


def find_cuda_root() -> str | None:
  '''Locate a usable CUDA toolkit. Tries (in order):
    - `$CUDA_HOME` / `$CUDA_PATH`
    - NVIDIA HPC SDK: `/opt/nvidia/hpc_sdk/.../cuda/<version>`
    - Standard install: `/usr/local/cuda`, `/opt/cuda`

  Among multiple HPC SDK CUDA versions, prefer one where
  `thrust/optional.h` exists — that header was removed in CUDA 13.0+
  but RMM (bundled in vendor/) still needs it. So 12.9 is preferred
  over 13.0+ on machines that have both.
  '''
  for var in ("CUDA_HOME", "CUDA_PATH"):
    if os.environ.get(var) and os.path.isdir(os.environ[var]):
      return os.environ[var]

  hpc_sdk = sorted(glob.glob("/opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/*"))
  standard = ["/usr/local/cuda", "/opt/cuda"]

  def _score(path: str) -> tuple[int, str]:
    if not os.path.isfile(os.path.join(path, "include", "cuda_runtime.h")):
      return (-1, path)
    has_thrust_opt = os.path.isfile(
      os.path.join(path, "targets", "x86_64-linux", "include", "thrust", "optional.h")
    ) or os.path.isfile(os.path.join(path, "include", "thrust", "optional.h"))
    return (1 if has_thrust_opt else 0, path)

  candidates = [(_score(p), p) for p in hpc_sdk + standard]
  candidates = [(s, p) for s, p in candidates if s[0] >= 0]
  if not candidates:
    return None
  candidates.sort(reverse=True)
  return candidates[0][1]


def cuda_include_paths(cuda_root: str | None = None) -> list[str]:
  '''All -I paths a CUDA-enabled compile needs. Auto-detects via
  `find_cuda_root()` when `cuda_root` is None; raises RuntimeError
  if no toolkit is found.'''
  root = cuda_root or find_cuda_root()
  if root is None:
    raise RuntimeError(
      "No CUDA toolkit found. Install one (NVIDIA HPC SDK or CUDA Toolkit) "
      "and set $CUDA_HOME, or pass `cuda_root=` explicitly."
    )
  paths = [
    os.path.join(root, "include"),
    os.path.join(root, "targets", "x86_64-linux", "include"),
    os.path.join(root, "targets", "x86_64-linux", "include", "crt"),
    # CUDA 13.0+ relocates libcudacxx (cuda/std/*, cuda/stream_ref, etc.)
    # under a `cccl/` subdir; older versions had it directly.
    os.path.join(root, "targets", "x86_64-linux", "include", "cccl"),
  ]
  # math_libs (curand, cublas) lives one level above `cuda/<ver>` in
  # NVIDIA HPC SDK layouts.
  parent = os.path.dirname(root)
  ver = os.path.basename(root)
  math_libs = os.path.join(parent, "..", "math_libs", ver, "targets", "x86_64-linux", "include")
  if os.path.isdir(math_libs):
    paths.append(math_libs)
  return [p for p in paths if os.path.isdir(p)]


def cuda_compile_flags(
  cuda_root: str | None = None,
  gpu_arch: str = "sm_89",
) -> list[str]:
  '''CUDA-specific clang flags (`-x cuda`, `--cuda-gpu-arch=`,
  `--cuda-path=`). Combine with `runtime_include_paths()` +
  `cuda_include_paths()` for a full CompilerConfig.'''
  root = cuda_root or find_cuda_root()
  if root is None:
    raise RuntimeError("CUDA toolkit not found")
  return [
    "-Qunused-arguments",
    "-x",
    "cuda",
    f"--cuda-gpu-arch={gpu_arch}",
    f"--cuda-path={root}",
    # Match Nim's docker/config.nims passC set so our compiled `.so`
    # matches the Nim executable's codegen. `-O3` changes GPU codegen
    # (loop unrolling, register allocation); `-m64` locks ABI bit-width;
    # the atomic flags keep boost::atomic from switching cmpxchg16b
    # variants under our feet.
    "-O3",
    "-g",
    "-m64",
    "-DSRDATALOG_GPU_AVAILABLE=1",
    "-DENABLE_LOGGING",
    "-DBOOST_ATOMIC_NO_CMPXCHG16B",
    "-U__GCC_HAVE_SYNC_COMPARE_AND_SWAP_16",
    "-Wno-unknown-cuda-version",
  ]


def _vendor_lib_dir() -> str | None:
  '''Locate the vendor boost-static libs.

  The emitted `.so` includes vendor boost HEADERS (version 1.90 by
  `BOOST_VERSION` in `vendor/include/boost/version.hpp`). Linking the
  system's `libboost_container.so.1.74` would be an ABI mismatch (16
  minor versions apart) and surfaces as downstream corruption — in
  practice, `cuKernelGetFunction: INVALID_HANDLE` on kernel launches.

  Search order:
    1. `SRDATALOG_VENDOR_LIB` env var — explicit override.
    2. The in-tree sibling at `<project>/src/srdatalog/runtime/vendor/lib`.
    3. The upstream Nim repo at `/home/stargazermiao/workspace/SRDatalog/src/srdatalog/vendor/lib`
       — fallback for dev machines where the wheel's populate_vendor hook
       hasn't run.
  '''
  env = os.environ.get("SRDATALOG_VENDOR_LIB")
  if env and os.path.isdir(env):
    return env
  here = os.path.dirname(os.path.abspath(__file__))
  local = os.path.join(here, "vendor", "lib")
  if os.path.isdir(local):
    return local
  nim = "/home/stargazermiao/workspace/SRDatalog/src/srdatalog/vendor/lib"
  if os.path.isdir(nim):
    return nim
  return None


def cuda_link_flags(cuda_root: str | None = None) -> list[str]:
  '''`-L` paths the link step needs. Pair with `cuda_libs()`.'''
  root = cuda_root or find_cuda_root()
  if root is None:
    raise RuntimeError("CUDA toolkit not found")
  candidates = [
    os.path.join(root, "lib64"),
    os.path.join(root, "targets", "x86_64-linux", "lib"),
    os.path.join(root, "lib"),
  ]
  flags = [f"-L{p}" for p in candidates if os.path.isdir(p)] + [
    # rpath embedding — .so stays loadable even when LD_LIBRARY_PATH
    # doesn't include the CUDA lib dir (common on NVIDIA HPC SDK
    # installs that rely on modulefile-set envs).
    f"-Wl,-rpath,{p}"
    for p in candidates
    if os.path.isdir(p)
  ]
  vendor = _vendor_lib_dir()
  if vendor is not None:
    flags.insert(0, f"-L{vendor}")
  return flags


def cuda_libs() -> list[str]:
  '''`-l<libname>` entries needed to satisfy the runtime's CUDA symbol
  references (cudaMemcpyAsync, cuLaunchKernel, ...) and the vendored
  boost libs whose ABI must match the vendored boost headers.

  Boost is listed BEFORE cudart so that the static `.a` in the vendor
  dir wins over any system `libboost_*.so` on the `-L` search path.
  '''
  return [
    "boost_container",
    "boost_log",
    "boost_log_setup",
    "boost_thread",
    "boost_filesystem",
    "boost_atomic",
    "cudart",
    "cuda",
  ]
