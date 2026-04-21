'''Bundled C++ runtime headers + vendored dependencies.

The Python-emitted JIT code (`jit_batch_<rule>.cpp`) `#include`s
`srdatalog.h` which transitively pulls in:
  - `generalized_datalog/`  (the SRDatalog runtime — bundled here)
  - boost::hana, boost::atomic, boost::unordered (bundled subset)
  - highway (bundled subset)
  - rmm + spdlog (bundled subset; spdlog ships inside rmm's include)
  - CUDA SDK headers (NOT bundled — must come from user's NVIDIA install)

`runtime_include_paths()` returns every -I path needed to compile
batch files, EXCEPT the CUDA toolkit. `cuda_include_paths(cuda_root)`
returns the CUDA-specific paths given a toolkit root (auto-detected
via `find_cuda_root()` if omitted).

Typical usage from CompilerConfig:

    from srdatalog.runtime import runtime_include_paths, cuda_include_paths
    cfg = CompilerConfig(
      include_paths=runtime_include_paths() + cuda_include_paths(),
      defines=runtime_defines(),
    )
'''
from __future__ import annotations

import glob
import os
from pathlib import Path


_HERE = Path(__file__).resolve().parent


def runtime_include_path() -> str:
  '''Absolute path to the bundled `generalized_datalog/` headers.

  Kept for API back-compat — prefer `runtime_include_paths()` (plural)
  which returns the full bundled set including vendored deps.
  '''
  return str(_HERE / "generalized_datalog")


def runtime_include_paths() -> list[str]:
  '''All bundled C++ include directories needed to compile a JIT
  batch file (excluding CUDA SDK paths — see `cuda_include_paths()`).

  Order mirrors the parent xmake's compile_commands.json exactly:
  runtime root → build/ subdir → gpu/runtime + nested executor dirs
  → vendored deps. The `build/` subdir is critical: it's how relative
  `#include "../mir.h"` style paths resolve (`build/../mir.h` =
  `mir.h`).
  '''
  rt = _HERE / "generalized_datalog"
  vendor = _HERE / "vendor"
  return [
    # Match parent xmake order exactly.
    str(rt),
    str(rt / "build"),
    str(rt / "gpu" / "runtime"),
    str(rt / "gpu" / "runtime" / "instructions"),
    str(rt / "gpu" / "runtime" / "executor_impl"),
    # Vendored deps come AFTER runtime so runtime-supplied shims
    # (gpu/shim/cuda/...) can override CUDA-side decisions if needed.
    str(vendor / "boost" / "include"),
    str(vendor / "highway" / "include"),
    str(vendor / "rmm" / "include"),
    str(vendor / "spdlog" / "include"),
  ]


def runtime_defines() -> list[str]:
  '''Preprocessor `-D` flags the runtime expects to be set, mirroring
  the parent xmake build. Append to `CompilerConfig.defines`.'''
  return [
    "ENABLE_LOGGING",
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


# ---------------------------------------------------------------------------
# CUDA toolkit auto-detection
# ---------------------------------------------------------------------------

def find_cuda_root() -> str | None:
  '''Locate a usable CUDA toolkit. Tries (in order):
    - `$CUDA_HOME` / `$CUDA_PATH`
    - NVIDIA HPC SDK: `/opt/nvidia/hpc_sdk/.../cuda/<version>`
    - Standard install: `/usr/local/cuda`, `/opt/cuda`
  Returns the path or None. Caller raises a friendly error.

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
    '''Higher score = better. Prefer toolkits that have thrust/optional.h
    (RMM-compatible), then newer over older.'''
    if not os.path.isfile(os.path.join(path, "include", "cuda_runtime.h")):
      return (-1, path)
    has_thrust_opt = os.path.isfile(
      os.path.join(path, "targets", "x86_64-linux", "include", "thrust", "optional.h")
    ) or os.path.isfile(
      os.path.join(path, "include", "thrust", "optional.h")
    )
    return (1 if has_thrust_opt else 0, path)

  candidates = [(_score(p), p) for p in hpc_sdk + standard]
  candidates = [(s, p) for s, p in candidates if s[0] >= 0]
  if not candidates:
    return None
  candidates.sort(reverse=True)  # highest score, then lex
  return candidates[0][1]


def cuda_include_paths(cuda_root: str | None = None) -> list[str]:
  '''Return all -I paths a CUDA-enabled compile needs.

  If `cuda_root` is None, auto-detects via `find_cuda_root()`. Raises
  RuntimeError when no toolkit is found.
  '''
  root = cuda_root or find_cuda_root()
  if root is None:
    raise RuntimeError(
      "No CUDA toolkit found. Install one (NVIDIA HPC SDK, CUDA Toolkit) "
      "and set $CUDA_HOME, or pass `cuda_root=` explicitly."
    )
  paths = [
    os.path.join(root, "include"),
    os.path.join(root, "targets", "x86_64-linux", "include"),
    os.path.join(root, "targets", "x86_64-linux", "include", "crt"),
    # CUDA 13.0+ relocates libcudacxx (cuda/std/*, cuda/stream_ref, etc.)
    # under a `cccl/` subdir; older versions had it directly under include/.
    os.path.join(root, "targets", "x86_64-linux", "include", "cccl"),
  ]
  # math_libs (curand, cublas) lives one level above `cuda/<ver>` in
  # NVIDIA HPC SDK layouts: /opt/nvidia/hpc_sdk/.../math_libs/<ver>/...
  # `cuda_root` for HPC SDK ends in `cuda/12.9` → math_libs/12.9 is a
  # sibling of cuda.
  parent = os.path.dirname(root)
  ver = os.path.basename(root)
  math_libs = os.path.join(parent, "..", "math_libs", ver,
                            "targets", "x86_64-linux", "include")
  if os.path.isdir(math_libs):
    paths.append(math_libs)
  return [p for p in paths if os.path.isdir(p)]


def cuda_compile_flags(
  cuda_root: str | None = None,
  gpu_arch: str = "sm_89",
) -> list[str]:
  '''CUDA-specific clang flags (`-x cuda`, `--cuda-gpu-arch=`,
  `--cuda-path=`). Combine with `runtime_include_paths()` +
  `cuda_include_paths()` to build a full CompilerConfig.'''
  root = cuda_root or find_cuda_root()
  if root is None:
    raise RuntimeError("CUDA toolkit not found")
  return [
    "-Qunused-arguments",
    "-x", "cuda",
    f"--cuda-gpu-arch={gpu_arch}",
    f"--cuda-path={root}",
  ]
