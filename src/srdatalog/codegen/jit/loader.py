'''Python ↔ C++ bridge: dlopen a compiled `.so` and call into it.

Phase 9 — closes the end-to-end loop started by Phases 6-8:
  - main_file.py emits the C++ source
  - cache.py writes the tree to disk
  - compiler.py turns it into a .so
  - THIS module turns that .so into callable Python.

The C++ runner methods (`<Ruleset>_Runner::run`, `load_data`, etc.)
are templates and can't be called directly from ctypes. Instead, the
user generates a small `extern "C"` shim that wraps the templated
calls into C-ABI entry points; this module handles the Python side
of that contract.

Public API:
  - `EntryPoint` — argtypes/restype spec for one C symbol.
  - `gen_runtime_shim_template(...)` — produces a starter shim.cpp
    the user fills in. Returned as a string so the caller can write
    it into the cache dir alongside main.cpp / jit_batch_N.cpp.
  - `JitRuntime` — ctypes.CDLL wrapper. Resolves symbols, applies
    the argspec, exposes typed `.call(name, *args)` / attribute
    shortcuts.
  - `build_and_load(...)` — one-shot: takes the project_result dict
    from `cache.write_jit_project` + a CompilerConfig, runs the
    full build, returns a ready `JitRuntime` on success.

Thread-safety: a single `JitRuntime` wraps one dlopen'd handle.
ctypes serializes calls through the C ABI so concurrent calls from
multiple Python threads are safe iff the underlying C function is.
We don't protect against that — it's the user's shim.
'''

from __future__ import annotations

import ctypes
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

# -----------------------------------------------------------------------------
# EntryPoint spec
# -----------------------------------------------------------------------------


@dataclass
class EntryPoint:
  '''Binding spec for one `extern "C"` function in the loaded .so.

  `name` — the exported symbol name (post-`extern "C"` name-mangling).
  `argtypes` — ctypes argument types, in positional order.
  `restype` — return type. Default `c_int` suits "return 0 on success".
  `errcheck` — optional callable run after the call for result mapping
      / error translation. Same protocol as ctypes.Function.errcheck:
      `(result, func, arguments) -> final_result_or_raise`.
  '''

  name: str
  argtypes: list[Any] = field(default_factory=list)
  restype: Any = ctypes.c_int
  errcheck: Any = None


def _apply_errcheck_default(result, func, arguments):
  '''Default errcheck for entry points returning int: nonzero → raise.
  Keeps the common pattern (return 0 on OK, nonzero on error) terse.
  '''
  if result != 0:
    raise RuntimeError(f"{func.__name__} returned nonzero status {result}")
  return result


# -----------------------------------------------------------------------------
# JitRuntime
# -----------------------------------------------------------------------------


class JitRuntime:
  '''Wrapper around `ctypes.CDLL` with a declared entry-point map.

  Usage:
    rt = JitRuntime("libmyproj.so", entry_points=[
      EntryPoint("srdatalog_init", restype=ctypes.c_int),
      EntryPoint("srdatalog_run", argtypes=[ctypes.c_char_p]),
      EntryPoint("srdatalog_get_size",
                 argtypes=[ctypes.c_char_p],
                 restype=ctypes.c_uint64,
                 errcheck=None),
    ])
    rt.srdatalog_init()
    rt.srdatalog_run(b"/data")
    n = rt.srdatalog_get_size(b"Path")
  '''

  def __init__(
    self,
    lib_path: str,
    entry_points: Sequence[EntryPoint] = (),
    *,
    mode: int = ctypes.RTLD_GLOBAL,
  ):
    if not os.path.exists(lib_path):
      raise FileNotFoundError(f"JitRuntime: {lib_path} not found")
    self.lib_path = lib_path
    # RTLD_GLOBAL makes symbols available to subsequently loaded libs
    # — important if the user dlopens dependent .so's.
    self._cdll = ctypes.CDLL(lib_path, mode=mode)
    self._entry_points: dict[str, EntryPoint] = {}
    for ep in entry_points:
      self.bind(ep)

  def bind(self, ep: EntryPoint) -> Any:
    '''Resolve `ep.name` in the library and apply argtypes / restype
    / errcheck. Returns the bound ctypes function.'''
    try:
      fn = getattr(self._cdll, ep.name)
    except AttributeError as e:
      raise AttributeError(f"JitRuntime: symbol {ep.name!r} not found in {self.lib_path}") from e
    fn.argtypes = list(ep.argtypes)
    fn.restype = ep.restype
    # errcheck semantics:
    #   None  → apply default (raise-on-nonzero) when restype is c_int
    #   False → explicitly opt out of any errcheck
    #   callable → use as-is
    if callable(ep.errcheck):
      fn.errcheck = ep.errcheck
    elif ep.errcheck is None and ep.restype is ctypes.c_int:
      fn.errcheck = _apply_errcheck_default
    # else: ep.errcheck is False or restype isn't c_int — leave
    # ctypes' default (no errcheck) in place.
    self._entry_points[ep.name] = ep
    return fn

  def __getattr__(self, name: str) -> Any:
    '''Attribute-style access to bound entry points.

    `rt.some_fn(args...)` resolves through the CDLL the first time
    and caches the typed function. Unbound symbols still surface as
    untyped ctypes functions — mirrors CDLL's default behavior.
    '''
    if name.startswith("_"):
      raise AttributeError(name)
    return getattr(self._cdll, name)

  def close(self) -> None:
    '''Drop the ctypes reference. Actual dlclose timing depends on
    Python's garbage collector — ctypes doesn't expose direct
    dlclose, so this is best-effort.'''
    self._cdll = None  # type: ignore[assignment]


# -----------------------------------------------------------------------------
# Shim template generator
# -----------------------------------------------------------------------------


def gen_runtime_shim_template(
  ruleset_name: str,
  db_blueprint_name: str,
  dest_relations: Sequence[tuple[str, str]] = (),
) -> str:
  '''Emit a starter `runtime_shim.cpp` with `extern "C"` entry points
  the Python loader binds.

  Args:
    ruleset_name: matches the `_Runner` struct name (so the shim
      calls `<ruleset>_Runner::load_data`, `run`).
    db_blueprint_name: user-declared blueprint type (e.g.,
      "TriangleDBBlueprint").
    dest_relations: list of `(symbol_suffix, cpp_type_name)` tuples
      exposing per-relation size queries. For each entry, the shim
      emits `uint64_t srdatalog_size_<suffix>()`.

  The template is a STARTING POINT — the caller may hand-edit the
  body (e.g., to add custom result-extraction logic) before handing
  it to `cache.write_jit_project`.

  The shim assumes the user has already included the main file
  (which defines the `_Runner` struct and DB blueprint) via
  `#include "main.cpp"` or similar — callers that shard the build
  differently need to adjust that line.
  '''
  header = [
    "// Auto-generated C-ABI shim for Python ctypes loader",
    "// Edit the bodies to suit your runtime shape.",
    "",
    '#include "srdatalog.h"',
    '#include "gpu/runtime/gpu.h"',
    '#include "gpu/runtime/query.h"',
    "",
    "// main.cpp provides the `_Runner` struct + DB blueprint.",
    '#include "main.cpp"',
    "",
    "namespace {",
    "using HostDB = SRDatalog::AST::SemiNaiveDatabase<" + db_blueprint_name + ">;",
    "HostDB* g_host_db = nullptr;",
    "",
    "// Device DB is templated on DeviceRelationType; hold via `void*` + cast on use.",
    "void* g_device_db = nullptr;",
    "} // anon namespace",
    "",
    'extern "C" {',
    "",
    "int srdatalog_init() {",
    "  try {",
    "    SRDatalog::GPU::init_cuda();",
    "    return 0;",
    "  } catch (...) { return 1; }",
    "}",
    "",
    "int srdatalog_run(const char* data_dir) {",
    "  try {",
    "    if (g_host_db) { delete g_host_db; g_host_db = nullptr; }",
    "    g_host_db = new HostDB();",
    f"    {ruleset_name}_Runner::load_data(*g_host_db, std::string(data_dir));",
    "    auto device_db = SRDatalog::GPU::copy_host_to_device(*g_host_db);",
    f"    {ruleset_name}_Runner::run(device_db);",
    "    return 0;",
    "  } catch (const std::exception& e) {",
    '    std::cerr << "srdatalog_run: " << e.what() << std::endl;',
    "    return 1;",
    "  }",
    "}",
    "",
    "int srdatalog_shutdown() {",
    "  if (g_host_db) { delete g_host_db; g_host_db = nullptr; }",
    "  return 0;",
    "}",
  ]
  # Per-destination size queries.
  for suffix, cpp_ty in dest_relations:
    header += [
      "",
      f"uint64_t srdatalog_size_{suffix}() {{",
      "  if (!g_host_db) return 0;",
      "  try {",
      f"    auto& rel = get_relation_by_schema<{cpp_ty}, FULL_VER>(*g_host_db);",
      "    return rel.size();",
      "  } catch (...) { return 0; }",
      "}",
    ]
  header += ["", "}  // extern \"C\""]
  return "\n".join(header) + "\n"


# -----------------------------------------------------------------------------
# One-shot build + load
# -----------------------------------------------------------------------------


def build_and_load(
  project_result: dict[str, Any],
  entry_points: Sequence[EntryPoint],
  compiler_config: Any | None = None,  # CompilerConfig; avoid circular import
  *,
  required_artifact: str | None = None,
) -> JitRuntime:
  '''Compile the Phase-7 project tree (via Phase 8) and dlopen the
  resulting .so.

  Raises on compile/link failure with the captured stderr in the
  message — the common failure mode during dev.

  `required_artifact` lets the caller override the default artifact
  name (e.g. the runner library name for a well-known runtime).
  '''
  from srdatalog.codegen.jit.compiler import CompilerConfig, compile_jit_project

  config = compiler_config or CompilerConfig()
  build = compile_jit_project(project_result, config)
  if not build.ok():
    errors: list[str] = []
    for r in build.compile_results:
      if r.returncode != 0:
        errors.append(f"[compile {r.output}] {r.stderr.strip() or '(no stderr)'}")
    if build.link_result and build.link_result.returncode != 0:
      errors.append(f"[link] {build.link_result.stderr.strip() or '(no stderr)'}")
    raise RuntimeError("build failed:\n" + "\n".join(errors))

  artifact = required_artifact or build.artifact
  if not artifact or not os.path.exists(artifact):
    raise FileNotFoundError(f"build succeeded but artifact missing: {artifact!r}")
  return JitRuntime(artifact, entry_points)
