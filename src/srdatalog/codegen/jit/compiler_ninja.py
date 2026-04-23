'''Ninja + PCH compile orchestrator.

Emits a `build.ninja` in the cache dir that:
  1. Precompiles `srdatalog.h` into a `.pch` once per build (srdatalog.h
     pulls in boost/hana/mp11/RMM/spdlog — ~4s per TU to parse cold, so
     one PCH saves ~(N-1) * 4s on N-shard projects like doop).
  2. Compiles every project `.cpp` with `-include-pch srdatalog.pch`.
  3. Links the resulting objects + extra_sources into a shared library.

Invokes ninja via the `ninja` PyPI wheel so we don't need a system
binary — `pip install srdatalog` pulls the ninja wheel as a transitive
dep (~500 KB).

Contract matches `compile_jit_project` in `compiler.py`:
  - Input: `project_result` dict from `cache.write_jit_project`, plus a
    `CompilerConfig`.
  - Output: `BuildResult` with compile + link results.

The ThreadPoolExecutor orchestrator in `compiler.py` remains as a
fallback (env `SRDATALOG_JIT_NO_NINJA=1` or `use_ninja=False`) for
contributors without ninja installed or for debugging a single-TU
compile path.
'''

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

from srdatalog.codegen.jit.compiler import (
  BuildResult,
  CompilerConfig,
  CompileResult,
  _artifact_name,
  _base_cxx_flags,
)

# ---------------------------------------------------------------------------
# build.ninja emission
# ---------------------------------------------------------------------------


def _find_srdatalog_h(config: CompilerConfig) -> str | None:
  '''Locate `srdatalog.h` on the runtime include paths.'''
  for p in config.include_paths:
    candidate = os.path.join(p, "srdatalog.h")
    if os.path.isfile(candidate):
      return candidate
  return None


def _ninja_escape(s: str) -> str:
  '''Escape characters that ninja treats specially ($, :, space in
  build paths). Most of our paths don't contain these, but be safe.'''
  return s.replace("$", "$$").replace(":", "$:").replace(" ", "$ ")


def _join_flags(flags: list[str]) -> str:
  '''Ninja variable substitution doesn't need shell-like quoting in
  `command =` — ninja exec()s the command via a shell, so we just
  space-join. Flags with spaces are rare (paths). Quote them if they
  contain a space.'''
  out = []
  for f in flags:
    if " " in f:
      out.append(f'"{f}"')
    else:
      out.append(f)
  return " ".join(out)


def emit_build_ninja(
  project_result: dict,
  config: CompilerConfig,
  *,
  use_pch: bool = False,
  use_ccache: bool | None = None,
) -> str:
  '''Write `<cache_dir>/build.ninja` from `project_result` + `config`.

  Returns the absolute path to the emitted ninja file.

  Args:
    use_pch: opt-in split host/device PCH. Disabled by default because
      clang's CUDA + PCH pipeline is fragile on our runtime headers
      (ptxas chokes when `-Xclang -emit-pch` runs with `--cuda-*-only`
      on headers that transitively pull CUDA intrinsics via
      `gpu/search.h`). Keep the code path so future runtime-header
      cleanups can flip it on with `use_pch=True`.
    use_ccache: prepend `ccache` to the compile command when it's on
      PATH. Defaults to True iff `ccache` is found. Warm rebuilds
      after `rm -rf build/` go from ~97s → ~5s on doop with ccache.
      Override via `SRDATALOG_JIT_NO_CCACHE=1`.
  '''
  # Absolute paths so `ninja -C <cache_dir>` can resolve inputs no
  # matter what the caller's cwd is.
  project_dir = os.path.abspath(str(project_result["dir"]))
  main_cpp = os.path.abspath(str(project_result["main"]))
  batches = [os.path.abspath(str(b)) for b in project_result["batches"]]
  output_dir = os.path.abspath(config.output_dir) if config.output_dir else project_dir
  os.makedirs(output_dir, exist_ok=True)
  artifact = _artifact_name(output_dir, config.shared)
  cxx = config.resolved_cxx()

  # Common flags — same set the ThreadPoolExecutor path uses.
  cxx_flags = _base_cxx_flags(config)

  # Split PCH approach for clang CUDA mode: build a PCH pair (one for
  # the host pass, one for the device pass) from the same stub `.cu`.
  # clang's CUDA two-pass compile needs each pass's AST to match the
  # PCH's recorded target; feeding a single host PCH to both passes
  # fails with a target-mismatch diagnostic, and a combined PCH built
  # with plain `-x cuda` trips ptxas on PCH bytes. The split approach
  # is what clang actually supports — host-pass uses host PCH,
  # device-pass uses device PCH.
  pch_header: str | None = None
  pch_host_obj = ""
  pch_device_obj = ""
  pch_stub_path = ""
  pch_include_clause = ""
  if use_pch:
    pch_header = _find_srdatalog_h(config)
    if pch_header is None:
      use_pch = False
    else:
      pch_stub_path = os.path.join(output_dir, "_pch_stub.cu")
      with open(pch_stub_path, "w") as f:
        f.write(
          f'// Auto-generated stub for split host/device PCH of srdatalog.h\n'
          f'#include "{pch_header}"\n'
        )

  # PCH note to caller via the returned path — can't raise because
  # callers rely on this function being pure file emission.
  lines: list[str] = []
  lines.append("# Auto-generated by srdatalog.codegen.jit.compiler_ninja")
  lines.append(f"# Generated for project dir: {project_dir}")
  lines.append("")
  # ccache detection — transparent speedup for warm rebuilds. The
  # compiler command becomes `ccache clang++ ...`, which is all ccache
  # needs to cache the .o file content-addressed by the source+flags.
  if use_ccache is None:
    use_ccache = (
      os.environ.get("SRDATALOG_JIT_NO_CCACHE", "") != "1" and shutil.which("ccache") is not None
    )
  cc_prefix = "ccache " if use_ccache else ""

  lines.append(f"cxx = {cc_prefix}{cxx}")
  lines.append(f"cxx_flags = {_join_flags(cxx_flags)}")
  # Link flags live on one line too. Ninja passes them verbatim to the shell.
  link_flags_list = list(config.link_flags)
  libs_list = [f"-l{lib}" for lib in config.libs]
  lines.append(f"link_flags = {_join_flags(link_flags_list + libs_list)}")
  lines.append(f"extra_sources = {_join_flags(list(config.extra_sources))}")
  lines.append("")

  if use_pch:
    pch_host_obj = os.path.join(output_dir, "srdatalog.host.pch")
    pch_device_obj = os.path.join(output_dir, "srdatalog.device.pch")
    # `--cuda-host-only`  → runs only the host pass, no nvptx/ptxas
    # `--cuda-device-only` + `--cuda-gpu-arch` → runs only the device pass
    # Both use `-x cuda` so clang interprets the `.cu` stub as CUDA. The
    # `-Xclang -emit-pch` forces PCH emission for either pass.
    lines.append("rule pch_host")
    lines.append("  command = $cxx $cxx_flags --cuda-host-only -Xclang -emit-pch -c $in -o $out")
    lines.append("  description = PCH-HOST $out")
    lines.append("")
    lines.append("rule pch_device")
    lines.append("  command = $cxx $cxx_flags --cuda-device-only -Xclang -emit-pch -c $in -o $out")
    lines.append("  description = PCH-DEVICE $out")
    lines.append("")
    pch_include_clause = (
      f" -include-pch {_ninja_escape(pch_host_obj)} -include-pch {_ninja_escape(pch_device_obj)}"
    )

  # Two compile rules:
  #   `cxx_host_only` — `-x cuda --cuda-host-only`. Runs ONE pass (host)
  #                      instead of the default two (host + device).
  #                      Used for TUs that don't DEFINE `__global__`
  #                      kernels — main.cpp, runner shards, etc.
  #                      ~50% faster per TU because the device pass
  #                      (which also re-parses srdatalog.h) is skipped.
  #   `cxx`           — full two-pass CUDA compile for jit_batch_*.cpp
  #                      which host actual __global__ kernel definitions.
  lines.append("rule cxx_host_only")
  lines.append(f"  command = $cxx $cxx_flags --cuda-host-only{pch_include_clause} -c $in -o $out")
  lines.append("  description = CXX-HOST $out")
  lines.append("")
  lines.append("rule cxx")
  lines.append(f"  command = $cxx $cxx_flags{pch_include_clause} -c $in -o $out")
  lines.append("  description = CXX $out")
  lines.append("")
  lines.append("rule link")
  if config.shared:
    lines.append("  command = $cxx -shared -o $out $in $extra_sources $link_flags")
  else:
    lines.append("  command = $cxx -o $out $in $extra_sources $link_flags")
  lines.append("  description = LINK $out")
  lines.append("")

  # Build statements.
  if use_pch:
    host_line = (
      f"build {_ninja_escape(pch_host_obj)}: pch_host "
      f"{_ninja_escape(pch_stub_path)} | {_ninja_escape(pch_header)}"
    )
    device_line = (
      f"build {_ninja_escape(pch_device_obj)}: pch_device "
      f"{_ninja_escape(pch_stub_path)} | {_ninja_escape(pch_header)}"
    )
    lines.append(host_line)
    lines.append(device_line)
    lines.append("")

  sources = [main_cpp] + batches
  object_paths: list[str] = []
  for src in sources:
    obj = os.path.join(output_dir, Path(src).stem + ".o")
    object_paths.append(obj)
    stem = Path(src).stem
    # Every TU goes through the full two-pass CUDA compile. An earlier
    # version of this code compiled main.cpp with `--cuda-host-only` to
    # halve compile time, on the theory that main only references
    # kernels. That was wrong: main.cpp triggers template instantiations
    # of thrust/cub helpers (e.g. set_difference, unique, scan) that no
    # other TU also instantiates with the same template arguments.
    # Host-only compilation silently dropped their device side, so the
    # final `.so` was missing ~113 kernels relative to the Nim build
    # and `cuKernelGetFunction` returned INVALID_HANDLE at runtime when
    # the fixpoint tried to launch them (observed on doop/batik_interned
    # after step 8). Matching Nim's pipeline — full CUDA on every TU —
    # is the correct behavior.
    rule = "cxx"
    line = f"build {_ninja_escape(obj)}: {rule} {_ninja_escape(src)}"
    if use_pch:
      # Order-only deps (||) — PCH files must exist before the TU
      # compiles, but changes to the PCH don't force recompile of
      # the .o (ninja already tracks -include-pch via input).
      line += f" || {_ninja_escape(pch_host_obj)} {_ninja_escape(pch_device_obj)}"
    lines.append(line)
  lines.append("")

  lines.append(
    f"build {_ninja_escape(artifact)}: link " + " ".join(_ninja_escape(o) for o in object_paths)
  )
  lines.append("")
  lines.append(f"default {_ninja_escape(artifact)}")
  lines.append("")

  ninja_path = os.path.join(output_dir, "build.ninja")
  with open(ninja_path, "w") as f:
    f.write("\n".join(lines))
  return ninja_path


# ---------------------------------------------------------------------------
# Invocation
# ---------------------------------------------------------------------------


def _locate_ninja_binary() -> str:
  '''Prefer the `ninja` PyPI wheel's binary (installed alongside
  srdatalog), fall back to any ninja on PATH.'''
  try:
    import ninja as _ninja_pkg  # type: ignore[import-not-found]

    candidate = os.path.join(_ninja_pkg.BIN_DIR, "ninja")
    if os.path.isfile(candidate):
      return candidate
  except ImportError:
    pass
  sys_ninja = shutil.which("ninja")
  if sys_ninja:
    return sys_ninja
  raise RuntimeError(
    "ninja not found. Install via `pip install ninja` or disable "
    "the ninja backend with use_ninja=False / SRDATALOG_JIT_NO_NINJA=1."
  )


def compile_jit_project_ninja(
  project_result: dict,
  config: CompilerConfig | None = None,
  *,
  use_pch: bool = False,
) -> BuildResult:
  '''Drop-in replacement for `compile_jit_project` that goes through
  ninja + PCH. Returns the same BuildResult shape so callers don't
  need to change.

  Compile-failure reporting: on non-zero ninja exit, we return a
  single synthetic CompileResult holding ninja's captured output —
  we don't parse per-TU diagnostics out of ninja's stream (ninja
  already shows them verbatim to stderr). Callers can still read
  `.stderr` + exit code.
  '''
  config = config or CompilerConfig()
  ninja_path = emit_build_ninja(project_result, config, use_pch=use_pch)
  ninja_bin = _locate_ninja_binary()

  jobs = config.resolved_jobs()
  cmd = [
    ninja_bin,
    "-C",
    os.path.dirname(ninja_path),
    "-f",
    os.path.basename(ninja_path),
    f"-j{jobs}",
  ]

  start = time.perf_counter()
  proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
  elapsed = time.perf_counter() - start

  output_dir = config.output_dir or str(project_result["dir"])
  artifact = _artifact_name(output_dir, config.shared)

  # Build a BuildResult that looks like the ThreadPoolExecutor one. We
  # fold every compile into one pseudo-CompileResult (ninja printed
  # diagnostics already); the link result mirrors ninja's exit code.
  synthesized_compile = CompileResult(
    command=cmd,
    output=ninja_path,
    returncode=proc.returncode,
    stdout=proc.stdout,
    stderr=proc.stderr,
    elapsed_sec=elapsed,
  )
  link_result = None
  if proc.returncode == 0:
    link_result = CompileResult(
      command=cmd,
      output=artifact,
      returncode=0,
      stdout="",
      stderr="",
      elapsed_sec=0.0,
    )
  return BuildResult(
    artifact=artifact if proc.returncode == 0 else "",
    compile_results=[synthesized_compile],
    link_result=link_result,
    elapsed_sec=elapsed,
  )
