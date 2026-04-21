'''C++ compiler wrapper — turns the on-disk `.cpp` tree from Phase 7
into a loadable `.so`.

Nim offloads compilation to its C++ backend via `{.compile:}` pragmas;
that backend shells out to the system compiler. Python has no such
backend, so we invoke a compiler (clang++/g++/nvcc) directly via
subprocess.

Design notes:
- `CompilerConfig` carries the full toolchain + flag set. All command
  assembly flows through `_build_compile_cmd` / `_build_link_cmd` —
  unit tests can verify argv without invoking a real compiler.
- Per-batch `.cpp` compilations run in parallel (ThreadPoolExecutor —
  subprocess bottleneck is I/O, threads are fine).
- Hash-based build cache: we sha256 the source + the CLI flags and
  write a `.stamp` sidecar file alongside each object file. If the
  stamp matches, compile is a no-op. Independent of the source
  mtime-guard in `cache.py` (content didn't change → no rewrite →
  no mtime bump) but works even when the source file is touched.
- `SRDATALOG_JIT_COMPILE_JOBS=N` overrides parallelism (0 = cpu_count).
- `SRDATALOG_JIT_SKIP_COMPILE=1` skips compile+link entirely — the
  stamp cache is trusted. Mirrors `SRDATALOG_SKIP_JIT_REGEN` on the
  cache side.

This module does NOT ship default include paths / link flags for
`generalized_datalog`. Callers (or a higher-level driver) must supply
those via `CompilerConfig.include_paths` / `link_flags`. The runtime
is built externally (xmake in this project); we link against its
artifacts.
'''
from __future__ import annotations

import concurrent.futures
import hashlib
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class CompilerConfig:
  '''Compile + link configuration.

  Defaults are minimal — `cxx_std=c++23` matches the project's xmake
  (`set_languages("cxx23")`). Callers add include/link/libs via the
  list fields. `extra_sources` is for object files / shared libs to
  feed into the final link (e.g., pre-built runtime artifacts).
  '''
  cxx: str = ""                              # empty → auto-detect
  cxx_std: str = "c++23"
  include_paths: list[str] = field(default_factory=list)
  defines: list[str] = field(default_factory=list)
  cxx_flags: list[str] = field(default_factory=list)
  link_flags: list[str] = field(default_factory=list)
  libs: list[str] = field(default_factory=list)
  extra_sources: list[str] = field(default_factory=list)
  output_dir: str = ""                       # empty → use cache dir
  jobs: int = 0                              # 0 → env or cpu_count
  shared: bool = True                        # shared .so vs static

  def resolved_cxx(self) -> str:
    return self.cxx or _detect_cxx()

  def resolved_jobs(self) -> int:
    if self.jobs > 0:
      return self.jobs
    env = os.environ.get("SRDATALOG_JIT_COMPILE_JOBS", "")
    if env:
      return max(1, int(env))
    return os.cpu_count() or 1


def _detect_cxx() -> str:
  '''First hit wins: $CXX → clang++ → g++. nvcc only if requested
  explicitly via config (it's a wrapper, not a drop-in C++ compiler).'''
  cxx = os.environ.get("CXX", "")
  if cxx:
    return cxx
  for candidate in ("clang++", "g++"):
    if shutil.which(candidate):
      return candidate
  raise RuntimeError(
    "no C++ compiler found — set $CXX or CompilerConfig.cxx"
  )


# -----------------------------------------------------------------------------
# Result types
# -----------------------------------------------------------------------------

@dataclass
class CompileResult:
  '''One compile invocation (source → object or link)'''
  command: list[str]
  output: str
  returncode: int = 0
  stdout: str = ""
  stderr: str = ""
  cached: bool = False                       # True = skipped via stamp
  elapsed_sec: float = 0.0


@dataclass
class BuildResult:
  '''Full `compile_jit_project` outcome.'''
  artifact: str                              # path to .so (or .a)
  compile_results: list[CompileResult] = field(default_factory=list)
  link_result: CompileResult | None = None
  elapsed_sec: float = 0.0

  def ok(self) -> bool:
    if self.link_result and self.link_result.returncode != 0:
      return False
    return all(r.returncode == 0 for r in self.compile_results)


# -----------------------------------------------------------------------------
# Command assembly (pure — no subprocess)
# -----------------------------------------------------------------------------

def _base_cxx_flags(config: CompilerConfig) -> list[str]:
  out = [f"-std={config.cxx_std}"]
  if config.shared:
    out.append("-fPIC")
  for d in config.defines:
    out.append(f"-D{d}")
  for i in config.include_paths:
    out.append(f"-I{i}")
  out.extend(config.cxx_flags)
  return out


def _build_compile_cmd(
  source: str, output: str, config: CompilerConfig,
) -> list[str]:
  cxx = config.resolved_cxx()
  cmd = [cxx, "-c", source, "-o", output]
  cmd += _base_cxx_flags(config)
  return cmd


def _build_link_cmd(
  objects: list[str], output: str, config: CompilerConfig,
) -> list[str]:
  cxx = config.resolved_cxx()
  cmd = [cxx]
  if config.shared:
    cmd.append("-shared")
  cmd += objects + config.extra_sources
  cmd += ["-o", output]
  cmd += config.link_flags
  for lib in config.libs:
    cmd.append(f"-l{lib}")
  return cmd


# -----------------------------------------------------------------------------
# Stamp-based cache
# -----------------------------------------------------------------------------

def _stamp_digest(source_path: str, argv: list[str]) -> str:
  '''Hash the source contents + the exact argv to invoke the compiler.
  If either changed (rename of flag, new -D, different include order),
  the stamp misses and we recompile.'''
  h = hashlib.sha256()
  try:
    with open(source_path, "rb") as f:
      h.update(f.read())
  except FileNotFoundError:
    h.update(b"<missing>")
  # argv serialized as null-separated bytes — avoids ambiguity on
  # flags that contain literal spaces.
  for a in argv:
    h.update(a.encode())
    h.update(b"\x00")
  return h.hexdigest()


def _stamp_path(object_path: str) -> str:
  return object_path + ".stamp"


def _check_stamp(object_path: str, digest: str) -> bool:
  '''True iff the stamp file exists AND its content matches `digest`
  AND the object file itself exists (don't trust a stale stamp whose
  object was deleted).'''
  if not os.path.exists(object_path):
    return False
  p = _stamp_path(object_path)
  try:
    with open(p) as f:
      return f.read().strip() == digest
  except FileNotFoundError:
    return False


def _write_stamp(object_path: str, digest: str) -> None:
  with open(_stamp_path(object_path), "w") as f:
    f.write(digest)


# -----------------------------------------------------------------------------
# Single-file compile
# -----------------------------------------------------------------------------

def compile_cpp(
  source: str, output: str, config: CompilerConfig,
) -> CompileResult:
  '''Compile one `.cpp` → `.o`. Short-circuits via stamp cache when
  the source + argv haven't changed. Never raises on compile error
  — returns a `CompileResult` with `returncode != 0` so the caller
  can aggregate.'''
  os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
  cmd = _build_compile_cmd(source, output, config)
  digest = _stamp_digest(source, cmd)
  if _check_stamp(output, digest):
    return CompileResult(command=cmd, output=output, cached=True)

  if os.environ.get("SRDATALOG_JIT_SKIP_COMPILE", "") == "1":
    return CompileResult(command=cmd, output=output, cached=True)

  start = time.perf_counter()
  proc = subprocess.run(
    cmd, capture_output=True, text=True, check=False,
  )
  elapsed = time.perf_counter() - start
  if proc.returncode == 0:
    _write_stamp(output, digest)
  return CompileResult(
    command=cmd, output=output, returncode=proc.returncode,
    stdout=proc.stdout, stderr=proc.stderr, elapsed_sec=elapsed,
  )


def link_shared(
  objects: list[str], output: str, config: CompilerConfig,
) -> CompileResult:
  '''Link objects + extra_sources into a shared library.'''
  os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
  cmd = _build_link_cmd(objects, output, config)
  if os.environ.get("SRDATALOG_JIT_SKIP_COMPILE", "") == "1":
    return CompileResult(command=cmd, output=output, cached=True)

  start = time.perf_counter()
  proc = subprocess.run(
    cmd, capture_output=True, text=True, check=False,
  )
  elapsed = time.perf_counter() - start
  return CompileResult(
    command=cmd, output=output, returncode=proc.returncode,
    stdout=proc.stdout, stderr=proc.stderr, elapsed_sec=elapsed,
  )


# -----------------------------------------------------------------------------
# Top-level: compile a Phase-7 project tree
# -----------------------------------------------------------------------------

def _artifact_name(project_dir: str, shared: bool) -> str:
  stem = os.path.basename(project_dir.rstrip("/"))
  ext = ".so" if shared else ".a"
  return os.path.join(project_dir, f"lib{stem}{ext}")


def compile_jit_project(
  project_result: dict[str, object],
  config: CompilerConfig | None = None,
) -> BuildResult:
  '''Compile the `.cpp` tree written by `cache.write_jit_project` into
  a shared library. `project_result` is the dict returned by that
  function — we pull `main` and `batches` out and feed them in.

  Returns a `BuildResult`. The caller inspects `.ok()` and
  `.compile_results`/`.link_result` for errors — this function never
  raises on a compile/link error.
  '''
  config = config or CompilerConfig()
  project_dir = str(project_result["dir"])
  main_cpp = str(project_result["main"])
  batches = list(project_result["batches"])

  output_dir = config.output_dir or project_dir
  os.makedirs(output_dir, exist_ok=True)

  sources: list[str] = [main_cpp, *batches]
  objects: list[str] = []
  results: list[CompileResult] = []

  start = time.perf_counter()

  # Parallel compile — ThreadPoolExecutor is enough since subprocess
  # doesn't hold the GIL during `run`.
  jobs = config.resolved_jobs()
  with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
    future_to_src = {}
    for src in sources:
      obj = os.path.join(
        output_dir, Path(src).stem + ".o",
      )
      objects.append(obj)
      future_to_src[pool.submit(compile_cpp, src, obj, config)] = src
    # Preserve input order so reports read sensibly.
    done = {f.result().output: f.result() for f in concurrent.futures.as_completed(future_to_src)}
    results = [done[obj] for obj in objects]

  all_ok = all(r.returncode == 0 for r in results)
  link_result: CompileResult | None = None
  if all_ok:
    artifact = _artifact_name(output_dir, config.shared)
    link_result = link_shared(objects, artifact, config)
  else:
    artifact = ""

  elapsed = time.perf_counter() - start
  return BuildResult(
    artifact=artifact if link_result and link_result.returncode == 0 else "",
    compile_results=results,
    link_result=link_result,
    elapsed_sec=elapsed,
  )
