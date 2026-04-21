'''Generic end-to-end runner for any benchmark auto-translated from Nim.

Takes a benchmark name (must match an `examples/<name>.py` produced by
`tools/nim_to_dsl.py`) and drives the full pipeline:

    DSL program → HIR → MIR → .cpp tree
                → ninja+clang → .so
                → dlopen → init → load CSVs → run → read sizes

Usage:

    # Triangle (synthetic small data, skips CSV load):
    python examples/run_benchmark.py triangle

    # Doop on batik:
    python examples/run_benchmark.py doop \\
        --data /path/to/doop_input_dir \\
        --meta /path/to/batik_meta.json

    # Transitive closure on a CSV:
    python examples/run_benchmark.py tc --data /path/to/edges

    # Run any benchmark, cap iterations for a quick sanity run:
    python examples/run_benchmark.py galen --data /path/to/data --max-iter 3

Benchmarks with `dataset_const` declarations (e.g. doop, ddisasm) need
`--meta <json>` pointing at the interning table. Others can omit it.

Each benchmark's `build_<name>_program()` returns a `Program`. When a
`build_<name>(meta_json)` function also exists, dataset_const
substitution is applied via that path.
'''
from __future__ import annotations

import argparse
import ctypes
import importlib
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


def _load_benchmark(name: str, meta_json: str | None):
  '''Import `examples/<name>.py` and return a `(program, consts_dict)` tuple.

  Prefers `build_<name>(meta_json)` when it exists AND a meta JSON was
  provided — that path does dataset_const substitution. Otherwise falls
  back to `build_<name>_program()`.
  '''
  m = importlib.import_module(name)
  sym_resolved = f"build_{name.lower()}"
  sym_program = f"build_{name.lower()}_program"
  # Translator sometimes uses the SCHEMA name rather than the file name
  # (e.g. doop.py emits `build_doopdb`). Fall back to anything that matches.
  if not hasattr(m, sym_resolved):
    sym_resolved = next(
      (s for s in dir(m) if s.startswith("build_") and not s.endswith("_program")
       and callable(getattr(m, s))),
      "",
    )
  if not hasattr(m, sym_program):
    sym_program = next(
      (s for s in dir(m) if s.startswith("build_") and s.endswith("_program")),
      "",
    )
  if meta_json and sym_resolved:
    prog, consts = getattr(m, sym_resolved)(meta_json)
    return prog, consts
  if not sym_program:
    raise RuntimeError(
      f"{name}.py: no build_* function found; translator expected "
      f"build_<name>_program() or build_<name>(meta_json)."
    )
  return getattr(m, sym_program)(), {}


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("benchmark", help="Name of examples/<benchmark>.py")
  p.add_argument("--data", default="",
                 help="Input CSV directory (required for benchmarks "
                      "whose relations have input_file set)")
  p.add_argument("--meta", default="",
                 help="Dataset meta JSON (needed when the benchmark "
                      "declares dataset_consts)")
  p.add_argument("--project", default="",
                 help="Project name (defaults to benchmark name "
                      "with first char capitalized + 'Plan')")
  p.add_argument("--cache-base", default="./build",
                 help="Root for the JIT cache dir (default: ./build)")
  p.add_argument("--max-iter", type=int, default=0,
                 help="Cap fixpoint iterations (0 = unlimited)")
  p.add_argument("--jobs", type=int, default=16,
                 help="Ninja parallel jobs (default: 16)")
  p.add_argument("--no-run", action="store_true",
                 help="Compile the .so but don't invoke the runtime "
                      "(useful for timing / fixture checks)")
  p.add_argument("--no-compile", action="store_true",
                 help="Skip compile; dlopen the existing .so in the cache dir")
  args = p.parse_args()

  project_name = args.project or args.benchmark.capitalize() + "Plan"

  from srdatalog import CompilerConfig, build_project, compile_jit_project
  from srdatalog.runtime import (
    cuda_compile_flags, cuda_include_paths, cuda_libs, cuda_link_flags,
    runtime_defines, runtime_include_paths,
  )

  # ---------- Build DSL program ----------
  t0 = time.time()
  prog, consts = _load_benchmark(args.benchmark, args.meta or None)
  print(f"[{args.benchmark}] program: "
        f"{len(prog.relations)} relations, {len(prog.rules)} rules"
        + (f", {len(consts)} dataset_consts" if consts else "")
        + f" ({time.time() - t0:.1f}s)")

  # Sanity: warn if the benchmark expects CSVs but --data is empty.
  loadable = [r for r in prog.relations if getattr(r, "input_file", "")]
  if loadable and not args.data and not args.no_run:
    print(f"[warn] {args.benchmark} has {len(loadable)} input relations but "
          f"--data is unset; use --data <dir> to populate them.",
          file=sys.stderr)

  # ---------- Emit C++ tree ----------
  t0 = time.time()
  project = build_project(
    prog, project_name=project_name, cache_base=args.cache_base,
  )
  print(f"[emit] {project['dir']} "
        f"(main + {len(project['batches'])} batches, "
        f"{time.time() - t0:.1f}s)")

  # ---------- Compile ----------
  cfg = CompilerConfig(
    include_paths=runtime_include_paths() + cuda_include_paths(),
    defines=runtime_defines(),
    cxx_flags=cuda_compile_flags() + ["-fPIC"],
    link_flags=cuda_link_flags(),
    libs=cuda_libs() + ["boost_container"],
    shared=True,
    jobs=args.jobs,
  )
  if args.no_compile:
    sos = list(Path(project["dir"]).glob("*.so"))
    if not sos:
      print(f"[error] --no-compile but no .so in {project['dir']}",
            file=sys.stderr)
      return 1
    artifact = str(sos[0])
    print(f"[compile] skipped; reusing {artifact}")
  else:
    t0 = time.time()
    br = compile_jit_project(project, cfg)
    elapsed = time.time() - t0
    if not br.ok():
      print(f"[compile] FAILED after {elapsed:.1f}s", file=sys.stderr)
      for r in br.compile_results:
        if r.returncode != 0:
          print((r.stderr or r.stdout)[-3000:], file=sys.stderr)
          break
      return 1
    artifact = br.artifact
    print(f"[compile] {artifact} ({elapsed:.1f}s)")

  if args.no_run:
    return 0

  # ---------- dlopen + bind extern "C" shim ----------
  lib = ctypes.CDLL(artifact, mode=ctypes.RTLD_GLOBAL)
  lib.srdatalog_init.restype = ctypes.c_int
  lib.srdatalog_load_all.restype = ctypes.c_int
  lib.srdatalog_load_all.argtypes = [ctypes.c_char_p]
  lib.srdatalog_run.restype = ctypes.c_int
  lib.srdatalog_run.argtypes = [ctypes.c_ulonglong]
  lib.srdatalog_size.restype = ctypes.c_ulonglong
  lib.srdatalog_size.argtypes = [ctypes.c_char_p]
  lib.srdatalog_shutdown.restype = ctypes.c_int

  if lib.srdatalog_init() != 0:
    print("[run] srdatalog_init failed", file=sys.stderr)
    return 1

  # ---------- Load + run ----------
  if args.data and loadable:
    t0 = time.time()
    rc = lib.srdatalog_load_all(str(args.data).encode())
    if rc != 0:
      print(f"[run] srdatalog_load_all({args.data}) returned {rc}",
            file=sys.stderr)
      return 1
    print(f"[load] {args.data} ({time.time() - t0:.1f}s)")
  elif loadable:
    print(f"[load] skipped — benchmark has input relations but no --data given")

  t0 = time.time()
  rc = lib.srdatalog_run(args.max_iter)
  elapsed = time.time() - t0
  if rc != 0:
    print(f"[run] srdatalog_run returned {rc}", file=sys.stderr)
    # Continue — partial results may still be in the DB.
  print(f"[run] fixpoint finished in {elapsed:.1f}s "
        f"(max_iter={args.max_iter or 'unlimited'})")

  print()
  print("  === Result sizes ===")
  print_rel_sizes = [r for r in prog.relations if getattr(r, "print_size", False)]
  if not print_rel_sizes:
    # Fall back: report all computed relations.
    print_rel_sizes = [r for r in prog.relations
                       if not getattr(r, "input_file", "")]
  for rel in print_rel_sizes:
    size = lib.srdatalog_size(rel.name.encode())
    print(f"  {rel.name:<30} {size:>14}")

  lib.srdatalog_shutdown()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
