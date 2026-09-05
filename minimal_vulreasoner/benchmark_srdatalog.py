#!/usr/bin/env python3
'''Build and benchmark the interval-valued AnalystAt query on SRDatalog.'''

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

import cupy as cp

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
  sys.path.insert(0, str(REPO_SRC))

from srdatalog import CompilerConfig, build_project, compile_jit_project
from srdatalog.runtime import (
  cuda_compile_flags,
  cuda_include_paths,
  cuda_libs,
  cuda_link_flags,
  runtime_defines,
  runtime_include_paths,
)

try:
  from .srdatalog_query import build_analyst_program
  from .stress_workload import (
    decode_float32_bits,
    emit_csv_dataset,
    generate_stress_workload,
    summarize_target_bounds,
    workload_summary,
  )
except ImportError:
  from srdatalog_query import build_analyst_program
  from stress_workload import (
    decode_float32_bits,
    emit_csv_dataset,
    generate_stress_workload,
    summarize_target_bounds,
    workload_summary,
  )


def _compiler_config(jobs: int) -> CompilerConfig:
  return CompilerConfig(
    include_paths=runtime_include_paths() + cuda_include_paths(),
    defines=runtime_defines(),
    cxx_flags=cuda_compile_flags() + ["-fPIC"],
    link_flags=cuda_link_flags(),
    libs=cuda_libs() + ["boost_container"],
    shared=True,
    jobs=jobs,
  )


def _bind(artifact: str):
  lib = ctypes.CDLL(artifact, mode=ctypes.RTLD_GLOBAL)
  lib.srdatalog_init.restype = ctypes.c_int
  lib.srdatalog_load_all.argtypes = [ctypes.c_char_p]
  lib.srdatalog_load_all.restype = ctypes.c_int
  lib.srdatalog_run.argtypes = [ctypes.c_ulonglong]
  lib.srdatalog_run.restype = ctypes.c_int
  lib.srdatalog_dev_count.argtypes = [ctypes.c_char_p]
  lib.srdatalog_dev_count.restype = ctypes.c_ulonglong
  lib.srdatalog_dev_ptr.argtypes = [ctypes.c_char_p, ctypes.c_uint]
  lib.srdatalog_dev_ptr.restype = ctypes.c_void_p
  lib.srdatalog_shutdown.restype = ctypes.c_int
  return lib


def _copy_analyst_rows(lib) -> list[tuple[int, int, float, float]]:
  count = int(lib.srdatalog_dev_count(b"AnalystAt"))
  host_columns = []
  for column in range(4):
    pointer = int(lib.srdatalog_dev_ptr(b"AnalystAt", column))
    memory = cp.cuda.UnownedMemory(pointer, count * 4, lib)
    device = cp.ndarray(
      (count,),
      dtype=cp.uint32,
      memptr=cp.cuda.MemoryPointer(memory, 0),
    )
    host_columns.append(device.get())
  cp.cuda.get_current_stream().synchronize()
  return [
    (
      int(host_columns[0][row]),
      int(host_columns[1][row]),
      decode_float32_bits(int(host_columns[2][row])),
      decode_float32_bits(int(host_columns[3][row])),
    )
    for row in range(count)
  ]


def _parse_shapes(args) -> list[tuple[int, int, int]]:
  if not args.case:
    return [(args.depth, args.width, args.fanout)]
  shapes = []
  for raw in args.case:
    try:
      values = tuple(int(value) for value in raw.split(","))
    except ValueError as exc:
      raise SystemExit(f"invalid --case {raw!r}; expected DEPTH,WIDTH,FANOUT") from exc
    if len(values) != 3:
      raise SystemExit(f"invalid --case {raw!r}; expected DEPTH,WIDTH,FANOUT")
    shapes.append(values)
  return shapes


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--depth", type=int, default=4)
  parser.add_argument("--width", type=int, default=4)
  parser.add_argument("--fanout", type=int, default=2)
  parser.add_argument("--case", action="append", default=[], metavar="DEPTH,WIDTH,FANOUT")
  parser.add_argument("--repeat", type=int, default=1)
  parser.add_argument("--cache-base", default="./build")
  parser.add_argument("--data-dir", default="./build/vulreasoner_stress")
  parser.add_argument("--jobs", type=int, default=8)
  parser.add_argument("--no-compile", action="store_true")
  args = parser.parse_args()

  emit_start = time.perf_counter()
  project = build_project(
    build_analyst_program(),
    "VulReasonerPlan",
    cache_base=args.cache_base,
  )
  emit_seconds = time.perf_counter() - emit_start

  compile_seconds = 0.0
  if args.no_compile:
    artifacts = list(Path(project["dir"]).glob("*.so"))
    if not artifacts:
      raise SystemExit(f"no cached shared library in {project['dir']}")
    artifact = str(artifacts[0].resolve())
  else:
    compile_start = time.perf_counter()
    build = compile_jit_project(project, _compiler_config(args.jobs))
    compile_seconds = time.perf_counter() - compile_start
    if not build.ok():
      failure = next(result for result in build.compile_results if result.returncode)
      raise SystemExit((failure.stderr or failure.stdout)[-12000:])
    artifact = str(Path(build.artifact).resolve())

  lib = _bind(artifact)
  try:
    for depth, width, fanout in _parse_shapes(args):
      workload = generate_stress_workload(depth, width, fanout)
      case_dir = Path(args.data_dir) / f"d{depth}_w{width}_f{fanout}"
      emit_csv_dataset(workload, case_dir)
      symbols = json.loads((case_dir / "symbols.json").read_text())

      if lib.srdatalog_init() != 0:
        raise RuntimeError("srdatalog_init failed")
      load_start = time.perf_counter()
      if lib.srdatalog_load_all(str(case_dir.resolve()).encode()) != 0:
        raise RuntimeError("srdatalog_load_all failed")
      load_seconds = time.perf_counter() - load_start

      for repeat in range(args.repeat):
        run_start = time.perf_counter()
        if lib.srdatalog_run(0) != 0:
          raise RuntimeError("srdatalog_run failed")
        run_seconds = time.perf_counter() - run_start

        query_start = time.perf_counter()
        rows = _copy_analyst_rows(lib)
        by_key = {(node, time_): (lower, upper) for node, time_, lower, upper in rows}
        target_bounds = {
          target: by_key[(symbols[target], depth)] for target in workload.target_nodes
        }
        query_seconds = time.perf_counter() - query_start
        print(
          json.dumps(
            {
              **workload_summary(workload),
              "repeat": repeat,
              "emit_seconds": emit_seconds,
              "compile_seconds": compile_seconds,
              "load_seconds": load_seconds,
              "run_seconds": run_seconds,
              "query_seconds": query_seconds,
              "analyst_rows": len(rows),
              **summarize_target_bounds(target_bounds),
            },
            sort_keys=True,
          ),
          flush=True,
        )
      lib.srdatalog_shutdown()
  finally:
    sys.stdout.flush()
    sys.stderr.flush()

  # CUDA/RMM owns process-global static resources.  A ctypes-loaded DSO can be
  # unloaded after those resources, causing a teardown-only crash.  The DB has
  # already been explicitly shut down, so exit without dlclose destructors.
  os._exit(0)


if __name__ == "__main__":
  raise SystemExit(main())
